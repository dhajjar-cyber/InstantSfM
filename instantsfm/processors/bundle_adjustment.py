import numpy as np
import tqdm
import sys
import time

from instantsfm.utils.cost_function import reproject_funcs
from instantsfm.scene.defs import get_camera_model_info

# used by torch LM
import torch
from torch import nn
import pypose as pp
from pypose.optim.kernel import Huber
from bae.utils.pysolvers import PCG
from bae.utils.ba import rotate_quat
from bae.optim import LM
from bae.autograd.function import TrackingTensor, map_transform

@map_transform
def compute_combined_poses(ref, rel):
    # ref is pp.SE3 (from TrackingTensor)
    # rel is Tensor (from buffer)
    rel_se3 = pp.SE3(rel)
    return ref * rel_se3

class TorchBA():
    def __init__(self, visualizer=None, device="cuda:0"):
        super().__init__()
        self.device = device
        self.visualizer = visualizer

    def Solve(self, cameras, images, tracks, BUNDLE_ADJUSTER_OPTIONS, single_only=False):
        # use an arbitrary image to determine if multi-folder optimization is needed
        is_multi = len(images.partner_ids[0]) > 1
        
        enforce_zero_baseline = BUNDLE_ADJUSTER_OPTIONS.get('enforce_zero_baseline', False)
        
        if is_multi and not single_only:
            print("Invoking Multi-Camera Bundle Adjustment...", flush=True)
            self.SolveMulti(cameras, images, tracks, BUNDLE_ADJUSTER_OPTIONS)
        else:
            print("Invoking Single-Camera Bundle Adjustment...", flush=True)
            if enforce_zero_baseline:
                print("WARNING: enforce_zero_baseline is True, but falling back to single-camera BA.")
                print(f"         is_multi={is_multi}, single_only={single_only}")
                print("         Rig constraints will be IGNORED.")
            self.SolveSingle(cameras, images, tracks, BUNDLE_ADJUSTER_OPTIONS)

    def SolveSingle(self, cameras, images, tracks, BUNDLE_ADJUSTER_OPTIONS):
        self.camera_model = cameras[0].model_id # assume all cameras are under the same model
        self.camera_model_info = get_camera_model_info(self.camera_model)
        try:
            cost_fn = reproject_funcs[self.camera_model.value]
        except:
            raise NotImplementedError("Unsupported camera model")
        class ReprojNonBatched(nn.Module):
            def __init__(self, image_extrs, camera_intrs, points_3d):
                super().__init__()
                self.extrinsics = nn.Parameter(TrackingTensor(image_extrs))  # [num_imgs, 7]
                self.intrinsics = nn.Parameter(TrackingTensor(camera_intrs))  # [num_cams, x], x is the number of intrinsics (excluding pp)
                self.extrinsics.requires_grad_(BUNDLE_ADJUSTER_OPTIONS['optimize_poses'])
                self.extrinsics.trim_SE3_grad = True
                self.points_3d = nn.Parameter(TrackingTensor(points_3d))  # [num_pts, 3]

            def forward(self, points_2d, image_indices, camera_indices, point_indices, camera_pps):
                points_3d = self.points_3d
                points_proj = cost_fn(points_3d[point_indices], self.extrinsics[image_indices], self.intrinsics[camera_indices], camera_pps[camera_indices])
                loss = points_proj - points_2d
                return loss
            
        @torch.no_grad()
        def update(cameras, images, tracks, points_3d, 
                   image_idx2id, camera_idx2id,
                   image_extrs, camera_intrs, camera_pps, remaining_indices, pp_indices):
            # Batch update track positions
            tracks.xyzs[:] = points_3d.detach().cpu().numpy()
            
            # Batch update image poses
            image_extrs_np = pp.SE3(image_extrs).matrix().cpu().numpy()
            for idx in range(image_extrs_np.shape[0]):
                image_id = image_idx2id[idx]
                images.world2cams[image_id] = image_extrs_np[idx]
            
            max_rem = int(torch.max(remaining_indices).item()) if remaining_indices.numel() > 0 else -1
            max_pp = int(torch.max(pp_indices).item()) if pp_indices.numel() > 0 else -1
            full_len = max(max_rem, max_pp) + 1
            camera_params_full = torch.zeros((camera_intrs.shape[0], full_len), dtype=camera_intrs.dtype, device=camera_intrs.device)
            # assign remaining and pp
            camera_params_full[:, remaining_indices] = camera_intrs
            camera_params_full[:, pp_indices] = camera_pps
            camera_params_np = camera_params_full.detach().cpu().numpy()
            for idx in range(camera_params_np.shape[0]):
                cam = cameras[camera_idx2id[idx]]
                cam.set_params(camera_params_np[idx])
        
        # filter out tracks with too few observations
        valid_tracks_mask = np.array([tracks.observations[i].shape[0] >= BUNDLE_ADJUSTER_OPTIONS['min_num_view_per_track'] 
                                      for i in range(len(tracks))], dtype=bool)        
        # Filter tracks in-place to maintain reference semantics
        tracks.filter_by_mask(valid_tracks_mask)
        
        if len(tracks) == 0:
            print("WARNING: No tracks remaining after filtering. Skipping Single-Camera BA.", flush=True)
            return
        
        # filter out images that have no tracks and cameras that have no images
        image_used = np.zeros(len(images), dtype=bool)
        for i in range(len(tracks)):
            unique_image_ids = np.unique(tracks.observations[i][:, 0])
            image_used[unique_image_ids] = True
            if all(image_used):
                break
        images.is_registered[:] = image_used     
        image_id2idx = {}
        image_idx2id = {}
        registered_mask = images.get_registered_mask()
        for image_id in range(len(images)):
            if not registered_mask[image_id]:
                continue
            image_id2idx[image_id] = len(image_id2idx)
            image_idx2id[len(image_idx2id)] = image_id

        camera_used = np.zeros(len(cameras), dtype=bool)
        registered_cam_ids = images.cam_ids[registered_mask]
        camera_used[registered_cam_ids] = True
        camera_id2idx = {}
        camera_idx2id = {}
        for camera_id, camera in enumerate(cameras):
            if not camera_used[camera_id]:
                continue
            camera_id2idx[camera_id] = len(camera_id2idx)
            camera_idx2id[len(camera_idx2id)] = camera_id

        # Batch extract registered image poses
        registered_indices = images.get_registered_indices()
        image_extrs_list = [pp.mat2SE3(images.world2cams[idx]).tensor() for idx in registered_indices]
        image_extrs = torch.stack(image_extrs_list, dim=0).to(self.device).to(torch.float64)
        camera_intrs_list = [torch.tensor(camera.params) for idx, camera in enumerate(cameras) if camera_used[idx]]
        camera_intrs = torch.stack(camera_intrs_list, dim=0).to(self.device).to(torch.float64)

        # because principal point is not optimized, remove it from camera params
        pp_indices = torch.tensor(self.camera_model_info['pp'], device=self.device)
        camera_pps = camera_intrs[..., pp_indices]
        all_indices = torch.arange(camera_intrs.shape[1], device=self.device)
        remaining_indices = torch.tensor([i for i in all_indices if i not in pp_indices], device=self.device)
        camera_intrs = camera_intrs[..., remaining_indices]
        
        # Batch extract track positions
        points_3d = torch.tensor(tracks.xyzs, device=self.device, dtype=torch.float64)

        points_2d_list = []
        image_indices_list = []
        camera_indices_list = []
        point_indices_list = []
        for track_id in range(len(tracks)):
            track_obs = tracks.observations[track_id]
            for image_id, feature_id in track_obs:
                if not images.is_registered[image_id]:
                    continue
                point2D = images.features[image_id][feature_id]
                points_2d_list.append(point2D)
                image_indices_list.append(image_id2idx[image_id])
                camera_indices_list.append(camera_id2idx[images.cam_ids[image_id]])
                point_indices_list.append(track_id)

        points_2d = torch.tensor(np.array(points_2d_list), dtype=torch.float64, device=self.device)
        image_indices = torch.tensor(np.array(image_indices_list), dtype=torch.int32, device=self.device)
        camera_indices = torch.tensor(np.array(camera_indices_list), dtype=torch.int32, device=self.device)
        point_indices = torch.tensor(np.array(point_indices_list), dtype=torch.int32, device=self.device)

        model = ReprojNonBatched(image_extrs, camera_intrs, points_3d)
        strategy = pp.optim.strategy.TrustRegion(radius=1e4, max=1e10, up=2.0, down=0.5**4)
        sparse_solver = PCG(tol=1e-5)
        huber_kernel = Huber(BUNDLE_ADJUSTER_OPTIONS['thres_loss_function'])
        optimizer = LM(model, strategy=strategy, solver=sparse_solver, kernel=huber_kernel, reject=30)

        input = {
            "points_2d": points_2d,
            "image_indices": image_indices,
            "camera_indices": camera_indices,
            "point_indices": point_indices,
            "camera_pps": camera_pps,
        }

        window_size = 4
        loss_history = []
        progress_bar = tqdm.trange(BUNDLE_ADJUSTER_OPTIONS['max_num_iterations'], file=sys.stdout)
        for _ in progress_bar:
            loss = optimizer.step(input)
            loss_history.append(loss.item())
            if len(loss_history) >= 2*window_size:
                avg_recent = np.mean(loss_history[-window_size:])
                avg_previous = np.mean(loss_history[-2*window_size:-window_size])
                improvement = (avg_previous - avg_recent) / avg_previous
                if abs(improvement) < BUNDLE_ADJUSTER_OPTIONS['function_tolerance']:
                    break
                if loss_history[-1] == loss_history[-2]: # no improvement likely because linear solver failed
                    break
            progress_bar.set_postfix({"loss": loss.item()})

            if self.visualizer:
                update(cameras, images, tracks, points_3d,
                       image_idx2id, camera_idx2id,
                       image_extrs, camera_intrs, camera_pps, remaining_indices, pp_indices)
                self.visualizer.add_step(cameras, images, tracks, "bundle_adjustment")
            
        progress_bar.close()
        update(cameras, images, tracks, points_3d,
               image_idx2id, camera_idx2id,
               image_extrs, camera_intrs, camera_pps, remaining_indices, pp_indices)

    def SolveMulti(self, cameras, images, tracks, BUNDLE_ADJUSTER_OPTIONS):
        self.camera_model = cameras[0].model_id # assume all cameras are under the same model
        self.camera_model_info = get_camera_model_info(self.camera_model)
        try:
            cost_fn = reproject_funcs[self.camera_model.value]
        except:
            raise NotImplementedError("Unsupported camera model")
        class ReprojNonBatched(nn.Module):
            def __init__(self, camera_intrs, points_3d, ref_poses, rel_poses, fix_rel_poses=True):
                super().__init__()
                self.intrs = nn.Parameter(TrackingTensor(camera_intrs))  # [num_cams, x]
                self.points_3d = nn.Parameter(TrackingTensor(points_3d))  # [num_pts, 3]
                
                # Conditionally make ref_poses a Parameter
                if BUNDLE_ADJUSTER_OPTIONS.get('optimize_poses', True):
                    self.ref_poses = nn.Parameter(TrackingTensor(pp.SE3(ref_poses)))
                    self.ref_poses.trim_SE3_grad = True
                else:
                    self.ref_poses = pp.SE3(ref_poses)

                self.fix_rel_poses = fix_rel_poses
                if self.fix_rel_poses:
                    # Register as buffer -> Not in model.parameters() -> Ignored by Optimizer
                    self.register_buffer('rel_poses_data', rel_poses)
                else:
                    # Register as Parameter -> Optimized
                    self.rel_poses = nn.Parameter(TrackingTensor(pp.SE3(rel_poses)))
                    self.rel_poses.trim_SE3_grad = True

            def forward(self, points_2d, camera_indices, grouping_indices, point_indices, camera_pps):
                group_idx = grouping_indices[:, 0]
                member_idx = grouping_indices[:, 1]
                
                # ref_poses is a TrackingTensor (N, 7)
                ref_vec = self.ref_poses[group_idx]
                
                if self.fix_rel_poses:
                    # rel_poses_data is a Tensor (N, 7) - constant
                    rel_vec = self.rel_poses_data[member_idx]
                else:
                    # rel_poses is a Parameter (N, 7) - optimized
                    rel_vec = self.rel_poses[member_idx]
                
                # Use map_transform helper to combine poses while preserving graph
                image_poses = compute_combined_poses(ref_vec, rel_vec)
                
                points_3d = self.points_3d
                points_proj = cost_fn(points_3d[point_indices], image_poses,
                                      self.intrs[camera_indices], camera_pps[camera_indices])
                loss = points_proj - points_2d
                return loss
                # Or rather, TrackingTensor wraps the underlying data.
                
                # If ref_poses is a Parameter(TrackingTensor(pp.SE3(...))), accessing it gives a TrackingTensor.
                # We need to convert it back to SE3 to call .matrix(), OR use pypose functional API.
                
                # Option 1: Convert to SE3 (this might re-introduce the graph issue if not careful)
                # ref_se3 = pp.SE3(ref_poses[group_idx])
                # ref_mat = ref_se3.matrix()
                
                # Option 2: Use pypose functional
                # ref_mat = pp.matrix(ref_poses[group_idx]) # If supported
                
                # Option 3: If ref_poses is just a tensor of shape (N, 7), we can use pp.SE3(tensor).matrix()
                # But ref_poses is a TrackingTensor.
                
                # Let's try constructing SE3 from the tensor data
                ref_se3 = pp.SE3(ref_poses[group_idx])
                ref_mat = ref_se3.matrix()
                
                rel_mat = rel_poses[member_idx].matrix()
                image_poses_mat = ref_mat @ rel_mat
                image_poses = pp.mat2SE3(image_poses_mat)
                
                points_3d = self.points_3d
                points_proj = cost_fn(points_3d[point_indices], image_poses,
                                      self.intrs[camera_indices], camera_pps[camera_indices])
                loss = points_proj - points_2d
                return loss
            
        @torch.no_grad()
        def update(cameras, images, tracks, points_3d, 
                   camera_intrs, camera_pps, remaining_indices, pp_indices,
                   image_group_idx, image_member_idx,
                   ref_poses, rel_poses):
            # Batch update track positions
            tracks.xyzs[:] = points_3d.detach().cpu().numpy()
            
            # Batch update image poses
            ref_poses_mats = pp.SE3(ref_poses).matrix().cpu().numpy()
            rel_poses_mats = pp.SE3(rel_poses).matrix().cpu().numpy()
            for image_id in range(len(images)):
                gid = image_group_idx[image_id]
                midx = image_member_idx[image_id]
                images.world2cams[image_id] = ref_poses_mats[gid] @ rel_poses_mats[midx]
            
            max_rem = int(torch.max(remaining_indices).item()) if remaining_indices.numel() > 0 else -1
            max_pp = int(torch.max(pp_indices).item()) if pp_indices.numel() > 0 else -1
            full_len = max(max_rem, max_pp) + 1
            camera_params_full = torch.zeros((camera_intrs.shape[0], full_len), dtype=camera_intrs.dtype, device=camera_intrs.device)
            # assign remaining and pp
            camera_params_full[:, remaining_indices] = camera_intrs
            camera_params_full[:, pp_indices] = camera_pps
            camera_params_np = camera_params_full.detach().cpu().numpy()
            for idx, cam in enumerate(cameras):
                cam.set_params(camera_params_np[idx])

        # --- SAFEGUARD 1: Subsampling ---
        # filter out tracks with too few observations
        print(f"Filtering {len(tracks)} tracks by min_num_view_per_track={BUNDLE_ADJUSTER_OPTIONS['min_num_view_per_track']}...", flush=True)
        valid_tracks_mask = np.array([tracks.observations[i].shape[0] >= BUNDLE_ADJUSTER_OPTIONS['min_num_view_per_track'] 
                                      for i in range(len(tracks))], dtype=bool)
        
        tracks.filter_by_mask(valid_tracks_mask)
        print(f"Tracks remaining after filtering: {len(tracks)}", flush=True)

        if len(tracks) == 0:
            print("WARNING: No tracks remaining after filtering. Skipping Multi-Camera BA.", flush=True)
            return

        # Subsample if too many tracks to prevent OOM
        max_tracks = BUNDLE_ADJUSTER_OPTIONS.get('max_tracks_for_ba', 200000)
        if len(tracks) > max_tracks:
            print(f"Subsampling tracks from {len(tracks)} to {max_tracks} to prevent OOM...", flush=True)
            indices = np.random.choice(len(tracks), max_tracks, replace=False)
            keep_mask = np.zeros(len(tracks), dtype=bool)
            keep_mask[indices] = True
            tracks.filter_by_mask(keep_mask)
            print(f"Tracks remaining after subsampling: {len(tracks)}", flush=True)

        # Build grouping (rows and columns) from partner_ids if available, similar to global_positioning
        registered_indices = images.get_registered_indices()
        partner_dicts = [images.partner_ids[idx] for idx in registered_indices]
        
        # 1. Find all unique folder keys across all registered images
        all_keys = set()
        for d in partner_dicts:
            all_keys.update(d.keys())
        folder_keys = sorted(list(all_keys))
        folder_key_to_idx = {k: i for i, k in enumerate(folder_keys)}

        # 2. Group images into rows (Rig Poses)
        rows = {}  # row_key (tuple of sorted image IDs) -> gid
        row_dicts = {} # gid -> partner_dict
        gid = 0
        
        registered_mask = images.get_registered_mask()
        for image_id in range(len(images)):
            if not registered_mask[image_id]:
                continue
            partner_dict = images.partner_ids[image_id]
            # Unique key for this group of images
            row_key = tuple(sorted(partner_dict.values()))

            if row_key not in rows:
                rows[row_key] = gid
                row_dicts[gid] = partner_dict
                gid += 1

        # 3. Map each image to (gid, column_idx)
        image_group_idx = {}
        image_member_idx = {}
        
        for gid, p_dict in row_dicts.items():
            for folder, img_id in p_dict.items():
                if img_id in image_group_idx: continue # Already processed
                
                col_idx = folder_key_to_idx[folder]
                image_group_idx[img_id] = gid
                image_member_idx[img_id] = col_idx

        num_groups = len(rows)
        num_columns = len(folder_keys)

        # 4. Initialize Poses
        enforce_zero_baseline = BUNDLE_ADJUSTER_OPTIONS.get('enforce_zero_baseline', False)
        if enforce_zero_baseline:
            print("Enforcing zero baseline (translation) for rig cameras in BA.")

        # Initialize Relative Poses (T_rig_cam)
        rel_poses_init = [None] * num_columns
        rel_poses_init[0] = torch.tensor([0, 0, 0, 0, 0, 0, 1], dtype=torch.float64, device=self.device) 
        
        ref_key = folder_keys[0]
        
        for c in range(1, num_columns):
            target_key = folder_keys[c]
            deltas = []
            
            for rid in range(num_groups):
                p_dict = row_dicts[rid]
                if ref_key in p_dict and target_key in p_dict:
                    id_ref = p_dict[ref_key]
                    id_tgt = p_dict[target_key]
                    
                    pose_ref = pp.mat2SE3(images.world2cams[id_ref])
                    pose_tgt = pp.mat2SE3(images.world2cams[id_tgt])
                    
                    delta = pose_ref.Inv() * pose_tgt
                    
                    if enforce_zero_baseline:
                        d_tensor = delta.tensor()
                        d_tensor[..., :3] = 0
                        deltas.append(d_tensor.to(self.device))
                    else:
                        deltas.append(delta.tensor().to(self.device))
            
            if deltas:
                rel_poses_init[c] = deltas[0]
            else:
                print(f"WARNING: No overlap found between {ref_key} and {target_key}. Initializing to Identity.")
                rel_poses_init[c] = torch.tensor([0, 0, 0, 0, 0, 0, 1], dtype=torch.float64, device=self.device)

        rel_poses = torch.stack(rel_poses_init).to(self.device).to(torch.float64)
        rel_poses.requires_grad_(False)

        # Initialize Group Poses (T_w_rig)
        group_pose_init = []
        for rid in range(num_groups):
            p_dict = row_dicts[rid]
            found = False
            for c, key in enumerate(folder_keys):
                if key in p_dict:
                    img_id = p_dict[key]
                    pose_cam_mat = torch.tensor(images.world2cams[img_id], device=self.device, dtype=torch.float64)
                    pose_cam = pp.mat2SE3(pose_cam_mat) # T_w_cam
                    rel_pose = pp.SE3(rel_poses[c]) # T_rig_cam
                    
                    pose_rig = pose_cam * rel_pose.Inv()
                    group_pose_init.append(pose_rig.tensor())
                    found = True
                    break
            
            if not found:
                group_pose_init.append(torch.tensor([0, 0, 0, 0, 0, 0, 1], dtype=torch.float64, device=self.device))

        ref_poses = torch.stack(group_pose_init, dim=0).to(self.device).to(torch.float64)

        # Build camera intrinsics
        camera_intrs_list = []
        for idx, camera in enumerate(cameras):
            cam_params = torch.tensor(camera.params)
            camera_intrs_list.append(cam_params)
        camera_intrs = torch.stack(camera_intrs_list, dim=0).to(self.device).to(torch.float64)
        
        pp_indices = torch.tensor(self.camera_model_info['pp'], device=self.device)
        camera_pps = camera_intrs[..., pp_indices]
        all_indices = torch.arange(camera_intrs.shape[1], device=self.device)
        remaining_indices = torch.tensor([i for i in all_indices if i not in pp_indices], device=self.device)
        camera_intrs = camera_intrs[..., remaining_indices]
        
        # Batch extract track positions
        points_3d = torch.tensor(tracks.xyzs, device=self.device, dtype=torch.float64)

        # --- SAFEGUARD 2: Vectorization ---
        print("Vectorizing track observations for Multi-Camera BA...", flush=True)
        start_vec = time.time()

        # Pre-compute lookups
        max_img_id = len(images)
        img_to_gid = np.full(max_img_id, -1, dtype=np.int32)
        img_to_midx = np.full(max_img_id, -1, dtype=np.int32)
        img_to_cam_idx = np.full(max_img_id, -1, dtype=np.int32)
        
        for img_id, gid in image_group_idx.items():
            img_to_gid[img_id] = gid
        for img_id, midx in image_member_idx.items():
            img_to_midx[img_id] = midx
        for img_id in range(len(images)):
            if images.is_registered[img_id]:
                img_to_cam_idx[img_id] = images.cam_ids[img_id]

        # 1. Flatten observations
        obs_lengths = np.array([len(o) for o in tracks.observations])
        if len(tracks.observations) > 0:
            all_obs = np.concatenate(tracks.observations)
            all_track_ids = np.repeat(np.arange(len(tracks)), obs_lengths)
        else:
            all_obs = np.zeros((0, 2), dtype=np.int32)
            all_track_ids = np.zeros(0, dtype=np.int32)

        # 2. Filter by registered images
        valid_mask = images.is_registered[all_obs[:, 0]]
        all_obs = all_obs[valid_mask]
        all_track_ids = all_track_ids[valid_mask]

        # 3. Sort by image_id for grouped processing
        sort_idx = np.argsort(all_obs[:, 0])
        all_obs = all_obs[sort_idx]
        all_track_ids = all_track_ids[sort_idx]

        # 4. Extract Features (Points 2D)
        unique_image_ids, split_indices = np.unique(all_obs[:, 0], return_index=True)
        split_indices = np.append(split_indices, len(all_obs))
        
        points_2d_list = []
        for i in range(len(unique_image_ids)):
            image_id = unique_image_ids[i]
            start_idx = split_indices[i]
            end_idx = split_indices[i+1]
            feature_ids = all_obs[start_idx:end_idx, 1]
            
            features = images.features[image_id][feature_ids]
            points_2d_list.append(features)
            
        if points_2d_list:
            points_2d_np = np.concatenate(points_2d_list)
        else:
            points_2d_np = np.zeros((0, 2))

        # 5. Map indices
        image_ids_flat = all_obs[:, 0]
        gids_flat = img_to_gid[image_ids_flat]
        midxs_flat = img_to_midx[image_ids_flat]
        cam_idxs_flat = img_to_cam_idx[image_ids_flat]
        
        grouping_indices_np = np.column_stack([gids_flat, midxs_flat])
        point_indices_np = all_track_ids
        camera_indices_np = cam_idxs_flat

        print(f"Vectorization complete in {time.time() - start_vec:.4f}s. Processing {len(points_2d_np)} observations.", flush=True)

        if len(points_2d_np) == 0:
            print("WARNING: No observations remaining. Skipping optimization.", flush=True)
            return

        points_2d = torch.tensor(points_2d_np, dtype=torch.float64, device=self.device)
        camera_indices = torch.tensor(camera_indices_np, dtype=torch.int32, device=self.device)
        grouping_indices = torch.tensor(grouping_indices_np, dtype=torch.int32, device=self.device)
        point_indices = torch.tensor(point_indices_np, dtype=torch.int32, device=self.device)

        model = ReprojNonBatched(camera_intrs, points_3d, ref_poses, rel_poses, fix_rel_poses=enforce_zero_baseline)
        strategy = pp.optim.strategy.TrustRegion(radius=1e4, max=1e10, up=2.0, down=0.5**4)
        sparse_solver = PCG(tol=1e-5)
        huber_kernel = Huber(BUNDLE_ADJUSTER_OPTIONS['thres_loss_function'])
        optimizer = LM(model, strategy=strategy, solver=sparse_solver, kernel=huber_kernel, reject=30)

        input = {
            "points_2d": points_2d,
            "camera_indices": camera_indices,
            "grouping_indices": grouping_indices,
            "point_indices": point_indices,
            "camera_pps": camera_pps,
        }

        window_size = 4
        loss_history = []
        max_iter = BUNDLE_ADJUSTER_OPTIONS['max_num_iterations']
        print(f"Starting Multi-Camera BA with {max_iter} iterations...", flush=True)
        
        for i in range(max_iter):
            iter_start = time.time()
            print(f"  [BA-Multi] Iteration {i+1}/{max_iter} started...", flush=True)
            
            loss = optimizer.step(input)
            
            iter_time = time.time() - iter_start
            loss_val = loss.item()
            print(f"  [BA-Multi] Iteration {i+1} completed in {iter_time:.2f}s. Loss: {loss_val:.6f}", flush=True)
            
            loss_history.append(loss_val)
            if len(loss_history) >= 2*window_size:
                avg_recent = np.mean(loss_history[-window_size:])
                avg_previous = np.mean(loss_history[-2*window_size:-window_size])
                improvement = (avg_previous - avg_recent) / avg_previous
                if abs(improvement) < BUNDLE_ADJUSTER_OPTIONS['function_tolerance']:
                    print(f"  [BA-Multi] Converged at iteration {i+1} (improvement {improvement:.2e} < {BUNDLE_ADJUSTER_OPTIONS['function_tolerance']})", flush=True)
                    break
                if loss_history[-1] == loss_history[-2]: 
                    print(f"  [BA-Multi] Stalled at iteration {i+1} (no change in loss)", flush=True)
                    break

            if self.visualizer:
                update(cameras, images, tracks, points_3d, 
                       camera_intrs, camera_pps, remaining_indices, pp_indices,
                       image_group_idx, image_member_idx,
                       ref_poses, rel_poses)
                self.visualizer.add_step(cameras, images, tracks, "bundle_adjustment")
            
        update(cameras, images, tracks, points_3d, 
               camera_intrs, camera_pps, remaining_indices, pp_indices,
               image_group_idx, image_member_idx,
               ref_poses, rel_poses)