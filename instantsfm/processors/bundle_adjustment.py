import numpy as np
import tqdm

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
from bae.autograd.function import TrackingTensor

class TorchBA():
    def __init__(self, visualizer=None, device="cuda:0"):
        super().__init__()
        self.device = device
        self.visualizer = visualizer

    def Solve(self, cameras, images, tracks, BUNDLE_ADJUSTER_OPTIONS, single_only=True):
        # use an arbitrary image to determine if multi-folder optimization is needed
        is_multi = len(images.partner_ids[0]) > 1
        if is_multi and not single_only:
            self.SolveMulti(cameras, images, tracks, BUNDLE_ADJUSTER_OPTIONS)
        else:
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
        progress_bar = tqdm.trange(BUNDLE_ADJUSTER_OPTIONS['max_num_iterations'])
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
            def __init__(self, camera_intrs, points_3d, ref_poses, rel_poses):
                super().__init__()
                self.intrs = nn.Parameter(TrackingTensor(camera_intrs))  # [num_cams, x], x is the number of intrinsics (excluding principal point)
                self.points_3d = nn.Parameter(TrackingTensor(points_3d))  # [num_pts, 3]
                self.ref_poses = nn.Parameter(TrackingTensor(pp.SE3(ref_poses)))
                self.rel_poses = nn.Parameter(TrackingTensor(pp.SE3(rel_poses)))
                self.ref_poses.requires_grad_(BUNDLE_ADJUSTER_OPTIONS['optimize_poses'])
                self.rel_poses.requires_grad_(BUNDLE_ADJUSTER_OPTIONS['optimize_poses'])
                self.ref_poses.trim_SE3_grad = True
                self.rel_poses.trim_SE3_grad = True

            def forward(self, points_2d, camera_indices, grouping_indices, point_indices, camera_pps):
                group_idx = grouping_indices[:, 0]
                member_idx = grouping_indices[:, 1]
                ref_poses = self.ref_poses
                rel_poses = self.rel_poses
                image_poses = ref_poses[group_idx] * rel_poses[member_idx]
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

        # filter out tracks with too few observations
        valid_tracks_mask = np.array([tracks.observations[i].shape[0] >= BUNDLE_ADJUSTER_OPTIONS['min_num_view_per_track'] 
                                      for i in range(len(tracks))], dtype=bool)
        valid_indices = np.where(valid_tracks_mask)[0]
        
        # Filter tracks in-place to maintain reference semantics
        tracks.filter_by_mask(valid_tracks_mask)

        # Build grouping (rows and columns) from partner_ids if available, similar to global_positioning
        registered_indices = images.get_registered_indices()
        partner_dicts = [images.partner_ids[idx] for idx in registered_indices]
        folder_keys = sorted(list(partner_dicts[0].keys()))

        # build rows: key by tuple of image ids in folder_keys order (unique per-row)
        rows = {}  # row_key(tuple of ids) -> gid
        row_images = {}  # gid -> list of image ids per column order
        image_group_idx = {}
        image_member_idx = {}
        gid = 0
        registered_mask = images.get_registered_mask()
        for image_id in range(len(images)):
            if not registered_mask[image_id]:
                continue
            partner_dict = images.partner_ids[image_id]
            row_key = tuple(int(partner_dict[k]) for k in folder_keys)

            if row_key not in rows:
                rows[row_key] = gid
                row_images[gid] = [int(v) for v in row_key]
                gid += 1

        # now fill image -> (gid, column_idx) mapping
        for rk, gid in rows.items():
            imgs = row_images[gid]
            for cidx, img_id in enumerate(imgs):
                image_group_idx[img_id] = gid
                image_member_idx[img_id] = cidx

        num_groups = len(row_images)
        num_columns = len(folder_keys)

        # initialize group SE3 poses and per-column relative poses
        group_pose_init = []
        rel_accum = [[] for _ in range(num_columns)]
        for rid in range(num_groups):
            imgs = row_images[rid]
            ref_img_id = imgs[0]
            ref_mat = images.world2cams[ref_img_id]
            ref_pose = pp.mat2SE3(ref_mat)
            group_pose_init.append(ref_pose.tensor())
            for cidx in range(num_columns):
                img_id = imgs[cidx]
                # ref_pose*rel_pose = real_pose
                rel_pose = ref_pose.Inv() * pp.mat2SE3(images.world2cams[img_id])
                rel_accum[cidx].append(rel_pose.tensor())

        # as the relative pose will be optimized, select an arbitrary reference
        rel_poses_init = [rel_accum[cidx][0] for cidx in range(num_columns)]

        ref_poses = torch.stack(group_pose_init, dim=0).to(self.device).to(torch.float64)
        rel_poses = torch.tensor(np.array(rel_poses_init), dtype=torch.float64, device=self.device)

        # Build camera intrinsics
        camera_intrs_list = []
        for idx, camera in enumerate(cameras):
            cam_params = torch.tensor(camera.params)
            camera_intrs_list.append(cam_params)
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
        camera_indices_list = []
        image_group_indices_list = []
        image_member_indices_list = []
        point_indices_list = []
        for track_id in range(len(tracks)):
            track_obs = tracks.observations[track_id]
            for image_id, feature_id in track_obs:
                if not images.is_registered[image_id]:
                    continue
                point2D = images.features[image_id][feature_id]
                points_2d_list.append(point2D)
                camera_indices_list.append(images.cam_ids[image_id])

                gid = image_group_idx[image_id]
                midx = image_member_idx[image_id]
                image_group_indices_list.append(gid)
                image_member_indices_list.append(midx)
                point_indices_list.append(track_id)

        points_2d = torch.tensor(np.array(points_2d_list), dtype=torch.float64, device=self.device)
        camera_indices = torch.tensor(np.array(camera_indices_list), dtype=torch.int32, device=self.device)
        grouping_indices = torch.tensor(np.array(list(zip(image_group_indices_list, image_member_indices_list))), dtype=torch.int32, device=self.device)
        point_indices = torch.tensor(np.array(point_indices_list), dtype=torch.int32, device=self.device)

        model = ReprojNonBatched(camera_intrs, points_3d, ref_poses, rel_poses)
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
        progress_bar = tqdm.trange(BUNDLE_ADJUSTER_OPTIONS['max_num_iterations'])
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
                       camera_intrs, camera_pps, remaining_indices, pp_indices,
                       image_group_idx, image_member_idx,
                       ref_poses, rel_poses)
                self.visualizer.add_step(cameras, images, tracks, "bundle_adjustment")
            
        progress_bar.close()
        update(cameras, images, tracks, points_3d, 
               camera_intrs, camera_pps, remaining_indices, pp_indices,
               image_group_idx, image_member_idx,
               ref_poses, rel_poses)