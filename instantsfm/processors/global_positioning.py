import numpy as np
import tqdm
import sys
import time
import gc

from instantsfm.utils.cost_function import pairwise_cost
from instantsfm.scene.defs import Tracks

# used by torch LM
import torch
from torch import nn
import pypose as pp
from pypose.optim.kernel import Huber
from bae.utils.pysolvers import PCG
from bae.optim import LM
from bae.autograd.function import TrackingTensor

class TorchGP():
    def __init__(self, visualizer=None, device='cuda:0'):
        super().__init__()
        self.device = device
        self.visualizer = visualizer
        
    def InitializePositions(self, cameras, images, tracks, view_graph, depths=None):
        print("Initializing global positions using Spanning Tree (BFS)...")
        
        # 1. Build Adjacency List
        adj = {}
        for pair in view_graph.image_pairs.values():
            if not pair.is_valid:
                continue
            if pair.image_id1 not in adj: adj[pair.image_id1] = []
            if pair.image_id2 not in adj: adj[pair.image_id2] = []
            adj[pair.image_id1].append((pair.image_id2, pair))
            adj[pair.image_id2].append((pair.image_id1, pair))
            
        # 2. Find Root (Max Degree)
        max_deg = -1
        root = -1
        for img_id, neighbors in adj.items():
            if len(neighbors) > max_deg:
                max_deg = len(neighbors)
                root = img_id
                
        if root == -1:
            print("Warning: No valid pairs found for initialization. Falling back to random.")
            self.InitializeRandomPositions(cameras, images, tracks, depths)
            return

        print(f"Selected root image {root} with degree {max_deg}.")
        
        # 3. BFS Initialization
        # Initialize all to 0
        images.world2cams[:, :3, 3] = 0
        visited = set([root])
        queue = [root]
        
        # Set root to origin (or keep its current random/zero if we want)
        # Actually, we should respect the rotation.
        # t_j = t_ij + R_ij * t_i
        # If t_root = 0, then t_j = t_root_j
        
        while queue:
            u = queue.pop(0)
            
            # Current global pose
            # R_u = images.world2cams[u, :3, :3]
            t_u = images.world2cams[u, :3, 3]
            
            if u not in adj: continue
            
            for v, pair in adj[u]:
                if v in visited:
                    continue
                
                # Get relative pose from u to v
                # pair stores id1 -> id2
                # if u == id1, then t_uv = pair.t
                # if u == id2, then t_vu = -R_uv.T * t_uv
                
                # R_uv = R_v * R_u^T
                # t_uv = t_v - R_uv * t_u
                # => t_v = t_uv + R_uv * t_u
                
                if u == pair.image_id1:
                    # Forward: u -> v
                    t_rel = pair.translation
                    q_rel = pair.rotation
                    R_rel = pp.SO3(q_rel).matrix().numpy()
                    
                    # t_v = t_rel + R_rel @ t_u
                    t_v = t_rel + R_rel @ t_u
                else:
                    # Backward: v -> u (stored is u -> v)
                    # We need t_vu and R_vu
                    # R_vu = R_uv.T
                    # t_vu = -R_vu @ t_uv
                    
                    t_uv = pair.translation
                    q_uv = pair.rotation
                    R_uv = pp.SO3(q_uv).matrix().numpy()
                    
                    R_vu = R_uv.T
                    t_vu = -R_vu @ t_uv
                    
                    # t_v = t_vu + R_vu @ t_u
                    t_v = t_vu + R_vu @ t_u
                
                images.world2cams[v, :3, 3] = t_v
                visited.add(v)
                queue.append(v)
                
        print(f"Initialized {len(visited)} images out of {len(images)}.")
        
        # Handle disconnected components (random init for them)
        # But usually we only keep the largest connected component anyway.
        
        # Batch initialize track positions
        # Triangulate tracks based on new camera poses
        print("Triangulating tracks for initialization...")
        # Simple triangulation: average of ray intersections?
        # Or just random around the cameras?
        # Let's stick to random for tracks, or use the existing logic but scaled?
        # The existing logic used scene_scale.
        # Since we used unit steps, the scale is roughly 1.0 per step.
        
        # Let's use the original random init for tracks, but centered?
        # Or better: Leave tracks as random, the optimizer will fix them quickly if cameras are good.
        # But we need to set is_initialized.
        
        scene_scale = 10.0 # Arbitrary
        tracks.xyzs[:] = scene_scale * np.random.uniform(-1, 1, (len(tracks), 3))
        tracks.is_initialized[:] = True

        if self.visualizer:
            self.visualizer.add_step(cameras, images, tracks)

    def InitializeRandomPositions(self, cameras, images, tracks, depths=None):
        # calculate average valid depth to estimate scale of the scene
        scene_scale = 100
        if depths is not None:
            valid_depths = depths[depths > 0]
            if len(valid_depths):
                scene_scale = np.mean(valid_depths) * 4.0

        # Batch initialize image translations
        images.world2cams[:, :3, 3] = scene_scale * np.random.uniform(-1, 1, (len(images), 3))

        # Batch initialize track positions
        tracks.xyzs[:] = scene_scale * np.random.uniform(-1, 1, (len(tracks), 3))
        tracks.is_initialized[:] = True

        if self.visualizer:
            self.visualizer.add_step(cameras, images, tracks)

    def Optimize(self, cameras, images, tracks, depths, GLOBAL_POSITIONER_OPTIONS, single_only=False):
        # use an arbitrary image to determine if multi-folder optimization is needed
        is_multi = len(images.partner_ids[0]) > 1
        if is_multi and not single_only:
            self.OptimizeMulti(cameras, images, tracks, depths, GLOBAL_POSITIONER_OPTIONS)
        else:
            self.OptimizeSingle(cameras, images, tracks, depths, GLOBAL_POSITIONER_OPTIONS)

    def OptimizeSingle(self, cameras, images, tracks, depths, GLOBAL_POSITIONER_OPTIONS):        
        cost_fn = pairwise_cost
        class PairwiseNonBatched(nn.Module):
            def __init__(self, image_translations, points_3d, scales, scale_indices=None):
                super().__init__()
                self.translations = nn.Parameter(TrackingTensor(image_translations))  # [num_cams, 3]
                self.points_3d = nn.Parameter(TrackingTensor(points_3d))  # [num_pts, 3]
                self.scales = nn.Parameter(TrackingTensor(scales))
                if scale_indices is not None:
                    all_indices = torch.arange(scales.shape[0], device=scales.device)
                    self.scales.optimize_indices = all_indices[~torch.isin(all_indices, scale_indices)]

            def forward(self, translations, image_indices, point_indices, is_calibrated):
                image_translations = self.translations
                points_3d = self.points_3d
                loss = cost_fn(points_3d[point_indices], image_translations[image_indices], self.scales, translations, is_calibrated[image_indices])
                return loss

        @torch.no_grad()
        def update(cameras, images, tracks, points_3d, 
                   image_idx2id, image_translations):
            # Batch update track positions
            tracks.xyzs[:] = points_3d.detach().cpu().numpy()
            
            # Batch update image translations for registered images
            image_translations_np = image_translations.detach().cpu().numpy()
            for idx in range(image_translations_np.shape[0]):
                image_id = image_idx2id[idx]
                images.world2cams[image_id, :3, 3] = image_translations_np[idx]
            
            # Transform translations to camera space - ONLY for all images (including unregistered)
            # This matches original logic: all images need the coordinate transform
            for image_id in range(len(images)):
                R = images.world2cams[image_id, :3, :3]
                t = images.world2cams[image_id, :3, 3]
                images.world2cams[image_id, :3, 3] = -(R @ t)

        # filter out tracks with too few observations
        print(f"Filtering {len(tracks)} tracks by min_num_view_per_track={GLOBAL_POSITIONER_OPTIONS['min_num_view_per_track']}...", flush=True)
        valid_mask = np.array([tracks.observations[i].shape[0] >= GLOBAL_POSITIONER_OPTIONS['min_num_view_per_track'] 
                               for i in range(len(tracks))])
        
        # Filter tracks in-place to maintain reference semantics
        tracks.filter_by_mask(valid_mask)
        print(f"Tracks remaining after filtering: {len(tracks)}", flush=True)

        # Subsample if too many tracks to prevent OOM
        max_tracks = GLOBAL_POSITIONER_OPTIONS.get('max_tracks_for_gp', 200000)
        if len(tracks) > max_tracks:
            print(f"Subsampling tracks from {len(tracks)} to {max_tracks} to prevent OOM...", flush=True)
            indices = np.random.choice(len(tracks), max_tracks, replace=False)
            keep_mask = np.zeros(len(tracks), dtype=bool)
            keep_mask[indices] = True
            tracks.filter_by_mask(keep_mask)
            print(f"Tracks remaining after subsampling: {len(tracks)}", flush=True)
        
        # filter out images that have no tracks
        image_used = np.zeros(len(images), dtype=bool)
        for track_id in range(len(tracks)):
            unique_image_ids = np.unique(tracks.observations[track_id][:, 0])
            image_used[unique_image_ids] = True
            if all(image_used):
                break
        images.is_registered[:] = images.is_registered & image_used
        
        image_id2idx = {}
        image_idx2id = {}
        registered_mask = images.get_registered_mask()
        for image_id in range(len(images)):
            if not registered_mask[image_id]:
                continue
            image_id2idx[image_id] = len(image_id2idx)
            image_idx2id[len(image_idx2id)] = image_id
        
        # Batch extract registered image translations
        registered_indices = images.get_registered_indices()
        image_translations = torch.tensor(images.world2cams[registered_indices, :3, 3], 
                                         dtype=torch.float64, device=self.device)
        
        # Batch extract track positions
        points_3d = torch.tensor(tracks.xyzs, dtype=torch.float64, device=self.device)

        print("Vectorizing track observations...", flush=True)
        start_vec = time.time()
        
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
        
        # 4. Process by image groups
        unique_image_ids, split_indices = np.unique(all_obs[:, 0], return_index=True)
        split_indices = np.append(split_indices, len(all_obs))
        
        translations_list = []
        depth_values_list = []
        depth_availability_list = []
        
        for i in range(len(unique_image_ids)):
            image_id = unique_image_ids[i]
            start_idx = split_indices[i]
            end_idx = split_indices[i+1]
            
            # Get feature IDs for this image
            feature_ids = all_obs[start_idx:end_idx, 1]
            
            # Vectorized translation computation
            R = images.world2cams[image_id, :3, :3]
            
            # Get features (N, 2) and make homogeneous (N, 3)
            features_2d = images.features_undist[image_id][feature_ids]
            if features_2d.shape[1] == 2:
                features_3d = np.column_stack([features_2d, np.ones(len(features_2d))])
            else:
                features_3d = features_2d
            
            # Compute translations: (R.T @ f.T).T = f @ R
            translations = features_3d @ R
            translations_list.append(translations)
            
            # Handle depths
            if depths is not None:
                img_depths = images.depths[image_id][feature_ids]
                available = img_depths > 0
                safe_depths = np.where(available, img_depths, 1.0)
                inv_depths = 1.0 / safe_depths
                depth_values_list.append(inv_depths)
                depth_availability_list.append(available)

        # Concatenate results
        if translations_list:
            translations_np = np.concatenate(translations_list)
        else:
            translations_np = np.zeros((0, 3))

        # Create image indices using lookup table
        max_img_id = len(images)
        lookup = np.full(max_img_id, -1, dtype=np.int32)
        lookup[registered_indices] = np.arange(len(registered_indices))
        image_indices_np = lookup[all_obs[:, 0]]
        
        # Point indices are just the track IDs
        point_indices_np = all_track_ids

        print(f"Vectorization complete in {time.time() - start_vec:.4f}s. Processing {len(translations_np)} observations.", flush=True)

        translations = torch.tensor(translations_np, dtype=torch.float64).to(self.device)
        image_indices = torch.tensor(image_indices_np, dtype=torch.int32).to(self.device)
        point_indices = torch.tensor(point_indices_np, dtype=torch.int32).to(self.device)
        is_calibrated = torch.tensor([cameras[images.cam_ids[idx]].has_prior_focal_length 
                                     for idx in registered_indices], 
                                     dtype=torch.bool, device=self.device)
        
        scale_indices = None
        if depths is None:
            scales = torch.ones(len(translations), 1, dtype=torch.float64, device=self.device)
        else:
            depth_values_np = np.concatenate(depth_values_list)
            depth_availability_np = np.concatenate(depth_availability_list)
            scales = torch.tensor(depth_values_np, dtype=torch.float64, device=self.device).unsqueeze(1)
            depth_availability = torch.tensor(depth_availability_np, dtype=torch.bool, device=self.device).unsqueeze(1)
            # indices for optimizer to calculate loss with valid depth scale
            scale_indices = torch.where(depth_availability == 1)[0]

        # Clear memory before building the graph
        gc.collect()
        torch.cuda.empty_cache()

        model = PairwiseNonBatched(image_translations, points_3d, scales, scale_indices=scale_indices)
        strategy = pp.optim.strategy.TrustRegion(radius=1e3, max=1e8, up=2.0, down=0.5**4)
        sparse_solver = PCG(tol=1e-5)
        huber_kernel = Huber(GLOBAL_POSITIONER_OPTIONS['thres_loss_function'])
        optimizer = LM(model, strategy=strategy, solver=sparse_solver, kernel=huber_kernel, reject=30)

        input = {
            "translations": translations,
            "image_indices": image_indices,
            "point_indices": point_indices,
            "is_calibrated": is_calibrated,
        }

        window_size = 4
        loss_history = []
        # progress_bar = tqdm.trange(GLOBAL_POSITIONER_OPTIONS['max_num_iterations'], file=sys.stdout)
        max_iter = GLOBAL_POSITIONER_OPTIONS['max_num_iterations']
        print(f"Starting optimization with {max_iter} iterations...", flush=True)
        
        for i in range(max_iter):
            iter_start = time.time()
            print(f"  [GP] Iteration {i+1}/{max_iter} started...", flush=True)
            
            loss = optimizer.step(input)
            
            iter_time = time.time() - iter_start
            loss_val = loss.item()
            print(f"  [GP] Iteration {i+1} completed in {iter_time:.2f}s. Loss: {loss_val:.6f}", flush=True)
            
            loss_history.append(loss_val)
            if len(loss_history) >= 2*window_size:
                avg_recent = np.mean(loss_history[-window_size:])
                avg_previous = np.mean(loss_history[-2*window_size:-window_size])
                improvement = (avg_previous - avg_recent) / avg_previous
                if abs(improvement) < GLOBAL_POSITIONER_OPTIONS['function_tolerance']:
                    print(f"  [GP] Converged at iteration {i+1} (improvement {improvement:.2e} < {GLOBAL_POSITIONER_OPTIONS['function_tolerance']})", flush=True)
                    break
            # progress_bar.set_postfix({"loss": loss.item()})

            if self.visualizer:
                update(cameras, images, tracks, points_3d, 
                       image_idx2id, image_translations)
                self.visualizer.add_step(cameras, images, tracks, "global_positioning")
            
        # progress_bar.close()
        update(cameras, images, tracks, points_3d, 
               image_idx2id, image_translations)

    def OptimizeMulti(self, cameras, images, tracks, depths, GLOBAL_POSITIONER_OPTIONS):
        cost_fn = pairwise_cost
        class PairwiseNonBatched(nn.Module):
            def __init__(self, points_3d, scales, ref_trans, rel_trans, scale_indices=None, fix_rel_trans=False):
                super().__init__()
                self.points_3d = nn.Parameter(TrackingTensor(points_3d))  # [num_pts, 3]
                self.ref_trans = nn.Parameter(TrackingTensor(ref_trans))
                
                self.fix_rel_trans = fix_rel_trans
                if self.fix_rel_trans:
                    self.register_buffer('rel_trans', TrackingTensor(rel_trans))
                else:
                    self.rel_trans = nn.Parameter(TrackingTensor(rel_trans))

                self.scales = nn.Parameter(TrackingTensor(scales))
                if scale_indices is not None:
                    all_indices = torch.arange(scales.shape[0], device=scales.device)
                    self.scales.optimize_indices = all_indices[~torch.isin(all_indices, scale_indices)]

            def forward(self, translations, grouping_indices, point_indices, is_calibrated):
                group_idx = grouping_indices[:, 0]
                member_idx = grouping_indices[:, 1]
                
                if self.fix_rel_trans:
                    # Disconnect rel_trans from graph -> Zero gradient -> Zero update (due to LM damping)
                    camera_translations = self.ref_trans[group_idx]
                else:
                    camera_translations = self.ref_trans[group_idx] + self.rel_trans[member_idx]
                
                calib_mask = is_calibrated[group_idx]

                points_3d = self.points_3d
                loss = cost_fn(points_3d[point_indices], camera_translations, self.scales, translations, calib_mask)
                return loss
            
        @torch.no_grad()
        def update(cameras, images, tracks, points_3d, 
                   image_idx2id, image_group_idx, image_member_idx, 
                   ref_trans, rel_trans):
            points_3d_np = points_3d.detach().cpu().numpy()
            # Batch update track xyzs
            tracks.xyzs[:] = points_3d_np
            
            # fetch group refs & rel_trans from model and construct per-image translations
            ref_trans_np = ref_trans.detach().cpu().numpy()
            rel_trans_np = rel_trans.detach().cpu().numpy()
            
            # for each registered image, reconstruct camera translation from group params
            for idx in range(len(image_idx2id)):
                image_id = image_idx2id[idx]
                gid = image_group_idx[image_id]
                midx = image_member_idx[image_id]
                cam_trans = ref_trans_np[gid] + rel_trans_np[midx]
                images.world2cams[image_id, :3, 3] = cam_trans
            
            # Convert camera-space translations back to world-space for all images
            for image_id in range(len(images)):
                R = images.world2cams[image_id, :3, :3]
                t = images.world2cams[image_id, :3, 3]
                images.world2cams[image_id, :3, 3] = -(R @ t)

        # filter out tracks with too few observations
        print(f"Filtering {len(tracks)} tracks by min_num_view_per_track={GLOBAL_POSITIONER_OPTIONS['min_num_view_per_track']}...", flush=True)
        valid_mask = np.array([tracks.observations[i].shape[0] >= GLOBAL_POSITIONER_OPTIONS['min_num_view_per_track'] 
                               for i in range(len(tracks))])
        
        # Filter tracks in-place to maintain reference semantics
        tracks.filter_by_mask(valid_mask)
        print(f"Tracks remaining after filtering: {len(tracks)}", flush=True)

        # Subsample if too many tracks to prevent OOM
        max_tracks = GLOBAL_POSITIONER_OPTIONS.get('max_tracks_for_gp', 200000)
        if len(tracks) > max_tracks:
            print(f"Subsampling tracks from {len(tracks)} to {max_tracks} to prevent OOM...", flush=True)
            indices = np.random.choice(len(tracks), max_tracks, replace=False)
            keep_mask = np.zeros(len(tracks), dtype=bool)
            keep_mask[indices] = True
            tracks.filter_by_mask(keep_mask)
            print(f"Tracks remaining after subsampling: {len(tracks)}", flush=True)
        
        # filter out images that have no tracks
        image_used = np.zeros(len(images), dtype=bool)
        for track_id in range(len(tracks)):
            unique_image_ids = np.unique(tracks.observations[track_id][:, 0])
            image_used[unique_image_ids] = True
            if all(image_used):
                break
        images.is_registered[:] = images.is_registered & image_used
        
        image_id2idx = {}
        image_idx2id = {}
        registered_indices = images.get_registered_indices()
        for idx, image_id in enumerate(registered_indices):
            image_id2idx[image_id] = idx
            image_idx2id[idx] = image_id
        
        # Batch create points_3d tensor
        points_3d = torch.tensor(tracks.xyzs, dtype=torch.float64, device=self.device)

        # build grouping as rows (groups) and columns (folder keys)
        partner_dicts = [images.partner_ids[idx] for idx in registered_indices]
        
        # Robustly collect all keys
        all_keys = set()
        for d in partner_dicts:
            all_keys.update(d.keys())
        folder_keys = sorted(list(all_keys))

        # build rows: key by tuple of image ids in folder_keys order (unique per-row)
        rows = {}  # row_key(tuple of ids) -> gid
        row_images = {}  # gid -> list of image ids per column order
        image_group_idx = {}
        image_member_idx = {}
        gid = 0
        for image_id in registered_indices:
            # Handle missing keys with -1
            row_key_list = []
            p_ids = images.partner_ids[image_id]
            for k in folder_keys:
                if k in p_ids:
                    row_key_list.append(int(p_ids[k]))
                else:
                    row_key_list.append(-1)
            row_key = tuple(row_key_list)

            if row_key not in rows:
                rows[row_key] = gid
                row_images[gid] = list(row_key)
                gid += 1

        # now fill image -> (gid, column_idx) mapping
        for rk, gid in rows.items():
            imgs = row_images[gid]
            for cidx, img_id in enumerate(imgs):
                if img_id != -1:
                    image_group_idx[img_id] = gid
                    image_member_idx[img_id] = cidx

        num_groups = len(row_images)
        num_columns = len(folder_keys)

        # initialize ref_trans (per-row) and shared rel_trans (per-column)
        group_refs_init = []
        # rel_trans per column
        rel_accum = [[] for _ in range(num_columns)]
        
        enforce_zero_baseline = GLOBAL_POSITIONER_OPTIONS.get('enforce_zero_baseline', False)
        if enforce_zero_baseline:
            print("Enforcing zero baseline (translation) for rig cameras.")
            
        # Pass 1: Accumulate relative translations (using rows where ref is present)
        for rid in range(num_groups):
            imgs = row_images[rid]
            ref_img_id = imgs[0]
            
            if ref_img_id == -1:
                continue

            ref_trans = images.world2cams[ref_img_id, :3, 3].copy()
            
            # per-column relative
            for cidx in range(num_columns):
                img_id = imgs[cidx]
                if img_id == -1: continue

                if enforce_zero_baseline:
                    # Force relative translation to be zero
                    rel_accum[cidx].append(np.zeros(3))
                else:
                    # Calculate actual relative translation
                    # T_rel = T_img - T_ref (approximate initialization)
                    rel = images.world2cams[img_id, :3, 3] - ref_trans
                    rel_accum[cidx].append(rel)
        
        # average per-column across groups (if no data, zeros)
        rel_trans_init = np.zeros((num_columns, 3), dtype=np.float64)
        for cidx in range(num_columns):
            if len(rel_accum[cidx]) > 0:
                rel_trans_init[cidx] = np.mean(np.stack(rel_accum[cidx], axis=0), axis=0)

        # Pass 2: Initialize group refs (handling missing ref images)
        for rid in range(num_groups):
            imgs = row_images[rid]
            ref_img_id = imgs[0]
            
            if ref_img_id != -1:
                ref_trans = images.world2cams[ref_img_id, :3, 3].copy()
            else:
                # Find fallback
                found = False
                for cidx, img_id in enumerate(imgs):
                    if img_id != -1:
                        # T_ref = T_img - T_rel
                        T_img = images.world2cams[img_id, :3, 3]
                        T_rel = rel_trans_init[cidx]
                        ref_trans = T_img - T_rel
                        found = True
                        break
                if not found:
                    ref_trans = np.zeros(3) # Should not happen if row exists

            group_refs_init.append(ref_trans)

        ref_trans = torch.tensor(np.array(group_refs_init), dtype=torch.float64, device=self.device)
        rel_trans = torch.tensor(rel_trans_init, dtype=torch.float64, device=self.device)

        # Pre-compute image_id -> (gid, midx) array for fast lookup
        max_img_id = len(images)
        img_to_gid = np.full(max_img_id, -1, dtype=np.int32)
        img_to_midx = np.full(max_img_id, -1, dtype=np.int32)
        for img_id, gid in image_group_idx.items():
            img_to_gid[img_id] = gid
        for img_id, midx in image_member_idx.items():
            img_to_midx[img_id] = midx

        print("Vectorizing track observations for Multi-Camera optimization...", flush=True)
        start_vec = time.time()

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

        # 4. Compute Translations
        unique_image_ids, split_indices = np.unique(all_obs[:, 0], return_index=True)
        split_indices = np.append(split_indices, len(all_obs))
        
        translations_list = []
        depth_values_list = []
        depth_availability_list = []

        for i in range(len(unique_image_ids)):
            image_id = unique_image_ids[i]
            start_idx = split_indices[i]
            end_idx = split_indices[i+1]
            
            feature_ids = all_obs[start_idx:end_idx, 1]
            
            # Vectorized translation computation
            R = images.world2cams[image_id, :3, :3]
            features_2d = images.features_undist[image_id][feature_ids]
            if features_2d.shape[1] == 2:
                features_3d = np.column_stack([features_2d, np.ones(len(features_2d))])
            else:
                features_3d = features_2d
            
            translations = features_3d @ R
            translations_list.append(translations)

            if depths is not None:
                img_depths = images.depths[image_id][feature_ids]
                available = img_depths > 0
                safe_depths = np.where(available, img_depths, 1.0)
                inv_depths = 1.0 / safe_depths
                depth_values_list.append(inv_depths)
                depth_availability_list.append(available)

        if translations_list:
            translations_np = np.concatenate(translations_list)
        else:
            translations_np = np.zeros((0, 3))

        # 5. Map to Group/Member indices
        image_ids_flat = all_obs[:, 0]
        gids_flat = img_to_gid[image_ids_flat]
        midxs_flat = img_to_midx[image_ids_flat]
        
        grouping_indices_np = np.column_stack([gids_flat, midxs_flat])
        point_indices_np = all_track_ids

        print(f"Vectorization complete in {time.time() - start_vec:.4f}s. Processing {len(translations_np)} observations.", flush=True)

        translations = torch.tensor(translations_np, dtype=torch.float64, device=self.device)
        grouping_indices = torch.tensor(grouping_indices_np, dtype=torch.int32, device=self.device)
        point_indices = torch.tensor(point_indices_np, dtype=torch.int32, device=self.device)
        is_calibrated = torch.tensor([cameras[images.cam_ids[idx]].has_prior_focal_length 
                                     for idx in registered_indices], 
                                     dtype=torch.bool, device=self.device)
        
        scale_indices = None
        if depths is None:
            scales = torch.ones(len(translations), 1, dtype=torch.float64, device=self.device)
        else:
            depth_values_np = np.concatenate(depth_values_list)
            depth_availability_np = np.concatenate(depth_availability_list)
            scales = torch.tensor(depth_values_np, dtype=torch.float64, device=self.device).unsqueeze(1)
            depth_availability = torch.tensor(depth_availability_np, dtype=torch.bool, device=self.device).unsqueeze(1)
            scale_indices = torch.where(depth_availability == 1)[0]

        # Clear memory
        gc.collect()
        torch.cuda.empty_cache()

        model = PairwiseNonBatched(points_3d, scales, ref_trans, rel_trans, scale_indices=scale_indices, fix_rel_trans=enforce_zero_baseline)
        strategy = pp.optim.strategy.TrustRegion(radius=1e3, max=1e8, up=2.0, down=0.5**4)
        sparse_solver = PCG(tol=1e-5)
        huber_kernel = Huber(GLOBAL_POSITIONER_OPTIONS['thres_loss_function'])
        
        if enforce_zero_baseline:
            print("  (Freezing relative translations by disconnecting from graph)")
        
        optimizer = LM(model, strategy=strategy, solver=sparse_solver, kernel=huber_kernel, reject=30)

        input = {
            "translations": translations,
            "grouping_indices": grouping_indices,
            "point_indices": point_indices,
            "is_calibrated": is_calibrated,
        }
        window_size = 4
        loss_history = []
        max_iter = GLOBAL_POSITIONER_OPTIONS['max_num_iterations']
        print(f"Starting Multi-Camera optimization with {max_iter} iterations...", flush=True)
        
        for i in range(max_iter):
            iter_start = time.time()
            print(f"  [GP-Multi] Iteration {i+1}/{max_iter} started...", flush=True)
            
            loss = optimizer.step(input)
            
            iter_time = time.time() - iter_start
            loss_val = loss.item()
            print(f"  [GP-Multi] Iteration {i+1} completed in {iter_time:.2f}s. Loss: {loss_val:.6f}", flush=True)
            
            loss_history.append(loss_val)
            if len(loss_history) >= 2*window_size:
                avg_recent = np.mean(loss_history[-window_size:])
                avg_previous = np.mean(loss_history[-2*window_size:-window_size])
                improvement = (avg_previous - avg_recent) / avg_previous
                if abs(improvement) < GLOBAL_POSITIONER_OPTIONS['function_tolerance']:
                    print(f"  [GP-Multi] Converged at iteration {i+1} (improvement {improvement:.2e} < {GLOBAL_POSITIONER_OPTIONS['function_tolerance']})", flush=True)
                    break

            if self.visualizer:
                update(cameras, images, tracks, model.points_3d, 
                       image_idx2id, image_group_idx, image_member_idx,
                       model.ref_trans, model.rel_trans)
                self.visualizer.add_step(cameras, images, tracks, "global_positioning")
            
        update(cameras, images, tracks, model.points_3d, 
               image_idx2id, image_group_idx, image_member_idx,
               model.ref_trans, model.rel_trans)