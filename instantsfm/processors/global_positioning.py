import numpy as np
import tqdm


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

    def Optimize(self, cameras, images, tracks, depths, GLOBAL_POSITIONER_OPTIONS, single_only=True):
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
        valid_mask = np.array([tracks.observations[i].shape[0] >= GLOBAL_POSITIONER_OPTIONS['min_num_view_per_track'] 
                               for i in range(len(tracks))])
        
        # Filter tracks in-place to maintain reference semantics
        tracks.filter_by_mask(valid_mask)
        
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

        translations_list = []
        image_indices_list = []
        point_indices_list = []
        depth_values_list = []
        depth_availability_list = []

        for track_id in range(len(tracks)):
            track_obs = tracks.observations[track_id]
            for image_id, feature_id in track_obs:
                if not images.is_registered[image_id]:
                    continue
                if depths is not None:
                    # get depth as scales for optimization
                    depth = images.depths[image_id][feature_id]
                    available = depth
                    depth = depth if available else 1.0 # default value
                    depth_values_list.append(1 / depth) # use inverse depth
                    depth_availability_list.append(available)
                R_transpose = images.world2cams[image_id, :3, :3].T
                feature_undist = images.features_undist[image_id][feature_id]
                translation = R_transpose @ feature_undist
                translations_list.append(translation)
                image_indices_list.append(image_id2idx[image_id])
                point_indices_list.append(track_id)
        
        translations = torch.tensor(np.array(translations_list), dtype=torch.float64).to(self.device)
        image_indices = torch.tensor(np.array(image_indices_list), dtype=torch.int32).to(self.device)
        point_indices = torch.tensor(np.array(point_indices_list), dtype=torch.int32).to(self.device)
        is_calibrated = torch.tensor([cameras[images.cam_ids[idx]].has_prior_focal_length 
                                     for idx in registered_indices], 
                                     dtype=torch.bool, device=self.device)
        
        scale_indices = None
        if depths is None:
            scales = torch.ones(len(translations_list), 1, dtype=torch.float64, device=self.device)
        else:
            scales = torch.tensor(np.array(depth_values_list), dtype=torch.float64, device=self.device).unsqueeze(1)
            depth_availability = torch.tensor(np.array(depth_availability_list), dtype=torch.bool, device=self.device).unsqueeze(1)
            # indices for optimizer to calculate loss with valid depth scale
            scale_indices = torch.where(depth_availability == 1)[0]

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
        progress_bar = tqdm.trange(GLOBAL_POSITIONER_OPTIONS['max_num_iterations'])
        for _ in progress_bar:
            loss = optimizer.step(input)
            loss_history.append(loss.item())
            if len(loss_history) >= 2*window_size:
                avg_recent = np.mean(loss_history[-window_size:])
                avg_previous = np.mean(loss_history[-2*window_size:-window_size])
                improvement = (avg_previous - avg_recent) / avg_previous
                if abs(improvement) < GLOBAL_POSITIONER_OPTIONS['function_tolerance']:
                    break
            progress_bar.set_postfix({"loss": loss.item()})

            if self.visualizer:
                update(cameras, images, tracks, points_3d, 
                       image_idx2id, image_translations)
                self.visualizer.add_step(cameras, images, tracks, "global_positioning")
            
        progress_bar.close()
        update(cameras, images, tracks, points_3d, 
               image_idx2id, image_translations)

    def OptimizeMulti(self, cameras, images, tracks, depths, GLOBAL_POSITIONER_OPTIONS):
        cost_fn = pairwise_cost
        class PairwiseNonBatched(nn.Module):
            def __init__(self, points_3d, scales, ref_trans, rel_trans, scale_indices=None):
                super().__init__()
                self.points_3d = nn.Parameter(TrackingTensor(points_3d))  # [num_pts, 3]
                self.ref_trans = nn.Parameter(TrackingTensor(ref_trans))
                self.rel_trans = nn.Parameter(TrackingTensor(rel_trans))
                self.scales = nn.Parameter(TrackingTensor(scales))
                if scale_indices is not None:
                    all_indices = torch.arange(scales.shape[0], device=scales.device)
                    self.scales.optimize_indices = all_indices[~torch.isin(all_indices, scale_indices)]

            def forward(self, translations, grouping_indices, point_indices, is_calibrated):
                group_idx = grouping_indices[:, 0]
                member_idx = grouping_indices[:, 1]
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
        # filter out tracks with too few observations - modify in-place
        valid_mask = np.array([tracks.observations[i].shape[0] >= GLOBAL_POSITIONER_OPTIONS['min_num_view_per_track'] 
                               for i in range(len(tracks))])
        valid_indices = np.where(valid_mask)[0]
        
        # Filter tracks in-place to maintain reference semantics
        tracks.filter_by_mask(valid_mask)
        
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
        # Each image that has partner_ids contains a dict: folder_name -> image_id
        # Different groups (rows) share the same set of folder names; we treat column 0
        # (first folder in sorted order) as the reference per-group. We create:
        #  - ref_trans: per-row reference translation (G x 3)
        #  - rel_trans: per-column relative translation shared across rows (C x 3)
        # and map each image -> (group_row_idx, column_idx).
        # Collect partner dicts to infer column (folder) ordering
        partner_dicts = [images.partner_ids[idx] for idx in registered_indices]
        folder_keys = sorted(list(partner_dicts[0].keys()))

        # build rows: key by tuple of image ids in folder_keys order (unique per-row)
        rows = {}  # row_key(tuple of ids) -> gid
        row_images = {}  # gid -> list of image ids per column order
        image_group_idx = {}
        image_member_idx = {}
        gid = 0
        for image_id in registered_indices:
            row_key = tuple(int(images.partner_ids[image_id][k]) for k in folder_keys)

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

        # initialize ref_trans (per-row) and shared rel_trans (per-column)
        group_refs_init = []
        # rel_trans per column
        rel_accum = [[] for _ in range(num_columns)]
        for rid in range(num_groups):
            imgs = row_images[rid]
            ref_img_id = imgs[0]
            ref_trans = images.world2cams[ref_img_id, :3, 3].copy()
            group_refs_init.append(ref_trans)
            # per-column relative
            for cidx in range(num_columns):
                img_id = imgs[cidx]
                t = images.world2cams[img_id, :3, 3] - ref_trans
                rel_accum[cidx].append(t)
        # average per-column across groups (if no data, zeros)
        rel_trans_init = np.zeros((num_columns, 3), dtype=np.float64)
        for cidx in range(num_columns):
            rel_trans_init[cidx] = np.mean(np.stack(rel_accum[cidx], axis=0), axis=0)

        ref_trans = torch.tensor(np.array(group_refs_init), dtype=torch.float64, device=self.device)
        rel_trans = torch.tensor(rel_trans_init, dtype=torch.float64, device=self.device)

        translations_list = []
        image_group_indices_list = []
        image_member_indices_list = []
        point_indices_list = []
        depth_values_list = []
        depth_availability_list = []

        for track_id in range(len(tracks)):
            for image_id, feature_id in tracks.observations[track_id]:
                if not images.is_registered[image_id]:
                    continue
                if depths is not None:
                    # get depth as scales for optimization
                    depth = images.depths[image_id][feature_id]
                    available = depth
                    depth = depth if available else 1.0 # default value
                    depth_values_list.append(1 / depth) # use inverse depth
                    depth_availability_list.append(available)
                R = images.world2cams[image_id, :3, :3]
                feature_undist = images.features_undist[image_id][feature_id]
                translation = R.T @ feature_undist
                translations_list.append(translation)
                # map to group and column indices (image ids were used as keys)
                gid = image_group_idx[image_id]
                midx = image_member_idx[image_id]
                image_group_indices_list.append(gid)
                image_member_indices_list.append(midx)
                point_indices_list.append(track_id)

        translations = torch.tensor(np.array(translations_list), dtype=torch.float64, device=self.device)
        grouping_indices = torch.tensor(np.array(list(zip(image_group_indices_list, image_member_indices_list))), dtype=torch.int32, device=self.device)
        point_indices = torch.tensor(np.array(point_indices_list), dtype=torch.int32, device=self.device)
        is_calibrated = torch.tensor([cameras[images.cam_ids[idx]].has_prior_focal_length 
                                     for idx in registered_indices], 
                                     dtype=torch.bool, device=self.device)
        
        scale_indices = None
        if depths is None:
            scales = torch.ones(len(translations_list), 1, dtype=torch.float64, device=self.device)
        else:
            scales = torch.tensor(np.array(depth_values_list), dtype=torch.float64, device=self.device).unsqueeze(1)
            depth_availability = torch.tensor(np.array(depth_availability_list), dtype=torch.bool, device=self.device).unsqueeze(1)
            # indices for optimizer to calculate loss with valid depth scale
            scale_indices = torch.where(depth_availability == 1)[0]

        model = PairwiseNonBatched(points_3d, scales, ref_trans, rel_trans, scale_indices=scale_indices)
        strategy = pp.optim.strategy.TrustRegion(radius=1e3, max=1e8, up=2.0, down=0.5**4)
        sparse_solver = PCG(tol=1e-5)
        huber_kernel = Huber(GLOBAL_POSITIONER_OPTIONS['thres_loss_function'])
        optimizer = LM(model, strategy=strategy, solver=sparse_solver, kernel=huber_kernel, reject=30)

        input = {
            "translations": translations,
            "grouping_indices": grouping_indices,
            "point_indices": point_indices,
            "is_calibrated": is_calibrated,
        }
        window_size = 4
        loss_history = []
        progress_bar = tqdm.trange(GLOBAL_POSITIONER_OPTIONS['max_num_iterations'])
        for _ in progress_bar:
            loss = optimizer.step(input)
            loss_history.append(loss.item())
            if len(loss_history) >= 2*window_size:
                avg_recent = np.mean(loss_history[-window_size:])
                avg_previous = np.mean(loss_history[-2*window_size:-window_size])
                improvement = (avg_previous - avg_recent) / avg_previous
                if abs(improvement) < GLOBAL_POSITIONER_OPTIONS['function_tolerance']:
                    break
            progress_bar.set_postfix({"loss": loss.item()})

            if self.visualizer:
                update(cameras, images, tracks, points_3d, 
                       image_idx2id, image_group_idx, image_member_idx,
                       ref_trans, rel_trans)
                self.visualizer.add_step(cameras, images, tracks, "global_positioning")
            
        progress_bar.close()
        update(cameras, images, tracks, points_3d, 
               image_idx2id, image_group_idx, image_member_idx,
               ref_trans, rel_trans)