import numpy as np

EPSILON = 1e-10

def FilterTracksByAngle(cameras, images, tracks, max_angle_error):
    counter = 0
    thres = np.cos(np.deg2rad(max_angle_error))
    for track_id in range(len(tracks)):
        valid_idx = []
        for idx, (image_id, feature_id) in enumerate(tracks.observations[track_id]):
            world2cam = images.world2cams[image_id]
            feature_undist = images.features_undist[image_id][feature_id]
            pt_calc = world2cam[:3, :3] @ tracks.xyzs[track_id] + world2cam[:3, 3]
            if pt_calc[2] < EPSILON:
                continue
            pt_calc = pt_calc / np.linalg.norm(pt_calc)
            if np.dot(pt_calc, feature_undist) > thres:
                valid_idx.append(idx)
        
        if len(valid_idx) != len(tracks.observations[track_id]):
            counter += 1
            tracks.observations[track_id] = tracks.observations[track_id][valid_idx]
    print(f'Filtered {counter} / {len(tracks)} tracks by angle error')
    return tracks

def FilterTracksByReprojectionNormalized(cameras, images, tracks, max_reprojection_error):
    counter = 0
    image_world2cams = images.world2cams  # (M, 4, 4)
    track_xyzs = np.hstack([tracks.xyzs, np.ones((len(tracks), 1))]) # (N, 4)
    image_ids = []
    
    track_idx = []
    features_undist = []
    for track_id in range(len(tracks)):
        track_obs = tracks.observations[track_id]
        track_idx += [track_id] * track_obs.shape[0]
        image_ids.append(track_obs[:, 0])
        for (image_id, feature_id) in track_obs:
            feature_undist = images.features_undist[image_id][feature_id]
            features_undist.append(feature_undist)

    image_ids = np.concatenate(image_ids) # (X)
    image_world2cams = image_world2cams[image_ids] # (X, 4, 4)
    track_idx = np.array(track_idx) # (X)
    track_xyzs = track_xyzs[track_idx] # (X, 4)
    features_undist = np.array(features_undist) # (X, 3)
    features_undist_reproj = features_undist[:, :2] / (features_undist[:, 2:] + EPSILON) # (X, 2)

    pts_calc = np.einsum('ijk,ik->ij', image_world2cams, track_xyzs) # (X, 4)
    pts_calc = pts_calc[:, :3] # (X, 3)
    valid_mask = pts_calc[:, 2] > EPSILON # (X)
    pts_reproj = pts_calc[:, :2] / (pts_calc[:, 2:] + EPSILON) # (X, 2)
    reprojection_errors = np.linalg.norm(pts_reproj - features_undist_reproj, axis=1) # (X)
    valid_mask = valid_mask & (reprojection_errors < max_reprojection_error) # (X)

    count = 0
    for track_id in range(len(tracks)):
        obs_count = tracks.observations[track_id].shape[0]
        tracks.observations[track_id] = tracks.observations[track_id][valid_mask[count:count + obs_count]]
        count += obs_count
        if not np.all(valid_mask[count:count + obs_count]):
            counter += 1

    print(f'Filtered {counter} / {len(tracks)} tracks by reprojection error')
    return counter

def FilterTracksByReprojection(cameras, images, tracks, max_reprojection_error):
    counter = 0
    image_world2cams = images.world2cams  # (M, 4, 4)
    camera_indices = images.cam_ids  # (M)
    track_xyzs = np.hstack([tracks.xyzs, np.ones((len(tracks), 1))]) # (N, 4)
    image_ids = []
    
    track_idx = []
    features = []
    for track_id in range(len(tracks)):
        track_obs = tracks.observations[track_id]
        track_idx += [track_id] * track_obs.shape[0]
        image_ids.append(track_obs[:, 0])
        for (image_id, feature_id) in track_obs:
            feature = images.features[image_id][feature_id]
            features.append(feature)

    image_ids = np.concatenate(image_ids) # (X)
    image_world2cams = image_world2cams[image_ids] # (X, 4, 4)
    camera_indices = camera_indices[image_ids] # (X)
    track_idx = np.array(track_idx) # (X)
    track_xyzs = track_xyzs[track_idx] # (X, 4)
    features = np.array(features) # (X, 2)

    pts_calc = np.einsum('ijk,ik->ij', image_world2cams, track_xyzs) # (X, 4)
    pts_calc = pts_calc[:, :3] # (X, 3)
    valid_mask = pts_calc[:, 2] > EPSILON # (X)
    pts_reproj = cameras.cam2img(pts_calc, camera_indices)

    reprojection_errors = np.linalg.norm(pts_reproj - features, axis=1) # (X)
    valid_mask = valid_mask & (reprojection_errors < max_reprojection_error) # (X)

    count = 0
    for track_id in range(len(tracks)):
        obs_count = tracks.observations[track_id].shape[0]
        tracks.observations[track_id] = tracks.observations[track_id][valid_mask[count:count + obs_count]]
        count += obs_count
        if not np.all(valid_mask[count:count + obs_count]):
            counter += 1

    print(f'Filtered {counter} / {len(tracks)} tracks by reprojection error')
    return counter

def FilterTracksTriangulationAngle(cameras, images, tracks, min_angle):
    from ..scene.defs import Tracks
    counter = 0
    thres = np.cos(np.deg2rad(min_angle))
    # Batch compute image centers
    Rs = images.world2cams[:, :3, :3]  # (N, 3, 3)
    ts = images.world2cams[:, :3, 3]   # (N, 3)
    image_centers = -np.einsum('nij,nj->ni', Rs.transpose(0, 2, 1), ts)  # (N, 3)

    valid_mask = np.ones(len(tracks), dtype=bool)
    for track_id in range(len(tracks)):
        pts_calc = []
        
        unique_image_ids = np.unique(tracks.observations[track_id][:, 0])
        vectors = tracks.xyzs[track_id] - image_centers[unique_image_ids]
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        pts_calc = vectors / (norms + EPSILON)
        result_matrix = pts_calc @ pts_calc.T

        # If the triangulation angle is too small, mark as invalid
        if np.all(result_matrix > thres):
            valid_mask[track_id] = False
            counter += 1

    # Create filtered tracks container using filter_by_mask
    valid_indices = np.where(valid_mask)[0]
    tracks.filter_by_mask(valid_mask)

    print(f'Filtered {counter} / {counter + len(tracks)} tracks by too small triangulation angle')
    return counter
