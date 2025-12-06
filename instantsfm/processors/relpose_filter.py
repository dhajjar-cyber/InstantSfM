import numpy as np
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from instantsfm.scene.defs import ViewGraph

def _check_rotation_pair(args):
    pair, images, max_angle = args
    if not pair.is_valid:
        return 0
    image1 = images[pair.image_id1]
    image2 = images[pair.image_id2]
    if image1.is_registered and image2.is_registered:
        pose1 = image2.world2cam @ np.linalg.inv(image1.world2cam)
        pose2 = pair.get_cam1to2()
        inv_pose1_rot = np.linalg.inv(pose1[:3, :3])
        rotation_matrix = inv_pose1_rot @ pose2[:3, :3]
        trace = np.trace(rotation_matrix)
        cos_r = np.clip((trace - 1) / 2, -1.0, 1.0)
        angle = np.rad2deg(np.arccos(cos_r))
        if angle > max_angle:
            pair.is_valid = False
            return 1
    return 0

def FilterRotations(view_graph:ViewGraph, images, max_angle):
    num_invalid = 0
    pairs = list(view_graph.image_pairs.values())
    num_threads = int(os.environ.get('POSE_ESTIMATION_THREADS', 64))
    print(f"Filtering rotations with {num_threads} threads...")
    
    args_list = [(pair, images, max_angle) for pair in pairs]
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = list(tqdm(executor.map(_check_rotation_pair, args_list), total=len(pairs), desc="Filtering Rotations", file=sys.stdout))
    
    num_invalid = sum(results)
    print('Filtered', num_invalid, 'relative rotation with angle >', max_angle, 'degrees')
    
    # DEBUG: Check if critical pairs survived rotation filtering
    debug_ids_str = os.environ.get("DEBUG_IMAGE_IDS", "")
    debug_ids = [int(x) for x in debug_ids_str.split(",")] if debug_ids_str else []
    
    for p in pairs:
        if p.image_id1 in debug_ids and p.image_id2 in debug_ids:
            status = "PASSED" if p.is_valid else "FAILED"
            print(f"[DEBUG] FilterRotations Pair {p.image_id1}-{p.image_id2} {status}")

def FilterInlierNum(view_graph:ViewGraph, min_inlier_num):
    print(f"Filtering inlier num (Vectorized)...")
    pairs = list(view_graph.image_pairs.values())
    
    # Filter only valid pairs
    valid_pairs = [p for p in pairs if p.is_valid]
    if not valid_pairs:
        return

    # Extract counts
    n_inliers = np.array([len(p.inliers) if p.inliers is not None else 0 for p in valid_pairs])
    
    # Filter
    mask = n_inliers < min_inlier_num
    
    # Apply
    num_invalid = np.count_nonzero(mask)
    bad_indices = np.where(mask)[0]
    
    debug_ids = [253, 1674, 10200]
    
    for idx in bad_indices:
        valid_pairs[idx].is_valid = False

    # Debug logging
    for i, p in enumerate(valid_pairs):
        if p.image_id1 in debug_ids or p.image_id2 in debug_ids:
            status = "PASSED" if not mask[i] else "FAILED"
            print(f"[DEBUG] FilterInlierNum Pair {p.image_id1}-{p.image_id2} {status}: {n_inliers[i]} vs {min_inlier_num}")

    print('Filtered', num_invalid, 'relative pose with inlier number <', min_inlier_num)

def FilterInlierRatio(view_graph:ViewGraph, min_inlier_ratio):
    print(f"Filtering inlier ratio (Vectorized)...")
    pairs = list(view_graph.image_pairs.values())
    
    valid_pairs = [p for p in pairs if p.is_valid]
    if not valid_pairs:
        return

    n_inliers = np.array([len(p.inliers) if p.inliers is not None else 0 for p in valid_pairs])
    n_matches = np.array([len(p.matches) if p.matches is not None else 0 for p in valid_pairs])
    
    with np.errstate(divide='ignore', invalid='ignore'):
        ratios = n_inliers / n_matches
    
    mask = ratios < min_inlier_ratio
    mask[np.isnan(ratios)] = False # 0 matches -> Valid (or at least not filtered by ratio)
    
    num_invalid = np.count_nonzero(mask)
    bad_indices = np.where(mask)[0]
    
    debug_ids = [253, 1674, 10200]
    
    for idx in bad_indices:
        valid_pairs[idx].is_valid = False
    
    # Debug logging
    for i, p in enumerate(valid_pairs):
        if p.image_id1 in debug_ids or p.image_id2 in debug_ids:
            status = "PASSED" if not mask[i] else "FAILED"
            r = ratios[i] if not np.isnan(ratios[i]) else 0.0
            print(f"[DEBUG] FilterInlierRatio Pair {p.image_id1}-{p.image_id2} {status}: {r:.3f} vs {min_inlier_ratio}")

    print('Filtered', num_invalid, 'relative pose with inlier ratio <', min_inlier_ratio)