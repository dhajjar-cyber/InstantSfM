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

def _check_inlier_num(args):
    pair, min_inlier_num = args
    if not pair.is_valid:
        return 0
    if len(pair.inliers) < min_inlier_num:
        pair.is_valid = False
        return 1
    return 0

def FilterInlierNum(view_graph:ViewGraph, min_inlier_num):
    num_invalid = 0
    pairs = list(view_graph.image_pairs.values())
    num_threads = int(os.environ.get('POSE_ESTIMATION_THREADS', 64))
    print(f"Filtering inlier num with {num_threads} threads...")

    args_list = [(pair, min_inlier_num) for pair in pairs]

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = list(tqdm(executor.map(_check_inlier_num, args_list), total=len(pairs), desc="Filtering Inlier Num", file=sys.stdout))

    num_invalid = sum(results)
    print('Filtered', num_invalid, 'relative pose with inlier number <', min_inlier_num)

def _check_inlier_ratio(args):
    pair, min_inlier_ratio = args
    if not pair.is_valid:
        return 0
    if len(pair.matches) == 0:
        return 0
    if len(pair.inliers) / len(pair.matches) < min_inlier_ratio:
        pair.is_valid = False
        return 1
    return 0

def FilterInlierRatio(view_graph:ViewGraph, min_inlier_ratio):
    num_invalid = 0
    pairs = list(view_graph.image_pairs.values())
    num_threads = int(os.environ.get('POSE_ESTIMATION_THREADS', 64))
    print(f"Filtering inlier ratio with {num_threads} threads...")

    args_list = [(pair, min_inlier_ratio) for pair in pairs]

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = list(tqdm(executor.map(_check_inlier_ratio, args_list), total=len(pairs), desc="Filtering Inlier Ratio", file=sys.stdout))

    num_invalid = sum(results)
    print('Filtered', num_invalid, 'relative pose with inlier ratio <', min_inlier_ratio)