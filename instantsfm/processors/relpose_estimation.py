import tqdm
import numpy as np
import os
from scipy.spatial.transform import Rotation as R
import cv2
from concurrent.futures import ThreadPoolExecutor, as_completed

from instantsfm.scene.defs import ConfigurationType, ViewGraph

def estimate_pair_relative_pose_poselib(pair, cameras, images):
    from poselib import estimate_relative_pose

    def to_poselib(cam):
        return {
            'model': 'SIMPLE_RADIAL',
            'width': cam.width,
            'height': cam.height,
            'params': cam.params.tolist()
        }
    image_id1, image_id2 = pair.image_id1, pair.image_id2
    image1 = images[image_id1]
    image2 = images[image_id2]
    matches = pair.matches
    points2d1 = image1.features[matches[:, 0]]
    points2d2 = image2.features[matches[:, 1]]
    relpose = estimate_relative_pose(points2d1, points2d2, to_poselib(cameras[image1.cam_id]), to_poselib(cameras[image2.cam_id]),
                                     {'max_iterations': 50000}, {})
    pair.rotation = np.array([relpose[0].q[1], relpose[0].q[2], relpose[0].q[3], relpose[0].q[0]])
    pair.translation = np.array(relpose[0].t)
    # still need to compute matrices
    '''K1 = cameras[image1.cam_id].get_K()
    E, mask = cv2.findEssentialMat(points2d1, points2d2, cameraMatrix=K1, method=cv2.RANSAC)
    if E is None or E.shape != (3, 3):
        pair.is_valid = False
        return
    inliers = np.where(mask.ravel())[0]
    pair.inliers = inliers'''

def estimate_pair_relative_pose_opencv(pair, cameras, images):
    # DEBUG LOGGING
    debug_ids_str = os.environ.get("DEBUG_IMAGE_IDS", "")
    debug_ids = [int(x) for x in debug_ids_str.split(",")] if debug_ids_str else []
    is_debug = pair.image_id1 in debug_ids or pair.image_id2 in debug_ids
    
    if pair.config not in [ConfigurationType.PLANAR, ConfigurationType.PANORAMIC, ConfigurationType.PLANAR_OR_PANORAMIC, ConfigurationType.UNCALIBRATED, ConfigurationType.CALIBRATED]:
        if is_debug: print(f"[DEBUG] Pair {pair.image_id1}-{pair.image_id2} INVALID CONFIG: {pair.config}")
        pair.is_valid = False
        return

    # FORCE UNCALIBRATED: Ignore COLMAP's geometric classification.
    # We need to find a valid 3D relative pose (E/F) for recoverPose to work.
    # Treating pairs as PLANAR/PANORAMIC but feeding them to recoverPose(E) causes failure.
    if pair.config in [ConfigurationType.PLANAR, ConfigurationType.PANORAMIC, ConfigurationType.PLANAR_OR_PANORAMIC]:
        if is_debug: print(f"[DEBUG] Pair {pair.image_id1}-{pair.image_id2} Overriding config {pair.config} -> UNCALIBRATED")
        pair.config = ConfigurationType.UNCALIBRATED

    image_id1, image_id2 = pair.image_id1, pair.image_id2
    image1 = images[image_id1]
    image2 = images[image_id2]
    cam1 = cameras[image1.cam_id]
    cam2 = cameras[image2.cam_id]
    matches = pair.matches
    
    if is_debug: print(f"[DEBUG] Pair {pair.image_id1}-{pair.image_id2} START. Matches: {len(matches)} Config: {pair.config}")

    points2d1 = image1.features[matches[:, 0]]
    points2d2 = image2.features[matches[:, 1]]
    points2d1_norm = cam1.img2cam(points2d1)
    points2d2_norm = cam2.img2cam(points2d2)
    E, E_mask = cv2.findEssentialMat(points2d1_norm, points2d2_norm, method=cv2.RANSAC, threshold=0.001)
    if E is None or E.shape != (3, 3):
        if is_debug: print(f"[DEBUG] Pair {pair.image_id1}-{pair.image_id2} FAILED Essential Matrix")
        pair.is_valid = False
        return
    if pair.config == ConfigurationType.UNCALIBRATED:
        F, F_mask = cv2.findFundamentalMat(points2d1, points2d2, method=cv2.FM_RANSAC)
        pair.F = F
        if F is None or F.shape != (3, 3):
            if is_debug: print(f"[DEBUG] Pair {pair.image_id1}-{pair.image_id2} FAILED Fundamental Matrix")
            pair.is_valid = False
            return
        mask = F_mask
    elif pair.config == ConfigurationType.CALIBRATED:
        mask = E_mask
    else: # PLANAR, PANORAMIC, PLANAR_OR_PANORAMIC
        H, H_mask = cv2.findHomography(points2d1, points2d2, method=cv2.RANSAC)
        pair.H = H
        if H is None or H.shape != (3, 3):
            if is_debug: print(f"[DEBUG] Pair {pair.image_id1}-{pair.image_id2} FAILED Homography")
            pair.is_valid = False
            return
        mask = H_mask

    inliers = np.where(mask.ravel())[0]
    if is_debug: print(f"[DEBUG] Pair {pair.image_id1}-{pair.image_id2} Geometric Inliers: {len(inliers)}")

    points2d1_norm = points2d1_norm[inliers]
    points2d2_norm = points2d2_norm[inliers]
    inlier_num, R_mat, t, mask = cv2.recoverPose(E, points2d1_norm, points2d2_norm)
    pair.rotation = R.from_matrix(R_mat).as_quat()
    pair.translation = t.flatten()
    pair.E = E
    pair.inliers = inliers[np.where(mask.ravel())[0]] # inliers
    
    if is_debug: print(f"[DEBUG] Pair {pair.image_id1}-{pair.image_id2} Final Inliers (Cheirality): {len(pair.inliers)}")

import sys

def EstimateRelativePose(view_graph: ViewGraph, cameras, images, use_poselib=False):
    valid_pairs = [pair for pair in view_graph.image_pairs.values() if pair.is_valid]
    num_threads = int(os.environ.get('POSE_ESTIMATION_THREADS', 64))
    print(f"Using {num_threads} threads for relative pose estimation.")
    sys.stdout.flush()

    if use_poselib:
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            # Submit tasks with progress bar
            futures = []
            for pair in tqdm.tqdm(valid_pairs, desc="Submitting Tasks", file=sys.stdout):
                futures.append(executor.submit(estimate_pair_relative_pose_poselib, pair, cameras, images))
            
            # Monitor completion
            for _ in tqdm.tqdm(as_completed(futures), total=len(futures), desc="Processing Pairs", file=sys.stdout):
                pass
    else:
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            # Submit tasks with progress bar
            futures = []
            for pair in tqdm.tqdm(valid_pairs, desc="Submitting Tasks", file=sys.stdout):
                futures.append(executor.submit(estimate_pair_relative_pose_opencv, pair, cameras, images))
            
            # Monitor completion
            for _ in tqdm.tqdm(as_completed(futures), total=len(futures), desc="Processing Pairs", file=sys.stdout):
                pass

    print('Estimating relative pose done')
    sys.stdout.flush()