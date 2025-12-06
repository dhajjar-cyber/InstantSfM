import time
import numpy as np
import sys
import pickle
import os
import datetime
import torch

from instantsfm.scene.defs import ViewGraph
from instantsfm.controllers.config import Config
from instantsfm.processors.view_graph_manipulation import UpdateImagePairsConfig, DecomposeRelPose
from instantsfm.processors.view_graph_calibration import SolveViewGraphCalibration, TorchVGC
from instantsfm.processors.image_undistortion import UndistortImages
from instantsfm.processors.relpose_estimation import EstimateRelativePose
from instantsfm.processors.image_pair_inliers import ImagePairInliersCount
from instantsfm.processors.relpose_filter import FilterInlierNum, FilterInlierRatio, FilterRotations
from instantsfm.processors.rotation_averaging import RotationEstimator
from instantsfm.processors.track_establishment import TrackEngine
from instantsfm.processors.reconstruction_normalizer import NormalizeReconstruction
from instantsfm.processors.global_positioning import TorchGP
from instantsfm.processors.track_filter import FilterTracksByAngle, FilterTracksByReprojectionNormalized, FilterTracksTriangulationAngle
from instantsfm.processors.bundle_adjustment import TorchBA
from instantsfm.processors.track_retriangulation import RetriangulateTracks
from instantsfm.processors.reconstruction_pruning import PruneWeaklyConnectedImages

def log(message):
    timestamp = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    print(f"{timestamp} {message}", flush=True)

def save_checkpoint(path, view_graph, cameras, images, tracks=None):
    try:
        log(f"Saving checkpoint to {path}...")
        with open(path, 'wb') as f:
            pickle.dump((view_graph, cameras, images, tracks), f)
        log(f"Successfully saved checkpoint to {path}.")
    except Exception as e:
        log(f"Error saving checkpoint to {path}: {e}")

def load_checkpoint(path):
    try:
        log(f"Loading checkpoint from {path}...")
        with open(path, 'rb') as f:
            data = pickle.load(f)
            if len(data) == 3:
                view_graph, cameras, images = data
                tracks = None
            elif len(data) == 4:
                view_graph, cameras, images, tracks = data
            else:
                raise ValueError("Unknown checkpoint format")
        log(f"Successfully loaded checkpoint from {path}.")
        return view_graph, cameras, images, tracks
    except Exception as e:
        log(f"Error loading checkpoint from {path}: {e}")
        return None, None, None, None

import json

def SolveGlobalMapper(view_graph:ViewGraph, cameras, images, config:Config, depths=None, visualizer=None, tracks=None):    
    # Set PyTorch memory configuration to avoid fragmentation
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    log(f"Starting Global Mapper with {len(images.ids)} images and {len(view_graph.image_pairs)} pairs.")

    log("==> Checking for checkpoint resume...")
    checkpoint_path = config.OPTIONS.get('checkpoint_path', None)
    resume = config.OPTIONS.get('resume_from_checkpoint', False)
    # tracks is now passed as argument, default None

    if resume and checkpoint_path:
        if not os.path.exists(checkpoint_path):
            log(f"Error: Resume requested but checkpoint not found at {checkpoint_path}")
            exit(1)

        loaded_vg, loaded_cams, loaded_imgs, loaded_tracks = load_checkpoint(checkpoint_path)
        if loaded_vg is not None:
            view_graph, cameras, images = loaded_vg, loaded_cams, loaded_imgs
            if loaded_tracks is not None:
                tracks = loaded_tracks
            
            config.OPTIONS['skip_preprocessing'] = True
            config.OPTIONS['skip_view_graph_calibration'] = True
            config.OPTIONS['skip_relative_pose_estimation'] = True
            
            resume_stage = config.OPTIONS.get('resume_stage', 'relpose')
            if resume_stage == 'rotation':
                config.OPTIONS['skip_rotation_averaging'] = True
            elif resume_stage == 'tracks':
                config.OPTIONS['skip_rotation_averaging'] = True
                config.OPTIONS['skip_track_establishment'] = True
                
            log(f"Resuming from checkpoint (stage: {resume_stage}), skipping completed steps.")
        else:
            log("Error: Failed to load checkpoint file.")
            exit(1)

    # --- EXCLUSION LIST HANDLING (Moved after resume) ---
    exclusion_file = config.OPTIONS.get('exclusion_list_path')
    if exclusion_file and os.path.exists(exclusion_file):
        try:
            with open(exclusion_file, 'r') as f:
                exclusion_data = json.load(f)
                excluded_names = set(exclusion_data.get("exclusions", []))
                
            if excluded_names:
                log(f"Found {len(excluded_names)} images in exclusion list: {exclusion_file}")
                count_excluded = 0
                
                # Map filenames to IDs for faster lookup
                name_to_id = {name: i for i, name in enumerate(images.filenames)}
                
                for name in excluded_names:
                    if name in name_to_id:
                        img_id = name_to_id[name]
                        if images.is_registered[img_id]:
                            images.is_registered[img_id] = False
                            count_excluded += 1
                            log(f"  -> Excluding image: {name} (ID: {img_id})")
                        else:
                            log(f"  -> Image {name} already unregistered.")
                    else:
                        log(f"  -> Warning: Excluded image {name} not found in dataset.")
                
                log(f"Successfully excluded {count_excluded} images from optimization.")
        except Exception as e:
            log(f"Error reading exclusion list: {e}")
    elif exclusion_file:
        log(f"Warning: Exclusion list path provided but file not found: {exclusion_file}")
    # -------------------------------

    if not config.OPTIONS['skip_preprocessing']:
        print('-------------------------------------')
        log('Running preprocessing ...')
        print('-------------------------------------')
        sys.stdout.flush()
        start_time = time.time()
        print('Step 1/2: UpdateImagePairsConfig...')
        sys.stdout.flush()
        UpdateImagePairsConfig(view_graph, cameras, images)
        print('Step 2/2: DecomposeRelPose...')
        sys.stdout.flush()
        DecomposeRelPose(view_graph, cameras, images)
        print('Preprocessing took: ', time.time() - start_time)
        sys.stdout.flush()

    if not config.OPTIONS['skip_view_graph_calibration']:
        print('-------------------------------------')
        print('Running view graph calibration (Ceres) ...')
        print('-------------------------------------')
        sys.stdout.flush()
        start_time = time.time()
        # vgc_engine = TorchVGC()
        # vgc_engine.Optimize(view_graph, cameras, images, config.VIEW_GRAPH_CALIBRATOR_OPTIONS)
        SolveViewGraphCalibration(view_graph, cameras, images, config.VIEW_GRAPH_CALIBRATOR_OPTIONS)
        print('View graph calibration took: ', time.time() - start_time)
        sys.stdout.flush()
        
    if not config.OPTIONS['skip_relative_pose_estimation']:
        print('-------------------------------------')
        print('Running relative pose estimation ...')
        print('-------------------------------------')
        sys.stdout.flush()
        start_time = time.time()
        UndistortImages(cameras, images)
        use_poselib = False
        if use_poselib:
            EstimateRelativePose(view_graph, cameras, images, use_poselib)
            ImagePairInliersCount(view_graph, cameras, images, config.INLIER_THRESHOLD_OPTIONS)
        else:
            EstimateRelativePose(view_graph, cameras, images)
        
        print("Filtering pairs by inlier number...")
        sys.stdout.flush()
        FilterInlierNum(view_graph, config.INLIER_THRESHOLD_OPTIONS['min_inlier_num'])
        print("Filtering pairs by inlier ratio...")
        sys.stdout.flush()
        FilterInlierRatio(view_graph, config.INLIER_THRESHOLD_OPTIONS['min_inlier_ratio'])
        print("Keeping largest connected component...")
        sys.stdout.flush()
        view_graph.keep_largest_connected_component(images)
        
        if checkpoint_path:
             save_checkpoint(checkpoint_path, view_graph, cameras, images)

        print('Relative pose estimation took: ', time.time() - start_time)
        sys.stdout.flush()

    if not config.OPTIONS['skip_rotation_averaging']:
        print('-------------------------------------')
        print('Running rotation averaging ...')
        print('-------------------------------------')
        sys.stdout.flush()
        start_time = time.time()
        ra_engine = RotationEstimator()
        ra_engine.EstimateRotations(view_graph, images, config.ROTATION_ESTIMATOR_OPTIONS, config.L1_SOLVER_OPTIONS)
        FilterRotations(view_graph, images, config.INLIER_THRESHOLD_OPTIONS['max_rotation_error'])
        if not view_graph.keep_largest_connected_component(images):
            print('Failed to keep the largest connected component.')
            sys.stdout.flush()
            exit()

        ra_engine.EstimateRotations(view_graph, images, config.ROTATION_ESTIMATOR_OPTIONS, config.L1_SOLVER_OPTIONS)
        FilterRotations(view_graph, images, config.INLIER_THRESHOLD_OPTIONS['max_rotation_error'])
        
        # DEBUG: Check pair status before component pruning
        debug_ids_str = os.environ.get("DEBUG_IMAGE_IDS", "")
        debug_ids = [int(x) for x in debug_ids_str.split(",")] if debug_ids_str else []
        
        for pid, pair in view_graph.image_pairs.items():
            if pair.image_id1 in debug_ids and pair.image_id2 in debug_ids:
                print(f"[DEBUG] Pre-Pruning Pair {pair.image_id1}-{pair.image_id2} Valid: {pair.is_valid}")

        if not view_graph.keep_largest_connected_component(images):
            print('Failed to keep the largest connected component.')
            sys.stdout.flush()
            exit()
        num_img = np.sum(images.is_registered)
        print(num_img, '/', len(images), 'images are within the connected component.')
        print('Rotation averaging took: ', time.time() - start_time)
        sys.stdout.flush()

        save_rot_path = config.OPTIONS.get('save_rotation_checkpoint_path')
        if save_rot_path:
             save_checkpoint(save_rot_path, view_graph, cameras, images)

    if not config.OPTIONS['skip_track_establishment']:
        print('-------------------------------------')
        print('Running track establishment ...')
        print('-------------------------------------')
        sys.stdout.flush()
        start_time = time.time()
        track_engine = TrackEngine(view_graph, images)
        tracks_orig = track_engine.EstablishFullTracks(config.TRACK_ESTABLISHMENT_OPTIONS)
        print('Initialized', len(tracks_orig), 'tracks')
        sys.stdout.flush()

        tracks = track_engine.FindTracksForProblem(tracks_orig, config.TRACK_ESTABLISHMENT_OPTIONS)
        print('Before filtering:', len(tracks_orig), ', after filtering:', len(tracks))
        print('Track establishment took: ', time.time() - start_time)
        sys.stdout.flush()

        save_tracks_path = config.OPTIONS.get('save_tracks_checkpoint_path')
        if save_tracks_path:
             save_checkpoint(save_tracks_path, view_graph, cameras, images, tracks)

    if not config.OPTIONS['skip_global_positioning']:
        print('-------------------------------------')
        log('Running global positioning ...')
        print('-------------------------------------')
        start_time = time.time()
        
        log("Undistorting images...")
        start_undistort = time.time()
        UndistortImages(cameras, images)
        log(f"Undistortion took {time.time() - start_undistort:.4f} seconds")

        gp_engine = TorchGP(visualizer=visualizer)
        
        log("Initializing positions (Spanning Tree)...")
        start_init = time.time()
        # gp_engine.InitializeRandomPositions(cameras, images, tracks, depths)
        gp_engine.InitializePositions(cameras, images, tracks, view_graph, depths)
        log(f"Initialization took {time.time() - start_init:.4f} seconds")
        
        log("Optimizing global positions...")
        # Clear cache before heavy optimization
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        start_opt = time.time()
        gp_engine.Optimize(cameras, images, tracks, depths, config.GLOBAL_POSITIONER_OPTIONS)
        log(f"Optimization took {time.time() - start_opt:.4f} seconds")
        
        log("Filtering tracks by angle...")
        start_filter = time.time()
        tracks = FilterTracksByAngle(cameras, images, tracks, config.INLIER_THRESHOLD_OPTIONS['max_angle_error'])
        log(f"Filtering took {time.time() - start_filter:.4f} seconds")
        
        log("Normalizing reconstruction...")
        NormalizeReconstruction(images, tracks, depths)
        
        log(f'Global positioning took: {time.time() - start_time} seconds')

        save_gp_path = config.OPTIONS.get('save_gp_checkpoint_path')
        if save_gp_path:
             save_checkpoint(save_gp_path, view_graph, cameras, images, tracks)

    if not config.OPTIONS['skip_bundle_adjustment']:
        print('-------------------------------------')
        log('Running bundle adjustment ...')
        if config.BUNDLE_ADJUSTER_OPTIONS.get('enforce_zero_baseline', False):
             log("Rig constraint (zero baseline) is ENABLED.")
        print('-------------------------------------')
        start_time = time.time()
        for iter in range(3):
            log(f"Bundle Adjustment Iteration {iter+1}/3...")
            ba_engine = TorchBA(visualizer=visualizer)
            
            log(f"  Solving BA...")
            # Clear cache before BA
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            start_ba = time.time()
            # Switch to single_only=False to enable Rig BA
            ba_engine.Solve(cameras, images, tracks, config.BUNDLE_ADJUSTER_OPTIONS, single_only=False)
            log(f"  BA Solve took {time.time() - start_ba:.4f} seconds")
            
            log(f"  Undistorting images...")
            UndistortImages(cameras, images)
            
            log(f"  Filtering tracks by reprojection...")
            start_filter = time.time()
            FilterTracksByReprojectionNormalized(cameras, images, tracks, config.INLIER_THRESHOLD_OPTIONS['max_reprojection_error'] * max(1, 3 - iter))
            log(f"  Filtering took {time.time() - start_filter:.4f} seconds")
            
        log(f'{np.sum(images.is_registered)} images are registered after BA.')
            
        log('Filtering tracks')
        UndistortImages(cameras, images)
        FilterTracksByReprojectionNormalized(cameras, images, tracks, config.INLIER_THRESHOLD_OPTIONS['max_reprojection_error'])
        FilterTracksTriangulationAngle(cameras, images, tracks, config.INLIER_THRESHOLD_OPTIONS['min_triangulation_angle'])
        NormalizeReconstruction(images, tracks, depths)
        print('Bundle adjustment took: ', time.time() - start_time)

        save_ba_path = config.OPTIONS.get('save_ba_checkpoint_path')
        if save_ba_path:
             save_checkpoint(save_ba_path, view_graph, cameras, images, tracks)

    if not config.OPTIONS['skip_retriangulation']:
        print('-------------------------------------')
        log('Running retriangulation ...')
        print('-------------------------------------')
        start_time = time.time()
        RetriangulateTracks(cameras, images, tracks, tracks_orig, config.TRIANGULATOR_OPTIONS, config.BUNDLE_ADJUSTER_OPTIONS)

        print('-------------------------------------')
        log('Running bundle adjustment ...')
        print('-------------------------------------')
        ba_engine = TorchBA()
        ba_engine.Solve(cameras, images, tracks, config.BUNDLE_ADJUSTER_OPTIONS, single_only=False)

        # NormalizeReconstruction(images, tracks)
        UndistortImages(cameras, images)
        log('Filtering tracks')
        FilterTracksByReprojectionNormalized(cameras, images, tracks, config.INLIER_THRESHOLD_OPTIONS['max_reprojection_error'])
        FilterTracksTriangulationAngle(cameras, images, tracks, config.INLIER_THRESHOLD_OPTIONS['min_triangulation_angle'])
        log(f'Retriangulation took: {time.time() - start_time:.4f} seconds')

    if not config.OPTIONS['skip_pruning']:
        print('-------------------------------------')
        print('Running postprocessing ...')
        print('-------------------------------------')
        start_time = time.time()
        PruneWeaklyConnectedImages(images, tracks)
        print('Postprocessing took: ', time.time() - start_time)

    return cameras, images, tracks
