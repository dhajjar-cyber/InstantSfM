import time
import numpy as np
import sys
import pickle
import os

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

def save_checkpoint(path, view_graph, cameras, images, tracks=None):
    try:
        print(f"Saving checkpoint to {path}...")
        sys.stdout.flush()
        with open(path, 'wb') as f:
            pickle.dump((view_graph, cameras, images, tracks), f)
        print(f"Successfully saved checkpoint to {path}.")
        sys.stdout.flush()
    except Exception as e:
        print(f"Error saving checkpoint to {path}: {e}")
        sys.stdout.flush()

def load_checkpoint(path):
    try:
        print(f"Loading checkpoint from {path}...")
        sys.stdout.flush()
        with open(path, 'rb') as f:
            data = pickle.load(f)
            if len(data) == 3:
                view_graph, cameras, images = data
                tracks = None
            elif len(data) == 4:
                view_graph, cameras, images, tracks = data
            else:
                raise ValueError("Unknown checkpoint format")
        print(f"Successfully loaded checkpoint from {path}.")
        sys.stdout.flush()
        return view_graph, cameras, images, tracks
    except Exception as e:
        print(f"Error loading checkpoint from {path}: {e}")
        sys.stdout.flush()
        return None, None, None, None

def SolveGlobalMapper(view_graph:ViewGraph, cameras, images, config:Config, depths=None, visualizer=None):    
    print(f"Starting Global Mapper with {len(images.ids)} images and {len(view_graph.image_pairs)} pairs.")
    sys.stdout.flush()

    checkpoint_path = config.OPTIONS.get('checkpoint_path', None)
    resume = config.OPTIONS.get('resume_from_checkpoint', False)
    tracks = None

    if resume and checkpoint_path:
        if not os.path.exists(checkpoint_path):
            print(f"Error: Resume requested but checkpoint not found at {checkpoint_path}")
            sys.stdout.flush()
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
                
            print(f"Resuming from checkpoint (stage: {resume_stage}), skipping completed steps.")
        else:
            print("Error: Failed to load checkpoint file.")
            sys.stdout.flush()
            exit(1)
        sys.stdout.flush()

    if not config.OPTIONS['skip_preprocessing']:
        print('-------------------------------------')
        print('Running preprocessing ...')
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
        print('Running global positioning ...')
        print('-------------------------------------')
        start_time = time.time()
        UndistortImages(cameras, images)

        gp_engine = TorchGP(visualizer=visualizer)
        gp_engine.InitializeRandomPositions(cameras, images, tracks, depths)
        gp_engine.Optimize(cameras, images, tracks, depths, config.GLOBAL_POSITIONER_OPTIONS)
        tracks = FilterTracksByAngle(cameras, images, tracks, config.INLIER_THRESHOLD_OPTIONS['max_angle_error'])
        NormalizeReconstruction(images, tracks, depths)
        print('Global positioning took: ', time.time() - start_time)

    if not config.OPTIONS['skip_bundle_adjustment']:
        print('-------------------------------------')
        print('Running bundle adjustment ...')
        print('-------------------------------------')
        start_time = time.time()
        for iter in range(3):
            ba_engine = TorchBA(visualizer=visualizer)
            ba_engine.Solve(cameras, images, tracks, config.BUNDLE_ADJUSTER_OPTIONS)
            UndistortImages(cameras, images)
            FilterTracksByReprojectionNormalized(cameras, images, tracks, config.INLIER_THRESHOLD_OPTIONS['max_reprojection_error'] * max(1, 3 - iter))
        print(f'{np.sum(images.is_registered)} images are registered after BA.')
            
        print('Filtering tracks')
        UndistortImages(cameras, images)
        FilterTracksByReprojectionNormalized(cameras, images, tracks, config.INLIER_THRESHOLD_OPTIONS['max_reprojection_error'])
        FilterTracksTriangulationAngle(cameras, images, tracks, config.INLIER_THRESHOLD_OPTIONS['min_triangulation_angle'])
        NormalizeReconstruction(images, tracks, depths)
        print('Bundle adjustment took: ', time.time() - start_time)

    if not config.OPTIONS['skip_retriangulation']:
        print('-------------------------------------')
        print('Running retriangulation ...')
        print('-------------------------------------')
        start_time = time.time()
        RetriangulateTracks(cameras, images, tracks, tracks_orig, config.TRIANGULATOR_OPTIONS, config.BUNDLE_ADJUSTER_OPTIONS)

        print('-------------------------------------')
        print('Running bundle adjustment ...')
        print('-------------------------------------')
        ba_engine = TorchBA()
        ba_engine.Solve(cameras, images, tracks, config.BUNDLE_ADJUSTER_OPTIONS)

        # NormalizeReconstruction(images, tracks)
        UndistortImages(cameras, images)
        print('Filtering tracks')
        FilterTracksByReprojectionNormalized(cameras, images, tracks, config.INLIER_THRESHOLD_OPTIONS['max_reprojection_error'])
        FilterTracksTriangulationAngle(cameras, images, tracks, config.INLIER_THRESHOLD_OPTIONS['min_triangulation_angle'])
        print('Retriangulation took: ', time.time() - start_time)

    if not config.OPTIONS['skip_pruning']:
        print('-------------------------------------')
        print('Running postprocessing ...')
        print('-------------------------------------')
        start_time = time.time()
        PruneWeaklyConnectedImages(images, tracks)
        print('Postprocessing took: ', time.time() - start_time)

    return cameras, images, tracks
