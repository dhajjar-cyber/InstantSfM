import time
import pickle
import os
import sys
from argparse import ArgumentParser

from instantsfm.controllers.config import Config
from instantsfm.controllers.data_reader import ReadData, ReadColmapDatabase, ReadDepthsIntoFeatures, ReadDepths
from instantsfm.controllers.global_mapper import SolveGlobalMapper
from instantsfm.controllers.reconstruction_writer import WriteGlomapReconstruction
from instantsfm.controllers.reconstruction_visualizer import ReconstructionVisualizer

def run_sfm():
    print("DEBUG: Running modified sfm.py from workspace")
    parser = ArgumentParser()
    parser.add_argument('--data_path', required=True, help='Path to the data folder')
    parser.add_argument('--enable_gui', action='store_true', help='Enable GUI for visualization')
    parser.add_argument('--record_recon', action='store_true', help='Save reconstruction data at each step')
    parser.add_argument('--record_path', default=None, help='Path to save the recorded reconstruction data')
    parser.add_argument('--disable_depths', action='store_true', help='Disable the use of depths if available')
    parser.add_argument('--export_txt', action='store_true', help='Export the reconstruction in plain text format')
    parser.add_argument('--manual_config_name', help='Name of the manual configuration file')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint if available')
    parser.add_argument('--checkpoint_path', help='Path to the checkpoint file')
    parser.add_argument('--save_rotation_checkpoint_path', help='Path to save checkpoint after rotation averaging')
    parser.add_argument('--save_tracks_checkpoint_path', help='Path to save checkpoint after track establishment')
    parser.add_argument('--save_gp_checkpoint_path', help='Path to save checkpoint after global positioning')
    parser.add_argument('--save_ba_checkpoint_path', help='Path to save checkpoint after bundle adjustment')
    parser.add_argument('--resume_stage', default='relpose', choices=['relpose', 'rotation', 'tracks', 'gp', 'ba'], help='Stage to resume from (relpose, rotation, tracks, gp, or ba)')
    parser.add_argument('--max_tracks_for_gp', type=int, default=200000, help='Maximum number of tracks to use for global positioning (subsampling)')
    mapper_args = parser.parse_args()

    path_info = ReadData(mapper_args.data_path)
    if not path_info:
        print('Invalid data path, please check the provided path')
        return
    
    # Checkpoint loading logic
    checkpoint_loaded = False
    tracks = None
    if mapper_args.resume and mapper_args.checkpoint_path and os.path.exists(mapper_args.checkpoint_path):
        print(f"==> Loading checkpoint from {mapper_args.checkpoint_path}...")
        sys.stdout.flush()
        try:
            with open(mapper_args.checkpoint_path, 'rb') as f:
                data = pickle.load(f)
                if len(data) == 3:
                    view_graph, cameras, images = data
                elif len(data) == 4:
                    view_graph, cameras, images, tracks = data
                else:
                    raise ValueError("Unknown checkpoint format")
            feature_name = 'colmap' # Default assumption when loading from checkpoint
            checkpoint_loaded = True
            print(f"==> Checkpoint loaded: {len(images.ids)} images, {len(view_graph.image_pairs)} pairs.")
            if tracks is not None:
                print(f"==> Loaded {len(tracks)} tracks from checkpoint.")
            sys.stdout.flush()
        except Exception as e:
            print(f"Failed to load checkpoint: {e}. Falling back to database read.")
            sys.stdout.flush()
            checkpoint_loaded = False

    if not checkpoint_loaded:
        print("==> Reading COLMAP database...")
        sys.stdout.flush()
        view_graph, cameras, images, feature_name = ReadColmapDatabase(path_info.database_path)
        if view_graph is None or cameras is None or images is None:
            return
        print(f"==> Database read: {len(images.ids)} images, {len(view_graph.image_pairs)} pairs.")
        sys.stdout.flush()

    if path_info.depth_path and not mapper_args.disable_depths:
        if checkpoint_loaded:
            print("Loading raw depths for global positioning...")
            depths = ReadDepths(path_info.depth_path)
        else:
            depths = ReadDepthsIntoFeatures(path_info.depth_path, cameras, images)
    else:
        depths = None

    # enable different configs for different feature handlers and image numbers
    print("==> Initializing configuration...")
    sys.stdout.flush()
    start_time = time.time()
    config = Config(feature_name, mapper_args.manual_config_name)
    print("==> Configuration loaded.")
    sys.stdout.flush()
    
    # Override config with command line arguments
    # If we loaded checkpoint externally, we don't want global_mapper to load it again
    config.OPTIONS['resume_from_checkpoint'] = False if checkpoint_loaded else mapper_args.resume
    config.OPTIONS['checkpoint_path'] = mapper_args.checkpoint_path
    config.OPTIONS['save_rotation_checkpoint_path'] = mapper_args.save_rotation_checkpoint_path
    config.OPTIONS['save_tracks_checkpoint_path'] = mapper_args.save_tracks_checkpoint_path
    config.OPTIONS['save_gp_checkpoint_path'] = mapper_args.save_gp_checkpoint_path
    config.OPTIONS['save_ba_checkpoint_path'] = mapper_args.save_ba_checkpoint_path
    config.OPTIONS['resume_stage'] = mapper_args.resume_stage
    
    # If checkpoint was loaded here, set skip flags
    if checkpoint_loaded:
        config.OPTIONS['skip_preprocessing'] = True
        config.OPTIONS['skip_view_graph_calibration'] = True
        config.OPTIONS['skip_relative_pose_estimation'] = True
        if mapper_args.resume_stage == 'rotation':
            config.OPTIONS['skip_rotation_averaging'] = True
        elif mapper_args.resume_stage == 'tracks':
            config.OPTIONS['skip_rotation_averaging'] = True
            config.OPTIONS['skip_track_establishment'] = True
        elif mapper_args.resume_stage == 'gp':
            config.OPTIONS['skip_rotation_averaging'] = True
            config.OPTIONS['skip_track_establishment'] = True
            config.OPTIONS['skip_global_positioning'] = True
        elif mapper_args.resume_stage == 'ba':
            config.OPTIONS['skip_rotation_averaging'] = True
            config.OPTIONS['skip_track_establishment'] = True
            config.OPTIONS['skip_global_positioning'] = True
            config.OPTIONS['skip_bundle_adjustment'] = True

    if mapper_args.enable_gui or mapper_args.record_recon:
        visualizer = ReconstructionVisualizer(save_data=mapper_args.record_recon, 
                                                save_dir=mapper_args.record_path if mapper_args.record_path else path_info.record_path)
    else:
        visualizer = None
    print("==> Starting Global Mapper...")
    sys.stdout.flush()
    cameras, images, tracks = SolveGlobalMapper(view_graph, cameras, images, config, depths=depths, visualizer=visualizer, tracks=tracks)
    print('Reconstruction done in', time.time() - start_time, 'seconds')
    WriteGlomapReconstruction(path_info.output_path, cameras, images, tracks, path_info.image_path, export_txt=mapper_args.export_txt)
    print('Reconstruction written to', path_info.output_path)

    if mapper_args.enable_gui:
        # block until the GUI is closed
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Visualization server terminated by user.")

def entrypoint():
    # Entry point for pyproject.toml
    run_sfm()
    
if __name__ == '__main__':
    entrypoint()

