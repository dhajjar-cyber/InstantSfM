import numpy as np
import time
import os
import cv2
import glob

from instantsfm.scene.defs import Images, ImagePair, Cameras, ConfigurationType, CameraModelId, ViewGraph
from instantsfm.utils.database import COLMAPDatabase, blob_to_array
from instantsfm.utils.depth_sample import sample_depth_at_pixel

class PathInfo:
    def __init__(self):
        self.image_path = ""
        self.database_path = ""
        self.output_path = ""
        self.database_exists = False
        self.depth_path = ""
        self.record_path = ""

def ReadData(path) -> PathInfo:
    path_info = PathInfo()
    if os.path.exists(os.path.join(path, 'images')):
        # COLMAP format
        path_info.image_path = os.path.join(path, 'images')
    elif os.path.exists(os.path.join(path, 'color')):  
        # ScanNet format
        path_info.image_path = os.path.join(path, 'color')
    else:
        # used in camera_per_folder mode
        path_info.image_path = path
    
    path_info.database_path = os.path.join(path, 'database.db')
    path_info.output_path = os.path.join(path, 'sparse')
    path_info.database_exists = os.path.exists(path_info.database_path)
    if os.path.exists(os.path.join(path, 'depth')):
        path_info.depth_path = os.path.join(path, 'depth')
    path_info.record_path = os.path.join(path, 'record')

    return path_info

def ReadColmapDatabase(path):
    print(f"Reading COLMAP database from {path}...")
    start_time = time.time()
    view_graph = ViewGraph()
    db = COLMAPDatabase.connect(path)
    
    print("Loading images from database...")
    # Read images into temporary dict for initial processing
    # Create temporary image data structures
    images_dict = {}
    for id, filename, cam_id in db.execute("SELECT image_id, name, camera_id FROM images"):
        images_dict[id] = {
            'id': id,
            'filename': filename,
            'cam_id': cam_id,
            'features': np.array([]),
            'is_registered': False,
            'cluster_id': -1,
            'world2cam': np.eye(4),
            'depths': np.array([]),
            'features_undist': np.array([]),
            'point3d_ids': [],
            'num_points3d': 0,
            'partner_ids': {}
        }
    print(f"Loaded {len(images_dict)} images.")

    # group images by their folder names
    image_folders = {}
    for image_data in images_dict.values():
        folder_name = os.path.dirname(image_data['filename'])
        if folder_name not in image_folders:
            image_folders[folder_name] = []
        image_folders[folder_name].append(image_data)

    # Sort images in each folder to ensure alignment
    for folder_name in image_folders:
        image_folders[folder_name].sort(key=lambda x: x['filename'])

    # Find the minimum number of images across all folders to prevent IndexError
    if image_folders:
        min_len = min(len(folder) for folder in image_folders.values())
        if any(len(folder) > min_len for folder in image_folders.values()):
            print(f"Warning: Image folders have different lengths. Truncating to minimum length: {min_len}")
            for folder_name in image_folders:
                image_folders[folder_name] = image_folders[folder_name][:min_len]
        else:
            print(f"All image folders have equal length: {min_len}")

    print("Loading cameras from database...")
    # Create temporary camera data structures
    camera_records = {}
    for id, model_id, width, height, params, prior_focal_length in db.execute("SELECT * FROM cameras"):
        camera_records[id] = {
            'id': id,
            'model_id': CameraModelId(model_id),
            'width': width,
            'height': height,
            'params': blob_to_array(params, np.float64),
            'has_prior_focal_length': prior_focal_length > 0
        }
    print(f"Loaded {len(camera_records)} cameras.")
    
    print("Loading keypoints...")
    keypoints = [(image_id, blob_to_array(data, np.float32, (-1, cols)))
                 for image_id, cols, data in db.execute("SELECT image_id, cols, data FROM keypoints") if not data is None]
    for image_id, data in keypoints:
        images_dict[image_id]['features'] = data[:, :2]
    print(f"Loaded keypoints for {len(keypoints)} images.")

    print("Loading matches and geometries (this may take a while)...")
    query = """
    SELECT m.pair_id, m.data, t.config, t.F, t.E, t.H
    FROM matches AS m
    INNER JOIN two_view_geometries AS t ON m.pair_id = t.pair_id
    """
    matches_and_geometries = db.execute(query)
    image_pairs = {}
    invalid_count = 0
    
    # Convert cursor to list to get length for progress logging
    matches_list = list(matches_and_geometries)
    total_matches = len(matches_list)
    print(f"Found {total_matches} match pairs. Processing...")

    rejection_reasons = {
        'UNDEFINED': 0,
        'DEGENERATE': 0,
        'WATERMARK': 0,
        'MULTIPLE': 0,
        'DATA_NONE': 0
    }

    for idx, group in enumerate(matches_list):
        if idx % 1000 == 0 and idx > 0:
            print(f"Processed {idx}/{total_matches} pairs...")
        pair_id, data, config, F_blob, E_blob, H_blob = group
        if data is None:
            invalid_count += 1
            rejection_reasons['DATA_NONE'] += 1
            continue
        data = blob_to_array(data, np.uint32, (-1, 2))
        # Convert COLMAP pair_id to image IDs
        image_id2 = pair_id % 2147483647
        image_id1 = (pair_id - image_id2) // 2147483647
        pair_key = (image_id1, image_id2)
        image_pairs[pair_key] = ImagePair(image_id1=image_id1, image_id2=image_id2)
        keypoints1 = images_dict[image_id1]['features']
        keypoints2 = images_dict[image_id2]['features']
        idx1 = data[:, 0]
        idx2 = data[:, 1]
        valid_indices = (idx1 != -1) & (idx2 != -1) & (idx1 < len(keypoints1)) & (idx2 < len(keypoints2))
        valid_matches = data[valid_indices]
        image_pairs[pair_key].matches = valid_matches

        config = ConfigurationType(config)
        image_pairs[pair_key].config = config
        if config in [ConfigurationType.UNDEFINED, ConfigurationType.DEGENERATE, ConfigurationType.WATERMARK, ConfigurationType.MULTIPLE]:
            image_pairs[pair_key].is_valid = False
            invalid_count += 1
            rejection_reasons[config.name] += 1
            continue

        F = blob_to_array(F_blob, np.float64).reshape(3, 3)
        E = blob_to_array(E_blob, np.float64).reshape(3, 3)
        H = blob_to_array(H_blob, np.float64).reshape(3, 3)
        image_pairs[pair_key].F = F
        image_pairs[pair_key].E = E
        image_pairs[pair_key].H = H
        image_pairs[pair_key].config = config

    view_graph.image_pairs = {pair_key: image_pair for pair_key, image_pair in image_pairs.items() if image_pair.is_valid}
    print(f'Pairs read done. {invalid_count} / {len(image_pairs)+invalid_count} are invalid')
    print(f'Rejection breakdown: {rejection_reasons}')

    # Convert dict to Images container with ID remapping
    camera_items = sorted(camera_records.items())
    cam_id2idx = {cam_id: idx for idx, (cam_id, _) in enumerate(camera_items)}
    cameras = Cameras(num_cameras=len(camera_items))
    for idx, (cam_id, cam_data) in enumerate(camera_items):
        # Camera ID is now the same as index, no need to set cameras.ids
        cameras.model_ids[idx] = cam_data['model_id'].value
        cameras.widths[idx] = cam_data['width']
        cameras.heights[idx] = cam_data['height']
        cameras.has_prior_focal_length[idx] = cam_data['has_prior_focal_length']
        cameras.set_params(idx, cam_data['params'], cam_data['model_id'])
    
    img_id2idx = {img_id:idx for idx, img_id in enumerate(images_dict.keys())}
    
    # Create Images container
    images = Images(num_images=len(images_dict))
    for idx, (img_id, image_data) in enumerate(sorted(images_dict.items())):
        images.ids[idx] = img_id2idx[img_id]
        images.cam_ids[idx] = cam_id2idx[image_data['cam_id']]
        images.filenames[idx] = image_data['filename']
        images.is_registered[idx] = image_data['is_registered']
        images.cluster_ids[idx] = image_data['cluster_id']
        images.world2cams[idx] = image_data['world2cam']
        images.features[idx] = image_data['features']
        images.depths[idx] = image_data['depths']
        images.features_undist[idx] = image_data['features_undist']
        images.point3d_ids[idx] = image_data['point3d_ids']
        images.num_points3d[idx] = image_data['num_points3d']
        images.partner_ids[idx] = image_data['partner_ids']
    
    # Update image pair IDs to use the new sequential indices
    updated_pairs = {}
    skipped_pairs_missing = 0
    skipped_pairs_self = 0
    for (old_id1, old_id2), pair in view_graph.image_pairs.items():
        if old_id1 not in img_id2idx or old_id2 not in img_id2idx:
            skipped_pairs_missing += 1
            continue
        new_id1 = img_id2idx[old_id1]
        new_id2 = img_id2idx[old_id2]
        
        # Ensure we don't create self-loops (id1 == id2)
        if new_id1 == new_id2:
            skipped_pairs_self += 1
            continue

        pair.image_id1 = new_id1
        pair.image_id2 = new_id2
        updated_pairs[(new_id1, new_id2)] = pair
    view_graph.image_pairs = updated_pairs
    print(f"Updated pairs: {len(updated_pairs)}. Skipped missing: {skipped_pairs_missing}, Skipped self-loops: {skipped_pairs_self}")

    # assign image partners here
    if image_folders:
        first_folder = list(image_folders.values())[0]
        # Use the truncated length (min_len) which is safe for all folders
        safe_len = len(first_folder) 
        
        for idx in range(safe_len):
            # Create group mapping: folder_name -> image_idx
            # We must ensure the image ID actually exists in our map (it should, since we built map from same data)
            image_group = {}
            for folder_name, folder in image_folders.items():
                img_id = folder[idx]['id']
                if img_id in img_id2idx:
                    image_group[folder_name] = img_id2idx[img_id]
            
            # Assign this group to every image in the group
            for folder in image_folders.values():
                img_id = folder[idx]['id']
                if img_id in img_id2idx:
                    image_idx = img_id2idx[img_id]
                    images.partner_ids[image_idx] = image_group

    print(f'Reading database took: {time.time() - start_time:.2f}')

    try:
        feature_name = db.execute("SELECT feature_name FROM feature_name").fetchone()[0]
    except:
        # if the database does not have feature_name, then assume it's originated from COLMAP-compatibale workflow
        feature_name = 'colmap'

    return view_graph, cameras, images, feature_name

def ReadDepthsIntoFeatures(path, cameras, images):
    depths = ReadDepths(path)
    for i in range(len(images)):
        image = images[i]
        image_id = image.id
        camera = cameras[image.cam_id]
        
        depths_list = []
        for feat in image.features:
            depth, available = sample_depth_at_pixel(depths[image_id], feat, camera.width, camera.height)
            depths_list.append(depth)
        images.depths[i] = np.array(depths_list, dtype=np.float32)

    return depths

def ReadDepths(path):
    depth_files = sorted(glob.glob(os.path.join(path, '*.png')))
    depths = []
    for depth_file in depth_files:
        depth = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED)
        # ScanNet format: depth represents millimeters
        depth = depth.astype(np.float32) / 1000.0
        depths.append(depth)
    return np.array(depths, dtype=np.float32)