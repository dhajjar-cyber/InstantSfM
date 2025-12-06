import numpy as np
import os
import sqlite3

from instantsfm.scene.reconstruction import Reconstruction

def ExportReconstruction(output_path, cameras, images, tracks, image_path, cluster_id=-1, include_image_points=False, export_txt=False, database_path=None):
    """Export reconstruction using Images/Tracks containers directly.
    
    Args:
        output_path: Output directory path
        cameras: List of camera models
        images: Images container
        tracks: Tracks container
        image_path: Path to image directory for color extraction
        cluster_id: Cluster ID to filter (-1 for all)
        include_image_points: Whether to include image points
        export_txt: Export as text format instead of binary
        database_path: Optional path to COLMAP database for restoring original Image IDs
    """
    # Create reconstruction with containers
    reconstruction = Reconstruction(cameras=cameras, images=images, tracks=tracks)
    
    # Filter by cluster if needed
    if cluster_id != -1:
        reconstruction.filter_by_cluster(cluster_id)
    
    # Filter by registration status
    reconstruction.filter_registered_only()
    
    # Build 2D-3D correspondences
    reconstruction.build_correspondences(min_track_length=3 if not include_image_points else 0)
    
    # Extract colors from images if path provided
    if image_path != "":
        reconstruction.extract_colors_batch(image_path)
    
    # Create output directory
    cluster_path = os.path.join(output_path, '0' if cluster_id == -1 else str(cluster_id))
    os.makedirs(cluster_path, exist_ok=True)

    # Load DB ID mapping if database provided
    db_id_map = None
    if database_path and os.path.exists(database_path):
        print(f"Loading original Image IDs from {database_path}...")
        try:
            conn = sqlite3.connect(database_path)
            cursor = conn.cursor()
            cursor.execute("SELECT name, image_id FROM images")
            db_id_map = {name: img_id for name, img_id in cursor.fetchall()}
            conn.close()
            print(f"Loaded {len(db_id_map)} Image IDs for restoration.")
        except Exception as e:
            print(f"Warning: Failed to load DB IDs: {e}")

    # Write reconstruction
    if export_txt:
        reconstruction.write_text(cluster_path, db_id_map=db_id_map)
    else:
        reconstruction.write_binary(cluster_path, db_id_map=db_id_map)
    
    print(f'Exported {reconstruction.num_points} points and {reconstruction.num_images} images')

def WriteGlomapReconstruction(output_path, cameras, images, tracks, image_path, export_txt=False, database_path=None):
    """Write reconstruction, handling multiple clusters if present.
    
    Args:
        output_path: Output directory path
        cameras: List of camera models
        images: Images container
        tracks: Tracks container
        image_path: Path to image directory
        export_txt: Export as text format instead of binary
        database_path: Optional path to COLMAP database
    """
    # Check for clusters
    if not hasattr(images, 'cluster_ids'):
        # No clustering, export all
        ExportReconstruction(output_path, cameras, images, tracks, image_path, export_txt=export_txt, database_path=database_path)
        return
    
    # Find max cluster ID
    registered_mask = images.is_registered
    if not np.any(registered_mask):
        print("No registered images to export")
        return
    
    cluster_ids = images.cluster_ids[registered_mask]
    unique_clusters = np.unique(cluster_ids)
    
    if len(unique_clusters) == 1:
        # Single cluster, export all
        ExportReconstruction(output_path, cameras, images, tracks, image_path, export_txt=export_txt, database_path=database_path)
    else:
        # Multiple clusters, export separately
        for cluster_id in unique_clusters:
            print(f'Exporting reconstruction for cluster {cluster_id} ({np.sum(cluster_ids == cluster_id)} images)')
            ExportReconstruction(output_path, cameras, images, tracks, 
                                 image_path, cluster_id, export_txt=export_txt, database_path=database_path)