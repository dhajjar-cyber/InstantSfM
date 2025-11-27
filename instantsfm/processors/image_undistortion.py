import numpy as np

def UndistortImages(cameras, images):
    """Undistort image features using camera models.
    
    Args:
        cameras: List of camera models
        images: Images container
    """
    for image_id in range(len(images)):
        cam = cameras[images.cam_ids[image_id]]
        features = images.features[image_id]
        
        # Convert image coordinates to camera coordinates
        features_undist = cam.img2cam(features)
        features_undist = np.hstack([features_undist, np.ones((features_undist.shape[0], 1))])
        features_undist = features_undist / np.linalg.norm(features_undist, axis=1, keepdims=True)
        
        # Directly update the container
        images.features_undist[image_id] = features_undist