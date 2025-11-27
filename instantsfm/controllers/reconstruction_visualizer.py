import time
import numpy as np
import viser
import threading
from scipy.spatial.transform import Rotation
from typing import Optional, List, Tuple, Dict, Any
import pickle
import os
from pathlib import Path
import datetime
import imageio
import tqdm

from instantsfm.utils.read_write_model import read_points3D_binary

class ReconstructionVisualizer:
    """Visualizer for Structure from Motion reconstruction process.
    
    This visualizer displays the evolving point cloud and camera poses during
    the SfM process, with smooth transitions between updates. Now includes
    save/load functionality for offline playback.
    """
    
    def __init__(
        self,
        min_update_interval: float = 0.1,  # Minimum time between visual updates in seconds
        point_size: float = 0.05,  # Increased default point size
        show_axes: bool = True,
        frustum_scale: float = 0.2,  # Increased default frustum scale
        save_data: bool = False,  # Whether to automatically save reconstruction steps
        save_dir: str = "./recon_records",  # Directory to save data (default: ./recon_records)
    ):
        """Initialize the SfM visualization.
        
        Args:
            min_update_interval: Minimum time between visual updates (seconds)
            point_size: Size of the 3D points
            show_axes: Whether to show coordinate axes
            frustum_scale: Scale of camera frustums
            save_data: Whether to automatically save each step
            save_dir: Directory to save reconstruction data
        """
        self.server = viser.ViserServer()
        self.min_update_interval = min_update_interval
        self.point_size = point_size
        self.show_axes = show_axes
        self.frustum_scale = frustum_scale
        self.save_data = save_data
        
        # Set up save directory
        self.save_dir = Path(save_dir)
        
        if self.save_data:
            self.save_dir.mkdir(exist_ok=True, parents=True)
            # Create a new session directory with timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.session_dir = self.save_dir / f"session_{timestamp}"
            self.session_dir.mkdir(exist_ok=True)
            print(f"Saving reconstruction data to: {self.session_dir}")
        
        # Initialize visualization elements
        self.points_handle: Optional[viser.PointCloudHandle] = None
        self.camera_handles: Dict[int, viser.CameraFrustumHandle] = {}
        
        # For smooth updates
        self.last_update_time = 0.0
        self.pending_update = False
        self.update_lock = threading.Lock()
        
        # Data storage
        self.points: Optional[np.ndarray] = None
        self.colors: Optional[np.ndarray] = None
        self.cameras: Dict[int, Dict[str, Any]] = {}
        
        # Step tracking for saving
        self.current_step = 0
        self.reconstruction_history = []
        
        # Configure visualization
        self._setup_visualization()
        
        # Start update thread
        self.running = True
        self.update_thread = threading.Thread(target=self._update_loop)
        self.update_thread.daemon = True
        self.update_thread.start()
    
    def _setup_visualization(self):
        """Set up the initial visualization elements."""
        # Add world axes
        self.server.scene.world_axes.visible = self.show_axes
        
        # Add GUI elements
        with self.server.gui.add_folder("Visualization Controls"):
            self.gui_point_size = self.server.gui.add_slider(
                "Point Size",
                min=0.001,
                max=0.1,
                step=0.001,
                initial_value=self.point_size,
            )
            
            self.gui_show_cameras = self.server.gui.add_checkbox(
                "Show Cameras", 
                initial_value=True
            )
            
            self.gui_frustum_scale = self.server.gui.add_slider(
                "Frustum Scale",
                min=0.01,
                max=0.5,
                step=0.01,
                initial_value=self.frustum_scale,
            )
        
        # Connect GUI callbacks
        @self.gui_point_size.on_update
        def _(_):
            self.point_size = self.gui_point_size.value
            if self.points_handle is not None:
                self.points_handle.point_size = self.point_size
        
        @self.gui_show_cameras.on_update
        def _(_):
            for camera_id, handle in self.camera_handles.items():
                handle.visible = self.gui_show_cameras.value
        
        @self.gui_frustum_scale.on_update
        def _(_):
            self.frustum_scale = self.gui_frustum_scale.value
            for camera_id, handle in self.camera_handles.items():
                handle.scale = self.frustum_scale
    
    def save_step_data(self, step_data: Dict, step_name = None):
        """Save data for a single step.
        
        Args:
            step_data: Dictionary containing step data
            step_name: Optional name for the step
        """
        if not self.save_data:
            return
        
        step_filename = f"step_{self.current_step:04d}"
        if step_name:
            step_filename += f"_{step_name}"
        step_filename += ".pkl"
        
        step_path = self.session_dir / step_filename
        
        # Add metadata
        step_data_with_meta = {
            'step_number': self.current_step,
            'step_name': step_name,
            'timestamp': datetime.datetime.now().isoformat(),
            'data': step_data
        }
        
        with open(step_path, 'wb') as f:
            pickle.dump(step_data_with_meta, f)
    
    def add_step(
        self,
        cameras,
        images,
        tracks,
        step_name = None,
    ):
        """Add results from a new SfM computation step
        
        Args:
            cameras: List of camera intrinsic objects
            images: List of image objects (containing pose information)
            tracks: List/dictionary of 3D point objects
            step_name: Name of current step (optional)
        """
        # Process point cloud data
        points_list = []
        colors_list = []
        
        # Extract points and colors from tracks
        # Tracks container has direct array access
        for track_id in range(len(tracks)):
            xyz = tracks.xyzs[track_id]
            if xyz is not None:
                points_list.append(xyz)
                color = tracks.colors[track_id]
                if np.any(color):
                    colors_list.append(color)
                else:
                    colors_list.append(np.array([255, 0, 0], dtype=np.uint8))
        
        # Process camera data
        cameras_dict = {}
        
        for i in range(len(images)):
            # Skip unregistered cameras
            if not images.is_registered[i]:
                continue
                
            # Get camera parameters
            world2cam = images.world2cams[i]
            if world2cam is None:
                continue
                
            R = world2cam[:3, :3]
            R = R.T
            t = -R @ world2cam[:3, 3]
            
            # Ensure camera ID exists
            cam_id = images.cam_ids[i]
            if cam_id >= len(cameras):
                continue
                
            cam = cameras[cam_id]
            
            # Calculate FOV
            fov = 2 * np.arctan2(cam.height / 2, cam.focal_length[1]) if hasattr(cam, 'focal_length') else 0.8
            aspect = cam.width / cam.height if hasattr(cam, 'width') and hasattr(cam, 'height') else 1.0
            
            # Create quaternion from rotation matrix
            xyzw = Rotation.from_matrix(R).as_quat()
            wxyz = np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])  # Convert to wxyz format
            
            # Create camera dictionary
            cameras_dict[i] = {
                'position': t,
                'wxyz': wxyz,
                'fov': fov,
                'aspect': aspect,
            }
            
            # Add image data if available (Images container may have image_data list)
            if hasattr(images, 'image_data') and images.image_data is not None and i < len(images.image_data):
                cameras_dict[i]['image'] = images.image_data[i]
        
        # Convert to numpy arrays
        if points_list and colors_list:
            points_array = np.array(points_list, dtype=np.float32)
            colors_array = np.array(colors_list, dtype=np.uint8)
            
            # Save step data if enabled
            if self.save_data:
                step_data = {
                    'cameras': cameras_dict,
                    'serialized_tracks': self._serialize_tracks(tracks),
                }
                self.save_step_data(step_data, step_name)
                self.current_step += 1
            
            # Update visualization through existing add_step method
            with self.update_lock:
                self.points = points_array
                self.colors = colors_array
                self.cameras = cameras_dict
                self.pending_update = True

            return True
        else:
            print("No valid points or cameras to visualize")
            return False
    
    def _serialize_tracks(self, tracks):
        """Serialize track objects for saving."""
        serialized = []
        
        for track_id in range(len(tracks)):
            track_data = {}
            track_data['id'] = track_id
            xyz = tracks.xyzs[track_id]
            if xyz is not None:
                track_data['xyz'] = xyz.tolist()
            color = tracks.colors[track_id]
            if color is not None:
                track_data['color'] = color.tolist()
            serialized.append(track_data)
        
        return serialized

    def _update_visualization(self):
        """Update the visualization with the latest data."""
        if self.points is not None and self.colors is not None:
            # Convert to numpy arrays if they aren't already
            points_array = np.asarray(self.points, dtype=np.float32)
            colors_array = np.asarray(self.colors, dtype=np.uint8)
            
            # Update or create point cloud
            if self.points_handle is None:
                self.points_handle = self.server.scene.add_point_cloud(
                    "pcd",
                    points=points_array,
                    colors=colors_array,
                    point_size=self.point_size,
                    point_shape="circle",
                )
            else:
                self.points_handle.points = points_array
                self.points_handle.colors = colors_array
                self.points_handle.point_size = self.point_size
        
        # Process cameras
        current_camera_ids = set(self.cameras.keys())
        existing_camera_ids = set(self.camera_handles.keys())
        
        # Remove cameras that are no longer present
        for camera_id in existing_camera_ids - current_camera_ids:
            self.camera_handles[camera_id].remove()
            del self.camera_handles[camera_id]
        
        # Update or add cameras
        for camera_id in current_camera_ids:
            camera = self.cameras[camera_id]
            
            if camera_id in self.camera_handles:
                # Update existing camera
                handle = self.camera_handles[camera_id]
                with self.server.atomic():
                    handle.wxyz = camera['wxyz']
                    handle.position = camera['position']
                    handle.scale = self.frustum_scale
                    if 'image' in camera:
                        handle.image = camera['image']
            else:
                frustum = self.server.scene.add_camera_frustum(
                    f"cam_{camera_id}",
                    fov=camera.get('fov', 0.8),
                    aspect=camera.get('aspect', 1.0),
                    scale=self.frustum_scale,
                    wxyz=camera['wxyz'],
                    position=camera['position'],
                    image=camera.get('image', None),
                    visible=self.gui_show_cameras.value,
                )
                self.camera_handles[camera_id] = frustum
        
        # Flush changes to ensure they're sent to clients
        self.server.flush()
    
    def _update_loop(self):
        """Background thread that handles smooth visual updates."""
        while self.running:
            current_time = time.time()
            
            with self.update_lock:
                should_update = (
                    self.pending_update and 
                    (current_time - self.last_update_time) >= self.min_update_interval
                )
                
                if should_update:
                    self.pending_update = False
                    self.last_update_time = current_time
            
            if should_update:
                self._update_visualization()
            
            # Sleep a small amount to prevent CPU hogging
            time.sleep(0.01)


class OfflinePlayer:
    """Player for offline playback of saved SfM reconstruction sessions."""
    
    def __init__(self, session_path: str, reconstruction_path: str):
        """Initialize the offline player.
        
        Args:
            session_path: Path to the session directory
            server: Optional viser server (if None, creates new one)
        """
        self.session_dir = Path(session_path)
        self.reconstruction_path = Path(reconstruction_path)
        self.server = viser.ViserServer()
        
        # Load all steps
        self.step_files = sorted(self.session_dir.glob("step_*.pkl"))
        self.steps_data = []
        self.current_step_index = 0
        
        if not self.step_files:
            raise ValueError(f"No step files found in {session_path}")
        
        # Load step data
        for step_file in tqdm.tqdm(self.step_files, desc="Loading"):
            with open(step_file, 'rb') as f:
                self.steps_data.append(pickle.load(f))
        
        print(f"Loaded {len(self.steps_data)} steps for playback")

        # Load final tracks for colors
        self.final_tracks = read_points3D_binary(os.path.join(self.reconstruction_path, 'points3D.bin'))
        self.final_track_colors = {}
        for track_id, track in self.final_tracks.items():
            self.final_track_colors[track_id] = track.rgb

        for step_data in tqdm.tqdm(self.steps_data, desc="Processing"):
            data = step_data['data']
            step_tracks = data.get('serialized_tracks', [])
            filtered_points = []
            filtered_colors = []
            for track in step_tracks:
                track_id = track['id']
                if track_id is not None and track_id in self.final_track_colors:
                    filtered_points.append(np.array(track['xyz'], dtype=np.float32))
                    filtered_colors.append(self.final_track_colors[track_id])
            if filtered_points and filtered_colors:
                points = np.stack(filtered_points, axis=0)
                colors = np.stack(filtered_colors, axis=0)
            else:
                points = np.zeros((0, 3), dtype=np.float32)
                colors = np.zeros((0, 3), dtype=np.uint8)

            step_data['filtered_points'] = points
            step_data['filtered_colors'] = colors
        
        # Visualization handles
        self.points_handle: Optional[viser.PointCloudHandle] = None
        self.camera_handles: Dict[int, viser.CameraFrustumHandle] = {}
        
        # Setup playback controls
        self._setup_playback_controls()
        
        # Display first step
        self._display_step(0)
    
    def _setup_playback_controls(self):
        """Setup GUI controls for playback."""
        with self.server.gui.add_folder("Playback Controls"):
            self.gui_play_button = self.server.gui.add_button("Play")
            self.gui_pause_button = self.server.gui.add_button("Pause")
            self.gui_prev_button = self.server.gui.add_button("Previous")
            self.gui_next_button = self.server.gui.add_button("Next")
            
            self.gui_speed_slider = self.server.gui.add_slider(
                "Playback Speed",
                min=0.1,
                max=5.0,
                step=0.1,
                initial_value=1.0,
            )

            self.gui_point_size = self.server.gui.add_slider(
                "Point Size",
                min=0.001,
                max=0.1,
                step=0.001,
                initial_value=0.05,
            )
            self.gui_frustum_scale = self.server.gui.add_slider(
                "Frustum Scale",
                min=0.01,
                max=0.5,
                step=0.01,
                initial_value=0.2,
            )

            self.gui_record_button = self.server.gui.add_button("Record Video")
        
        # Connect callbacks        
        @self.gui_play_button.on_click
        def _(_):
            self._start_playback()
        
        @self.gui_pause_button.on_click
        def _(_):
            self._pause_playback()
        
        @self.gui_prev_button.on_click
        def _(_):
            self._prev_step()
        
        @self.gui_next_button.on_click
        def _(_):
            self._next_step()

        @self.gui_point_size.on_update
        def _(_):
            if self.points_handle is not None:
                self.points_handle.point_size = self.gui_point_size.value

        @self.gui_frustum_scale.on_update
        def _(_):
            for handle in self.camera_handles.values():
                handle.scale = self.gui_frustum_scale.value

        @self.gui_record_button.on_click
        def _(event: viser.GuiEvent):
            self._record_video(event.client)
        
        # Playback state
        self.is_playing = False
        self.playback_thread = None
    
    def _display_step(self, step_index: int):
        """Display a specific step."""
        if step_index < 0 or step_index >= len(self.steps_data):
            return
        
        self.current_step_index = step_index
        step_data = self.steps_data[step_index]
        data = step_data['data']
        
        points = step_data['filtered_points']
        colors = step_data['filtered_colors']
            
        if self.points_handle is None:
            self.points_handle = self.server.scene.add_point_cloud(
                "pcd",
                points=points,
                colors=colors,
                point_size=0.05,
                point_shape="circle",
            )
        else:
            self.points_handle.points = points
            self.points_handle.colors = colors
            self.points_handle.point_size = self.gui_point_size.value
        
        # Update cameras
        current_camera_ids = set(data.get('cameras', {}).keys())
        existing_camera_ids = set(self.camera_handles.keys())
        
        # Remove old cameras
        for camera_id in existing_camera_ids - current_camera_ids:
            self.camera_handles[camera_id].remove()
            del self.camera_handles[camera_id]
        
        # Add/update cameras
        for camera_id, camera in data.get('cameras', {}).items():
            if camera_id in self.camera_handles:
                handle = self.camera_handles[camera_id]
                with self.server.atomic():
                    handle.wxyz = camera['wxyz']
                    handle.position = camera['position']
                    handle.scale = self.gui_frustum_scale.value
            else:
                frustum = self.server.scene.add_camera_frustum(
                    f"cam_{camera_id}",
                    fov=camera.get('fov', 0.8),
                    aspect=camera.get('aspect', 1.0),
                    scale=self.gui_frustum_scale.value,
                    wxyz=camera['wxyz'],
                    position=camera['position'],
                    image=camera.get('image', None),
                    color=np.array([255, 0, 0], dtype=np.uint8),
                )
                self.camera_handles[camera_id] = frustum
        
        self.server.flush()
    
    def _start_playback(self):
        """Start automatic playback."""
        if not self.is_playing:
            self.is_playing = True
            self.playback_thread = threading.Thread(target=self._playback_loop)
            self.playback_thread.daemon = True
            self.playback_thread.start()
    
    def _pause_playback(self):
        """Pause automatic playback."""
        self.is_playing = False
    
    def _prev_step(self):
        """Go to previous step."""
        if self.current_step_index > 0:
            self._display_step(self.current_step_index - 1)
    
    def _next_step(self):
        """Go to next step."""
        if self.current_step_index < len(self.steps_data) - 1:
            self._display_step(self.current_step_index + 1)
    
    def _playback_loop(self):
        """Playback loop for automatic progression."""
        while self.is_playing:
            if self.current_step_index < len(self.steps_data) - 1:
                self._display_step(self.current_step_index + 1)
                time.sleep(1.0 / self.gui_speed_slider.value)
            else:
                # Reached end, stop playback
                self.is_playing = False
                break

    def _record_video(self, client):
        images = []
        for idx in tqdm.tqdm(range(len(self.steps_data)), desc="Recording video"):
            self._display_step(idx)
            self.server.flush()
            time.sleep(0.1)
            
            img = client.camera.get_render(height=1080, width=1920)
            images.append(img)
        
        video_path = os.path.join(self.session_dir, "reconstruction_playback.mp4")
        with imageio.get_writer(video_path, fps=10, codec='libx264') as writer:
            for img in images:
                writer.append_data(img)
        print(f"Video saved to {video_path}")
