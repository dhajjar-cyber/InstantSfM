# InstantSFM Workflow
* Main caller : /workspace/3DCD/modules/phase_1/scripts/SFM/run_instantsfm_reconstruction.sh

This document outlines the high-level pipeline of InstantSFM. It follows a "Global SfM" approach, solving for all camera poses simultaneously rather than incrementally.

## Phase 1: Preprocessing & View Graph Construction
*   **Goal:** Build the "skeleton" of the reconstruction by establishing local relationships and identifying rig structures.
*   **Input:** 
    *   **Feature Matches:** 2D keypoint correspondences extracted from the database (e.g., COLMAP `.db`).
    *   **Image Metadata:** Camera IDs, filenames, and initial parameters.
    *   *Note:* Actual pixel data is **not** used in this phase; only the feature matches are required.
*   **Action:**
    *   **Data Loading & Rig Grouping:** 
        *   Reads `images`, `cameras`, and `matches` from the database.
        *   **Rig Identification:** Parses filenames (e.g., regex `f(\d+)`) to group images by timestamp. Assigns "partner" IDs to images in the same rig frame.
    *   **View Graph Construction:** 
        *   Creates a graph where nodes are images and edges are matches.
        *   **Filtering:** Applies strict thresholds (min inliers, inlier ratio) to discard weak connections.
        *   **Connectivity Check:** Keeps only the "Largest Connected Component" to ensure the graph is not fragmented.
*   **Output:** A `ViewGraph` containing thousands of potential pairwise relationships.
*   **Code Reference:**
    *   **Orchestrator:** `instantsfm/controllers/global_mapper.py` -> `SolveGlobalMapper`
    *   **Data Loading:** `instantsfm/controllers/data_reader.py` -> `ReadColmapDatabase` (Handles DB read & Rig Grouping)

## Phase 2: View Graph Calibration
*   **Goal:** Refine the camera intrinsics (specifically **Focal Lengths**) to ensure they are consistent with the observed geometry.
*   **Action:** 
    *   Formulates a global optimization problem using **Ceres Solver**.
    *   Minimizes the error between the focal lengths and the constraints imposed by the Fundamental Matrices ($F$) of the image pairs.
    *   **Self-Calibration:** Can adjust focal lengths even if the initial values were approximate, leveraging the redundancy in the ViewGraph.
*   **Output:** Refined `Camera` objects with more accurate focal lengths.
*   **Code Reference:**
    *   **Solver:** `instantsfm/processors/view_graph_calibration.py` -> `SolveViewGraphCalibration`
    *   **Cost Function:** `instantsfm/utils/cost_function.py` -> `FetzerFocalLengthCostFunction`

## Phase 3: Relative Pose Estimation & Filtering
*   **Goal:** Compute the precise relative 3D transformation (Rotation $R$, Translation $t$) for every connected pair and remove unreliable connections.
*   **Action:**
    *   **Undistortion:** First, undistorts the 2D keypoints using the refined camera parameters from Phase 2.
    *   **Pose Estimation:** 
        *   Computes the **Essential Matrix ($E$)** (or Homography $H$ for planar scenes) using RANSAC.
        *   Decomposes $E$ into relative rotation and translation ($R, t$) using `cv2.recoverPose`.
        *   **Important:** This step treats every pair **independently**. It does **not** yet enforce rig constraints (fixed relative pose between cameras).
    *   **Filtering:** 
        *   **Inlier Count:** Discards pairs with too few geometrically consistent matches (`FilterInlierNum`).
        *   **Inlier Ratio:** Discards pairs where the ratio of inliers to raw matches is too low (`FilterInlierRatio`).
    *   **Connectivity Check:** Ensures the graph remains a single connected component (`keep_largest_connected_component`). If the graph splits, only the largest group is kept.
*   **Output:** A robust `ViewGraph` where every edge has a trusted relative pose ($R_{ij}, t_{ij}$).
    *   **Checkpoint:** `checkpoint_relpose.pkl`
*   **Code Reference:**
    *   **Orchestrator:** `instantsfm/processors/relpose_estimation.py` -> `EstimateRelativePose`
    *   **Undistortion:** `instantsfm/processors/image_undistortion.py` -> `UndistortImages`
    *   **Filtering:** `instantsfm/processors/relpose_filter.py` -> `FilterInlierNum`, `FilterInlierRatio`

### Phase 3 Diagnostics
Before proceeding, validate the ViewGraph using `modules/phase_1/scripts/diagnostics/diagnose_viewgraph.py`.
*   **Check:** Ensure the graph is fully connected (1 component).
*   **Check:** Verify focal lengths are stable (no drift).
*   **Note:** "Unstable" rig pairs (high deviation) are expected here because rig constraints are not yet enforced.

## Phase 4: Rotation Averaging
*   **Goal:** Solve for the global orientation of every camera.
*   **Action:** 
    *   Takes all the pairwise rotations (A relative to B, B relative to C) and solves a global optimization problem (L1/L2 averaging) to find the absolute rotation of every camera in the world frame.
    *   **Rig Handling:** This phase solves for rotations based on visual evidence. It does **not** strictly enforce the rigid relative poses defined in `rig_config.json`.
*   **Output:** **Global Rotations** for every camera (but no positions yet).
    *   **Checkpoint:** `checkpoint_rotation.pkl`

### Phase 4 Diagnostics
Validate the rotation quality using `modules/phase_1/scripts/diagnostics/diagnose_rotation.py`.
*   **Internal Consistency:** The most important metric. Check if the *angles between cameras* match the rig configuration (e.g., `analyze_rig_internal_angles`).
*   **Global Offset:** A systematic deviation (e.g., 90° or 120° error on all pairs) is **normal** and indicates a coordinate system convention difference (World vs Rig frame), not a failure.
*   **Smoothness:** Check for low angular velocity (~0.02°/frame) to ensure stable tracking.

## Phase 5: Track Establishment
*   **Goal:** Build the 3D points structure by chaining pairwise matches into global tracks.
*   **Action:**
    *   **Feature Graph Construction:** Aggregates all verified pairwise matches from the ViewGraph into a massive graph where nodes are individual 2D feature observations and edges are matches.
    *   **Connected Components (Transitive Closure):** Identifies "Tracks" by finding connected components in this feature graph.
        *   *Logic:* If Feature A (Image 1) matches Feature B (Image 2), and Feature B matches Feature C (Image 3), they are merged into a single Track {A, B, C}.
    *   **Deduplication:** Cleans up tracks to ensure a single track does not contain multiple conflicting features from the same image.
    *   **Length Filtering:** Discards "weak" tracks that appear in too few images (e.g., < 3 observations). This step typically reduces the raw track count by ~50%, keeping only the most reliable points for triangulation.
*   **Output:** A `Tracks` object containing millions of 2D observation chains, ready for triangulation.
    *   **Checkpoint:** `checkpoint_tracks.pkl`

## Phase 6: Global Positioning
*   **Goal:** Solve for the global position (X, Y, Z) of every camera.
*   **Action:**
    *   **1. Initialization (Spanning Tree):**
        *   Constructs a graph from pairwise connections.
        *   Selects a root camera (highest connectivity).
        *   Propagates positions via Breadth-First Search (BFS) using the relative translations ($t_{ij}$) computed in Phase 3. This provides a rough initial "skeleton" for the scene.
    *   **2. Data Preparation:**
        *   Filters tracks to ensure they have enough observations (min_num_view_per_track).
        *   Subsamples tracks to fit within GPU memory limits during optimization. The limit is defined in `instantsfm/config/colmap.py` under `GLOBAL_POSITIONER_OPTIONS['max_tracks_for_gp']` (default: 500,000).
    *   **3. Optimization (Levenberg-Marquardt):**
        *   Formulates a non-linear least squares problem to minimize the difference between observed 2D features and projected 3D points.
        *   **Rig Mode (`OptimizeMulti`):** Decomposes camera position into `Group_Ref_Pos + Relative_Pos`.
            *   **Constraint Enforcement:** If `enforce_zero_baseline` is enabled, the `Relative_Pos` component is **frozen**.
                *   *Technical Detail:* This is achieved by registering the relative translation tensor as a **buffer** (`register_buffer`) instead of a `nn.Parameter`. This excludes it from the optimizer's parameter list entirely, preventing the Levenberg-Marquardt solver from attempting to calculate gradients or updates for it. This ensures the relative translations remain exactly at their initialization (zero) and prevents optimizer crashes due to size mismatches.
                *   *Result:* The optimizer moves the entire rig as a single rigid unit (`Group_Ref_Pos`), preventing individual cameras from drifting apart or violating the rig geometry.
        *   **Solver:** Uses a sparse solver (PCG) with a Huber loss function to be robust against outliers.
    *   **4. Post-Processing:**
        *   Updates global coordinates in the `images` and `tracks` objects.
        *   Normalizes the reconstruction (centers and scales the scene).
        *   Filters tracks based on triangulation angles to remove unstable points.
*   **Output:** **Global Positions** for every camera and rough 3D coordinates for the Tracks.
    *   **Checkpoint:** `checkpoint_gp.pkl`
*   **Code Reference:**
    *   **Orchestrator:** `instantsfm/processors/global_positioning.py` -> `TorchGP`
    *   **Optimization:** `instantsfm/processors/global_positioning.py` -> `OptimizeMulti` (Rig Mode) / `OptimizeSingle` (Standard)
    *   **Solver:** `instantsfm/processors/global_positioning.py` -> `SolveMulti` (Linear Solver for Rig Mode)

### Phase 6 Diagnostics
Validate the global positioning using `modules/phase_1/scripts/diagnostics/diagnose_gp.py`.
*   **Rig Consistency:** Checks the "spread" of relative translations within a rig. High spread indicates drift.
*   **Trajectory Smoothness:** Analyzes velocity and acceleration to detect jumps.
*   **Visualization:** Generates an interactive 3D plot (`gp_visualization.html`) to inspect camera placement and rig structure visually.

## Phase 7: Bundle Adjustment (BA)
*   **Goal:** The final polish.
*   **Action:** Optimizes **everything** simultaneously (Camera Rotations, Camera Positions, and 3D Track Points) to minimize "Reprojection Error" (the difference between where a point appears in the photo and where the 3D model says it should be).
    *   **Rig Mode:** Uses `OptimizeMulti` to maintain rigid constraints during the final polish.
        *   **Constraint Enforcement:** Similar to Phase 6, if `enforce_zero_baseline` is enabled, the relative translations are frozen using `register_buffer`. This ensures the rig remains perfectly rigid while the global pose and structure are refined.
*   **Output:** A highly accurate, refined sparse 3D reconstruction.
    *   **Checkpoint:** `checkpoint_ba.pkl`
*   **Code Reference:**
    *   **Orchestrator:** `instantsfm/processors/bundle_adjustment.py` -> `TorchBA`
    *   **Optimization:** `instantsfm/processors/bundle_adjustment.py` -> `OptimizeMulti` (Rig Mode)

## Phase 8: Retriangulation (Optional)
*   **Goal:** Fill in missing details.
*   **Action:** Uses the now-perfect camera poses to look for tracks that were missed or discarded earlier and adds them back in.
*   **Output:** A denser point cloud.
