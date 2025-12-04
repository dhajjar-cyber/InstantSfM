# InstantSFM Workflow

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
    *   **Relative Pose Estimation:** 
        *   Converts raw feature matches into **Relative Poses** (Rotation/Translation) using the Epipolar Constraint (Essential Matrix).
        *   *Distinction:* The ViewGraph stores **Verified Inliers** (geometrically consistent matches), which is a subset of the "Raw Matches" found in the database.
*   **Output:** A `ViewGraph` containing thousands of independent, verified pairwise relationships.
*   **Code Reference:**
    *   **Orchestrator:** `instantsfm/controllers/global_mapper.py` -> `SolveGlobalMapper`
    *   **Data Loading:** `instantsfm/controllers/data_reader.py` -> `ReadColmapDatabase` (Handles DB read & Rig Grouping)
    *   **Pose Estimation:** `instantsfm/processors/relpose_estimation.py` -> `EstimateRelativePose` (Computes E/F/H matrices)
    *   **Filtering:** `instantsfm/processors/relpose_filter.py` -> `FilterInlierNum`, `FilterInlierRatio`

### Phase 1 Diagnostics & Quality Assurance
Before proceeding to Phase 2, it is critical to validate the ViewGraph quality using `modules/phase_1/scripts/diagnostics/diagnose_viewgraph_phase1.py`.

**Key Metrics to Watch:**
1.  **Rig Connectivity:** Should be >80% for expected overlapping pairs.
2.  **Valid Pair Retention:**
    *   **Raw Drop Rate:** It is normal for ~60-70% of *raw* database pairs to be dropped.
    *   **Why?** Most raw pairs are `UNDEFINED` (failed geometric verification due to lack of overlap) or `WATERMARK` (zero parallax/static overlay).
    *   **The Metric that Matters:** Focus on the **Retention Rate of Valid Geometric Pairs** (Config 2-6, 8). This should be high (>75%).
    *   *Insight:* If `WATERMARK` drops are high, it usually indicates the camera was stationary (zero baseline), which is correctly filtered out to prevent triangulation errors.

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
    *   **Filtering:** 
        *   **Inlier Count:** Discards pairs with too few geometrically consistent matches (`FilterInlierNum`).
        *   **Inlier Ratio:** Discards pairs where the ratio of inliers to raw matches is too low (`FilterInlierRatio`).
    *   **Connectivity Check:** Ensures the graph remains a single connected component (`keep_largest_connected_component`). If the graph splits, only the largest group is kept.
*   **Output:** A robust `ViewGraph` where every edge has a trusted relative pose ($R_{ij}, t_{ij}$).
*   **Code Reference:**
    *   **Orchestrator:** `instantsfm/processors/relpose_estimation.py` -> `EstimateRelativePose`
    *   **Undistortion:** `instantsfm/processors/image_undistortion.py` -> `UndistortImages`
    *   **Filtering:** `instantsfm/processors/relpose_filter.py` -> `FilterInlierNum`, `FilterInlierRatio`
    *   **Graph Logic:** `instantsfm/scene/defs.py` -> `ViewGraph.keep_largest_connected_component`
    *   **Checkpoint:** `checkpoint_relpose.pkl` (Saved here if `checkpoint_path` is provided)

## Phase 4: Rotation Averaging
*   **Goal:** Solve for the global orientation of every camera.
*   **Action:** Takes all the pairwise rotations (A relative to B, B relative to C) and solves a global optimization problem to find the absolute rotation of A, B, and C in the world.
*   **Output:** **Global Rotations** for every camera (but no positions yet).
    *   **Checkpoint:** `checkpoint_rotation.pkl` (Saved here if `save_rotation_checkpoint_path` is provided)

## Phase 5: Track Establishment
*   **Goal:** Build the 3D points.
*   **Action:** Chains pairwise matches into **Tracks**.
    *   *Definition:* A Track represents a single physical point seen in multiple images.
    *   *Example:* If A matches B, and B matches C, create a single Track seen by {A, B, C}.
*   **Output:** A list of `Tracks` (raw 2D observation chains), ready for triangulation.
    *   **Checkpoint:** `checkpoint_tracks.pkl` (Saved here if `save_tracks_checkpoint_path` is provided)

## Phase 6: Global Positioning
*   **Goal:** Solve for the global position (X, Y, Z) of every camera.
*   **Action:**
    *   **Initialization:** Uses a Spanning Tree (BFS) to place cameras roughly in space.
    *   **Optimization:** Moves cameras to minimize the error between where they *should* be (based on tracks) and where they *are*.
    *   **Rig Mode:** If multiple cameras per timestamp are detected, `OptimizeMulti` is used to enforce rigid constraints between them, preventing drift in visually disconnected rigs.
*   **Output:** **Global Positions** for every camera and rough 3D coordinates for the Tracks.
*   **Code Reference:**
    *   **Orchestrator:** `instantsfm/processors/global_positioning.py` -> `TorchGP`
    *   **Optimization:** `instantsfm/processors/global_positioning.py` -> `OptimizeMulti` (Rig Mode) / `OptimizeSingle` (Standard)
    *   **Solver:** `instantsfm/processors/global_positioning.py` -> `SolveMulti` (Linear Solver for Rig Mode)

## Phase 7: Bundle Adjustment (BA)
*   **Goal:** The final polish.
*   **Action:** Optimizes **everything** simultaneously (Camera Rotations, Camera Positions, and 3D Track Points) to minimize "Reprojection Error" (the difference between where a point appears in the photo and where the 3D model says it should be).
    *   **Rig Mode:** Uses `OptimizeMulti` to maintain rigid constraints during the final polish.
*   **Output:** A highly accurate, refined sparse 3D reconstruction.
*   **Code Reference:**
    *   **Orchestrator:** `instantsfm/processors/bundle_adjustment.py` -> `TorchBA`
    *   **Optimization:** `instantsfm/processors/bundle_adjustment.py` -> `OptimizeMulti` (Rig Mode)

## Phase 8: Retriangulation (Optional)
*   **Goal:** Fill in missing details.
*   **Action:** Uses the now-perfect camera poses to look for tracks that were missed or discarded earlier and adds them back in.
*   **Output:** A denser point cloud.
