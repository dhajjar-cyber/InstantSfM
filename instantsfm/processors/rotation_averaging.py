import numpy as np
import sys
import os
import time
from scipy.spatial.transform import Rotation as R
from scipy.sparse import lil_matrix, csc_matrix, diags
from sksparse.cholmod import cholesky
from queue import Queue
import tqdm

from instantsfm.scene.defs import ViewGraph
from instantsfm.utils.tree import MaximumSpanningTree
from instantsfm.utils.l1_solver import L1Solver
    
class RotationEstimator:
    def __init__(self):
        self.fixed_camera_id = -1

    def InitializeFromMaximumSpanningTree(self, view_graph:ViewGraph, images):
        print("Initializing from Maximum Spanning Tree...")
        sys.stdout.flush()
        parents, root = MaximumSpanningTree(view_graph, images)
        children = [[] for _ in range(len(images))]
        for idx, parent in enumerate(parents):
            if idx != root:
                children[parent].append(idx)
        q = Queue()
        q.put(root)

        while not q.empty():
            current = q.get()
            for child in children[current]:
                q.put(child)
            if current == root or parents[current] == -1:
                continue
            # Try both key orders since we don't know which was stored
            parent_id = parents[current]
            pair_key = (current, parent_id) if (current, parent_id) in view_graph.image_pairs else (parent_id, current)
            image_pair = view_graph.image_pairs[pair_key]
            if image_pair.image_id1 == current:
                pair_rotation = R.from_quat(image_pair.rotation).as_matrix()
                rotation = np.linalg.inv(pair_rotation) @ images.world2cams[parents[current], :3, :3]
                images.world2cams[current, :3, :3] = rotation
            else:
                pair_rotation = R.from_quat(image_pair.rotation).as_matrix()
                rotation = pair_rotation @ images.world2cams[parents[current], :3, :3]
                images.world2cams[current, :3, :3] = rotation
        print("Initialization done.")
        sys.stdout.flush()

    def SetupLinearSystem(self, view_graph:ViewGraph, images):
        print("Setting up linear system...")
        sys.stdout.flush()
        registered_mask = images.is_registered
        self.images = {image_id: image for image_id, image in enumerate(images) if registered_mask[image_id]}
        
        # DEBUG: Check if Debug Images survived to Phase 4
        debug_ids_str = os.environ.get("DEBUG_IMAGE_IDS", "")
        debug_ids = [int(x) for x in debug_ids_str.split(",")] if debug_ids_str else []
        
        for dbg_id in debug_ids:
            if dbg_id in self.images:
                print(f"[DEBUG] Phase 4: Image {dbg_id} is REGISTERED and included in Rotation Averaging.")
            else:
                print(f"[DEBUG] Phase 4: Image {dbg_id} is NOT REGISTERED. Dropped before Phase 4.")

        # Filter valid pairs
        valid_pairs = [p for p in view_graph.image_pairs.values() if p.is_valid]
        
        self.image_id2idx = {}
        self.rotation_estimated = np.zeros(len(images) * 3)
        num_dof = 0
        
        # Batch compute axis angles for registered images
        Rs = images.world2cams[:, :3, :3]  # (N, 3, 3)
        axis_angles = R.from_matrix(Rs.reshape(-1, 3, 3)).as_rotvec().reshape(len(images), 3)  # (N, 3)
        
        for image_id in range(len(images)):
            if not registered_mask[image_id]:
                continue
            self.image_id2idx[image_id] = num_dof
            self.rotation_estimated[num_dof:num_dof+3] = axis_angles[image_id]
            num_dof += 3
            
        if self.fixed_camera_id == -1:
            self.fixed_camera_id = list(self.images.keys())[0]
            self.fixed_camera_rotation = self.images[self.fixed_camera_id].axis_angle()
        
        # Vectorized Setup
        num_pairs = len(valid_pairs)
        print(f"Processing {num_pairs} pairs...")
        sys.stdout.flush()
        
        # Extract IDs
        img1s = np.array([p.image_id1 for p in valid_pairs])
        img2s = np.array([p.image_id2 for p in valid_pairs])
        
        # Map to vector indices
        max_id = max(images.ids) if len(images.ids) > 0 else 0
        id_map = np.full(max_id + 1, -1, dtype=int)
        for img_id, idx in self.image_id2idx.items():
            id_map[img_id] = idx
            
        vec_idx1 = id_map[img1s]
        vec_idx2 = id_map[img2s]
        
        # Store for ComputeResiduals
        self.pair_vec_idx1 = vec_idx1
        self.pair_vec_idx2 = vec_idx2
        
        # Precompute R_rels
        quats = np.array([p.rotation for p in valid_pairs])
        self.pair_R_rels = R.from_quat(quats).as_matrix()
        
        # Build Sparse Matrix
        row_base = np.arange(num_pairs) * 3
        rows_block = np.repeat(row_base, 3).reshape(-1, 3) + np.arange(3)
        rows_flat = rows_block.flatten()
        all_rows = np.concatenate([rows_flat, rows_flat])
        
        cols1_block = np.repeat(vec_idx1, 3).reshape(-1, 3) + np.arange(3)
        cols1_flat = cols1_block.flatten()
        cols2_block = np.repeat(vec_idx2, 3).reshape(-1, 3) + np.arange(3)
        cols2_flat = cols2_block.flatten()
        all_cols = np.concatenate([cols1_flat, cols2_flat])
        
        all_data = np.concatenate([np.full(num_pairs * 3, -1), np.full(num_pairs * 3, 1)])
        
        # Fixed camera
        fixed_idx = self.image_id2idx[self.fixed_camera_id]
        fixed_rows = np.arange(num_pairs * 3, num_pairs * 3 + 3)
        fixed_cols = np.arange(fixed_idx, fixed_idx + 3)
        fixed_data = np.ones(3)
        
        final_rows = np.concatenate([all_rows, fixed_rows])
        final_cols = np.concatenate([all_cols, fixed_cols])
        final_data = np.concatenate([all_data, fixed_data])
        
        self.sparse_matrix = csc_matrix((final_data, (final_rows, final_cols)), shape=(num_pairs * 3 + 3, num_dof))
        
        self.tangent_space_step = np.zeros(num_dof)
        self.tangent_space_residual = np.zeros(num_pairs * 3 + 3)
        print("Linear system setup done.")
        sys.stdout.flush()

    def ComputeResiduals(self):
        # Vectorized residual computation
        idx1_all = np.repeat(self.pair_vec_idx1, 3).reshape(-1, 3) + np.arange(3)
        idx2_all = np.repeat(self.pair_vec_idx2, 3).reshape(-1, 3) + np.arange(3)
        
        rot_vecs1 = self.rotation_estimated[idx1_all] # (num_pairs, 3)
        rot_vecs2 = self.rotation_estimated[idx2_all] # (num_pairs, 3)
        
        Rs1 = R.from_rotvec(rot_vecs1).as_matrix()
        Rs2 = R.from_rotvec(rot_vecs2).as_matrix()
        
        # R_diff = R2.T @ R_rel @ R1
        Rs2_T = Rs2.transpose(0, 2, 1)
        term = Rs2_T @ self.pair_R_rels @ Rs1
        
        residuals = -R.from_matrix(term).as_rotvec() # (N, 3)
        
        self.tangent_space_residual[:-3] = residuals.flatten()
        
        # Fixed camera residual
        fixed_idx = self.image_id2idx[self.fixed_camera_id]
        fixed_rot_vec = self.rotation_estimated[fixed_idx:fixed_idx+3]
        fixed_R = R.from_rotvec(fixed_rot_vec).as_matrix()
        fixed_target_R = R.from_rotvec(self.fixed_camera_rotation).as_matrix()
        
        self.tangent_space_residual[-3:] = R.from_matrix(fixed_target_R.T @ fixed_R).as_rotvec()

    def SolveL1Regression(self, ROTATION_ESTIMATOR_OPTIONS, L1_SOLVER_OPTIONS):
        self.ComputeResiduals()
        iteration = 0
        curr_norm = 0
        l1_solver = L1Solver(self.sparse_matrix)
        LOCAL_L1_SOLVER_OPTIONS = L1_SOLVER_OPTIONS.copy()
        LOCAL_L1_SOLVER_OPTIONS['max_num_iterations'] = 10

        print(f"Starting L1 Regression (max {ROTATION_ESTIMATOR_OPTIONS['max_num_l1_iterations']} iterations)...")
        sys.stdout.flush()

        while iteration < ROTATION_ESTIMATOR_OPTIONS['max_num_l1_iterations']:
            last_norm = curr_norm
            self.tangent_space_step = l1_solver.solve(self.tangent_space_residual, LOCAL_L1_SOLVER_OPTIONS)
            if np.any(np.isnan(self.tangent_space_step)):
                print('nan error')
                sys.stdout.flush()
                return False

            curr_norm = np.linalg.norm(self.tangent_space_step)
            self.UpdateGlobalRotations()
            self.ComputeResiduals()

            iteration += 1
            avg_step = self.ComputeAverageStepSize()
            print(f"L1 Iteration {iteration}: norm={curr_norm:.6f}, avg_step={avg_step:.6f}")
            sys.stdout.flush()

            EPS = 1e-6
            if avg_step < ROTATION_ESTIMATOR_OPTIONS['l1_step_convergence_threshold'] or np.abs(last_norm - curr_norm) < EPS:
                break
            LOCAL_L1_SOLVER_OPTIONS['max_num_iterations'] = min(LOCAL_L1_SOLVER_OPTIONS['max_num_iterations'] * 2, 100)
        return True

    def SolveIRLS(self, ROTATION_ESTIMATOR_OPTIONS):
        llt = cholesky(csc_matrix(self.sparse_matrix.T @ self.sparse_matrix))

        sigma = np.deg2rad(ROTATION_ESTIMATOR_OPTIONS['irls_loss_parameter_sigma'])

        self.ComputeResiduals()
        print(f"Starting IRLS (max {ROTATION_ESTIMATOR_OPTIONS['max_num_irls_iterations']} iterations)...")
        sys.stdout.flush()
        
        num_pairs = len(self.pair_vec_idx1)
        weights_irls = np.ones(num_pairs * 3 + 3)
        
        for iter_idx in range(ROTATION_ESTIMATOR_OPTIONS['max_num_irls_iterations']):
            # Vectorized weight update
            residuals = self.tangent_space_residual[:-3].reshape(-1, 3)
            err_squared = np.sum(residuals**2, axis=1) # (N,)
            
            tmp = err_squared + sigma**2
            w = sigma**2 / (tmp**2)
            
            if np.any(np.isnan(w)):
                print("nan weight!")
                sys.stdout.flush()
                return False
            
            # Expand weights to (N*3,)
            w_expanded = np.repeat(w, 3)
            weights_irls[:-3] = w_expanded
            weights_irls[-3:] = 1 # Fixed camera constraint weight

            at_weight = self.sparse_matrix.T @ diags(weights_irls)

            llt = cholesky(csc_matrix(at_weight @ self.sparse_matrix))

            self.tangent_space_step.fill(0)
            self.tangent_space_step = llt(at_weight @ self.tangent_space_residual)
            self.UpdateGlobalRotations()
            self.ComputeResiduals()

            avg_step = self.ComputeAverageStepSize()
            print(f"IRLS Iteration {iter_idx + 1}: avg_step={avg_step:.6f}")
            sys.stdout.flush()

            if avg_step < ROTATION_ESTIMATOR_OPTIONS['irls_step_convergence_threshold']:
                break
        return True

    def UpdateGlobalRotations(self):
        for image_id in self.images.keys():
            vector_idx = self.image_id2idx[image_id]
            R_ori = R.from_rotvec(self.rotation_estimated[vector_idx:vector_idx+3]).as_matrix()
            self.rotation_estimated[vector_idx:vector_idx+3] = R.from_matrix(
                R_ori @ R.from_rotvec(-self.tangent_space_step[vector_idx:vector_idx+3]).as_matrix()).as_rotvec()

    def ComputeAverageStepSize(self):
        total_update = 0.
        for image_id in self.images.keys():
            vector_idx = self.image_id2idx[image_id]
            total_update += np.linalg.norm(self.tangent_space_step[vector_idx:vector_idx+3])
        return total_update / len(self.image_id2idx)

    def EstimateRotations(self, view_graph:ViewGraph, images, ROTATION_ESTIMATOR_OPTIONS, L1_SOLVER_OPTIONS):
        start_time_total = time.time()

        self.InitializeFromMaximumSpanningTree(view_graph, images)
        self.SetupLinearSystem(view_graph, images)

        if ROTATION_ESTIMATOR_OPTIONS['max_num_l1_iterations'] > 0:
            start_time_l1 = time.time()
            if not self.SolveL1Regression(ROTATION_ESTIMATOR_OPTIONS, L1_SOLVER_OPTIONS):
                return False
            print(f"RotationEstimator: L1 regression took {time.time() - start_time_l1:.2f} seconds")
            sys.stdout.flush()
            
        if ROTATION_ESTIMATOR_OPTIONS['max_num_irls_iterations'] > 0:
            start_time_irls = time.time()
            if not self.SolveIRLS(ROTATION_ESTIMATOR_OPTIONS):
                return False
            print(f"RotationEstimator: IRLS took {time.time() - start_time_irls:.2f} seconds")
            sys.stdout.flush()
            
        # Batch write rotation results
        for image_id in range(len(images)):
            if not images.is_registered[image_id]:
                continue
            idx = self.image_id2idx[image_id]
            images.world2cams[image_id, :3, :3] = R.from_rotvec(self.rotation_estimated[idx:idx+3]).as_matrix()
        
        print(f"RotationEstimator: total rotation averaging took {time.time() - start_time_total:.2f} seconds")
        sys.stdout.flush()
        return True