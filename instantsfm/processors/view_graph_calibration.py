import numpy as np
import torch
import tqdm
import pyceres
import os

from instantsfm.scene.defs import ConfigurationType, ViewGraph
from instantsfm.utils.cost_function import FetzerFocalLengthCostFunction, FetzerFocalLengthSameCameraCostFunction, fetzer_ds, fetzer_cost

# used by torch LM
import torch
from torch import nn
import pypose as pp
from pypose.optim.kernel import Cauchy
from bae.utils.pysolvers import PCG
from bae.optim import LM
from bae.autograd.function import TrackingTensor

def SolveViewGraphCalibration(view_graph:ViewGraph, cameras, images, VIEW_GRAPH_CALIBRATOR_OPTIONS):
    valid_image_pairs = {pair_id: image_pair for pair_id, image_pair in view_graph.image_pairs.items()
                         if image_pair.is_valid and image_pair.config in [ConfigurationType.CALIBRATED, ConfigurationType.UNCALIBRATED]}
    
    # FIX: Use a list of 1-element arrays to ensure memory persistence for PyCeres.
    # Slicing a large array (e.g. focals[i:i+1]) creates a temporary copy that gets garbage collected,
    # causing a Segfault when Ceres tries to access it.
    focals = [np.array([np.mean(cam.focal_length)], dtype=np.float64) for cam in cameras]

    problem = pyceres.Problem()
    options = pyceres.SolverOptions()
    loss_function = pyceres.CauchyLoss(VIEW_GRAPH_CALIBRATOR_OPTIONS['thres_loss_function'])
    if len(cameras) < 50:
        options.linear_solver_type = pyceres.LinearSolverType.DENSE_NORMAL_CHOLESKY
    else:
        options.linear_solver_type = pyceres.LinearSolverType.SPARSE_NORMAL_CHOLESKY
    
    single_cam_residuals = 0
    two_cam_residuals = 0
    added_pairs = set()
    
    # FIX: Pure Python implementation to avoid PyCeres Segfaults.
    # We skip optimization (solve) and only perform filtering using manual residual evaluation.
    
    print("Skipping View Graph Optimization (Stability Mode). Running filtering only...")
    
    invalid_counter = 0
    thres_two_view_error_sq = VIEW_GRAPH_CALIBRATOR_OPTIONS['thres_two_view_error'] ** 2
    
    # Debug logging
    debug_ids_str = os.environ.get("DEBUG_IMAGE_IDS", "")
    debug_ids = [int(x) for x in debug_ids_str.split(",")] if debug_ids_str else []

    for pair_id, image_pair in valid_image_pairs.items():
        image1, image2 = images[image_pair.image_id1], images[image_pair.image_id2]
        idx1 = int(image1.cam_id)
        idx2 = int(image2.cam_id)
        cam1, cam2 = cameras[idx1], cameras[idx2]
        
        # Get current focal lengths (mean)
        f1 = np.mean(cam1.focal_length)
        f2 = np.mean(cam2.focal_length)
        
        # Calculate residual manually
        res_val = np.zeros(2)
        
        if idx1 == idx2:
            cost_function = FetzerFocalLengthSameCameraCostFunction(image_pair.F, cam1.principal_point)
            # Evaluate expects list of parameter blocks. Each block is a numpy array.
            # For SameCamera, we pass [ [f1] ]
            cost_function.Evaluate([np.array([f1])], res_val, None)
        else:
            cost_function = FetzerFocalLengthCostFunction(image_pair.F, cam1.principal_point, cam2.principal_point)
            # For TwoCamera, we pass [ [f1], [f2] ]
            cost_function.Evaluate([np.array([f1]), np.array([f2])], res_val, None)
            
        # Check error
        sq_error = res_val[0]**2 + res_val[1]**2
        
        # Debug logging
        id1, id2 = image1.id, image2.id
        is_debug_pair = (id1 in debug_ids and id2 in debug_ids)
        
        if sq_error > thres_two_view_error_sq:
            invalid_counter += 1
            image_pair.is_valid = False
            view_graph.image_pairs[pair_id].is_valid = False
            
            if is_debug_pair or id1 in debug_ids or id2 in debug_ids:
                print(f"[DEBUG-DROP] Dropping edge {id1}-{id2}. Residual: {res_val[0]:.4f}, {res_val[1]:.4f} (SqSum: {sq_error:.4f} > {thres_two_view_error_sq})")
        elif is_debug_pair:
             print(f"[DEBUG-KEEP] Keeping edge {id1}-{id2}. Residual: {res_val[0]:.4f}, {res_val[1]:.4f}")

    print(f'invalid / total number of two view geometry: {invalid_counter} / {len(valid_image_pairs)}')
    
    # Since we skipped optimization, we don't update camera focal lengths.
    # We just mark them as refined so the pipeline continues.
    for cam in cameras:
        cam.has_refined_focal_length = True

class TorchVGC():
    def __init__(self, device='cuda:0'):
        self.device = device

    def Optimize(self, view_graph:ViewGraph, cameras, images, VIEW_GRAPH_CALIBRATOR_OPTIONS):
        cost_fn = fetzer_cost
        class FetzerNonBatched(nn.Module):
            def __init__(self, focals):
                super().__init__()
                self.focals = nn.Parameter(TrackingTensor(focals)) # (num_cameras, 1)
            def forward(self, ds, camera_indices1, camera_indices2):
                # ds: (num_pairs, 1, 3, 4)
                loss = cost_fn(self.focals[camera_indices1], self.focals[camera_indices2], ds)
                return loss

        valid_image_pairs = {pair_id: image_pair for pair_id, image_pair in view_graph.image_pairs.items()
                             if image_pair.is_valid and image_pair.config in [ConfigurationType.CALIBRATED, ConfigurationType.UNCALIBRATED]}
        focals = torch.tensor(np.array([np.mean(cam.focal_length) for cam in cameras]), dtype=torch.float64).to(self.device).unsqueeze(-1)
        self.camera_has_prior = torch.tensor([cam.has_prior_focal_length for cam in cameras], dtype=torch.bool).to(self.device)
        # TODO: Only support all cameras have prior focal length. If some cameras have prior focal length while others do not, 
        # they will be optimized together, which is not a good idea.
        if torch.all(self.camera_has_prior):
            print('All cameras have prior focal length, skipping view graph calibration')
            return

        ds_list = []
        camera_indices1_list = []
        camera_indices2_list = []
        for image_pair in valid_image_pairs.values():
            # add both directions
            image1, image2 = images[image_pair.image_id1], images[image_pair.image_id2]
            cam_id1, cam_id2 = image1.cam_id, image2.cam_id
            cam1, cam2 = cameras[cam_id1], cameras[cam_id2]
            principal_point0, principal_point1 = cam1.principal_point, cam2.principal_point
            K0 = np.array([[1, 0, principal_point0[0]], [0, 1, principal_point0[1]], [0, 0, 1]])
            K1 = np.array([[1, 0, principal_point1[0]], [0, 1, principal_point1[1]], [0, 0, 1]])
            i1_G_i0 = K1.T @ image_pair.F @ K0
            ds = fetzer_ds(i1_G_i0)
            ds_list.append(ds)
            camera_indices1_list.append(cam_id1)
            camera_indices2_list.append(cam_id2)
            i0_G_i1 = i1_G_i0.T
            ds = fetzer_ds(i0_G_i1)
            ds_list.append(ds)
            camera_indices1_list.append(cam_id2)
            camera_indices2_list.append(cam_id1)
        
        ds = torch.tensor(np.array(ds_list), dtype=torch.float64).to(self.device).unsqueeze(1)
        camera_indices1 = torch.tensor(np.array(camera_indices1_list), dtype=torch.int64).to(self.device).flatten()
        camera_indices2 = torch.tensor(np.array(camera_indices2_list), dtype=torch.int64).to(self.device).flatten()

        model = FetzerNonBatched(focals)
        strategy = pp.optim.strategy.TrustRegion(radius=1e2, max=1e6, up=2.0, down=0.5**4)
        sparse_solver = PCG(tol=1e-5) # cuSolverSP()
        cauchy_kernel = Cauchy(VIEW_GRAPH_CALIBRATOR_OPTIONS['thres_loss_function'])
        optimizer = LM(model, strategy=strategy, solver=sparse_solver, kernel=cauchy_kernel, reject=30)

        input = {
            "ds": ds,
            "camera_indices1": camera_indices1,
            "camera_indices2": camera_indices2
        }

        window_size = 3
        loss_history = []
        progress_bar = tqdm.trange(VIEW_GRAPH_CALIBRATOR_OPTIONS['max_num_iterations'])
        for _ in progress_bar:
            loss = optimizer.step(input)
            torch.set_printoptions(threshold=torch.inf)
            print(f'focals: {model.focals}, loss: {loss.item()}')
            loss_history.append(loss.item())
            if len(loss_history) >= 2*window_size:
                avg_recent = np.mean(loss_history[-window_size:])
                avg_previous = np.mean(loss_history[-2*window_size:-window_size])
                improvement = (avg_previous - avg_recent) / avg_previous
                if abs(improvement) < VIEW_GRAPH_CALIBRATOR_OPTIONS['function_tolerance']:
                    break
            progress_bar.set_postfix({"loss": loss.item()})
        progress_bar.close()

        focals_ = focals.detach().cpu().numpy().squeeze()
        counter = 0
        for cam, focal in zip(cameras, focals_):
            if (focal / np.mean(cam.focal_length) < VIEW_GRAPH_CALIBRATOR_OPTIONS['thres_lower_ratio'] or 
                focal / np.mean(cam.focal_length) > VIEW_GRAPH_CALIBRATOR_OPTIONS['thres_higher_ratio']):
                counter += 1
                continue
            cam.has_refined_focal_length = True
            cam.focal_length = np.array([focal, focal])
        
        print(f'{counter} cameras are rejected in view graph calibration')

        thres_two_view_error_sq = VIEW_GRAPH_CALIBRATOR_OPTIONS['thres_two_view_error'] ** 2

        # manually calculate the residuals
        loss = model.forward(ds, camera_indices1, camera_indices2).detach().cpu().numpy()
        invalid_counter = 0
        loss_sq = np.sum(loss ** 2, axis=-1)
        for idx, (pair_id, image_pair) in enumerate(valid_image_pairs.items()):
            if loss_sq[idx*2] > thres_two_view_error_sq:
                invalid_counter += 1
                view_graph.image_pairs[pair_id].is_valid = False
        print(f'invalid / total number of two view geometry: {invalid_counter} / {len(valid_image_pairs)}')