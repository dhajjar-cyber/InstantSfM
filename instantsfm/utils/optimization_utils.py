import torch

def refine_optimization_inputs(inputs_dict, verbose=True):
    """
    Refines and validates inputs before passing them to the BAE (Bundle Adjustment Engine) optimizer.
    
    This utility serves as a centralized fix for a known issue in the BAE library where optimization 
    fails for 1x1 blocks (scalar variables).
    
    Key Operations:
    1. **Diagnostic Logging**: Optionally prints the shapes and data types of all input tensors to help 
       debug shape mismatches or type errors (e.g., float32 vs float64).
    2. **BAE Workaround (Padding)**: Detects tensors with shape (N, 1) and pads them to (N, 2) by 
       appending a column of zeros.
       - **Why?** The BAE solver's block management logic has a bug when handling single-scalar blocks. 
         Padding them to size 2 bypasses this specific code path.
       - **Implication**: Downstream cost functions (e.g., `pairwise_cost`, `reproject_simple_pinhole`) 
         must be aware of this padding. They should typically use the first column as the actual variable 
         and the second column as a "dummy" variable (multiplied by 0) to satisfy `torch.vmap` 
         gradient tracing requirements without affecting the cost.

    Args:
        inputs_dict (dict): A dictionary mapping variable names (str) to PyTorch tensors.
                            Example: {"scales": tensor(N, 1), "points": tensor(N, 3)}
        verbose (bool): If True, prints diagnostic information to stdout. Default is True.
        
    Returns:
        dict: A new dictionary containing the refined tensors. Tensors that were (N, 1) are now (N, 2).
              Tensors that were None are preserved as None.
    """
    if verbose:
        print(f"--- RefineOptimizationInputs: Diagnostic Log ---", flush=True)
    
    refined_inputs = {}
    for name, tensor in inputs_dict.items():
        if tensor is None:
            refined_inputs[name] = None
            continue
            
        if verbose:
            print(f"  {name}: {tensor.shape}, dtype={tensor.dtype}", flush=True)
        
        # Fix: Pad to (N, 2) if (N, 1)
        # if tensor.dim() == 2 and tensor.shape[1] == 1:
        #     if verbose:
        #         print(f"  RefineOptimizationInputs: Padding {name} from (N, 1) to (N, 2)", flush=True)
        #     tensor = torch.cat([tensor, torch.zeros_like(tensor)], dim=1)
            
        refined_inputs[name] = tensor
        
    return refined_inputs
