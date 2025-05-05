# Filename: neural_memory_lib.py
# coding: utf-8

import torch
from torch import nn, Tensor, is_tensor, cat, stack
import torch.nn.functional as F
from torch.nn import Module, ModuleList, Linear, Parameter, ParameterList
from torch.utils._pytree import tree_map
from collections import namedtuple
import copy # For deep copying optimizer state safely
import traceback
import math # For weight init

print("Neural Memory Library Loading...")

# --- Helper Functions ---
def exists(v): return v is not None
def default(*args):
    for arg in args:
        if exists(arg): return arg
    return None

# --- LayerNorm (Corrected Implementation) ---
class LayerNorm(Module):
    def __init__(self, dim, eps=1e-5): # Added epsilon for stability
        super().__init__()
        # Learnable gain (gamma), initialized to 1
        self.gamma = Parameter(torch.ones(dim))
        # Learnable bias (beta), initialized to 0
        self.beta = Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        # Calculate mean and variance along the feature dimension (-1)
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True, unbiased=False) # Use population std dev

        # Normalize
        x_normalized = (x - mean) / (std + self.eps)

        # Scale and shift
        return self.gamma * x_normalized + self.beta


# --- MemoryMLP (Titans-Inspired - Corrected Init) ---
class MemoryMLP(Module):
    """ Simple MLP for the memory model """
    def __init__(self, dim, depth, expansion_factor = 2.):
        super().__init__()
        dim_hidden = int(dim * expansion_factor)
        layers = []
        current_dim = dim
        # print(f"  MemoryMLP: Dim={dim}, Depth={depth}, Hidden={dim_hidden}") # Reduce console noise
        if depth < 1:
            print("  Warning: MemoryMLP depth < 1, creating Identity.")
            self.net = nn.Identity()
            return

        for i in range(depth):
            is_last = i == (depth - 1)
            out_dim = dim if is_last else dim_hidden
            layers.append(nn.Linear(current_dim, out_dim))
            if not is_last:
                layers.append(nn.GELU()) # Activation between hidden layers
            current_dim = out_dim

        self.net = nn.Sequential(*layers)

        # Initialize weights (important for stability)
        for m in self.net.modules():
             if isinstance(m, nn.Linear):
                 # Xavier/Glorot initialization for layers with GELU
                 nn.init.xavier_uniform_(m.weight)
                 if m.bias is not None:
                     nn.init.zeros_(m.bias)
        # print(f"  MemoryMLP initialization complete.") # Reduce console noise


    def forward(self, x):
        return self.net(x)

# --- Neural Memory State ---
# Represents the persistent state of ONE bot's memory module
NeuralMemState = namedtuple('NeuralMemState', [
    'seq_index',        # Current sequence index processed by this memory instance
    'weights',          # Dictionary of Tensors: memory model parameters for this bot
    'optim_state',      # Dictionary: Optimizer state (e.g., momentum buffers for AdamW)
])

def _recursive_detach_clone_to_device(data, device):
    """ Helper to detach, clone, and move nested tensors in optimizer state """
    if is_tensor(data):
        return data.detach().clone().to(device)
    elif isinstance(data, dict):
        return {k: _recursive_detach_clone_to_device(v, device) for k, v in data.items()}
    elif isinstance(data, list):
        return [_recursive_detach_clone_to_device(item, device) for item in data]
    # Handle other potential types like tuples if necessary
    elif isinstance(data, tuple):
         return tuple(_recursive_detach_clone_to_device(item, device) for item in data)
    return data # Return non-tensor types as is

def mem_state_detach(state: NeuralMemState):
    """ Creates a detached copy of the NeuralMemState, ensuring tensors are on their original device """
    if not isinstance(state, NeuralMemState): return state

    target_device = None
    try:
        # Determine target device from the first weight tensor found
        if state.weights:
            first_weight_tensor = next(iter(state.weights.values()), None)
            if first_weight_tensor is not None:
                target_device = first_weight_tensor.device
            else: # Fallback if weights dict is empty but exists
                 print("Warning: mem_state_detach - Weights dict empty, cannot determine device. Assuming CPU.")
                 target_device = torch.device('cpu')
        else: # Fallback if weights dict is None or missing
             print("Warning: mem_state_detach - No weights found, cannot determine device. Assuming CPU.")
             target_device = torch.device('cpu')


        # Detach weights (parameters) - ensure they stay on the correct device
        detached_weights = {k: v.detach().clone().to(target_device) for k, v in state.weights.items()}

        # Detach optimizer state (more complex, needs deep copy and recursive tensor detachment)
        detached_optim_state = {}
        if state.optim_state:
            try:
                # Deep copy first to avoid modifying original state during detachment
                temp_optim_state = copy.deepcopy(state.optim_state)
                # Recursively detach/clone/move tensors within the copied state
                detached_optim_state = _recursive_detach_clone_to_device(temp_optim_state, target_device)

            except Exception as e:
                print(f"Warning: Error deep copying or detaching optimizer state: {e}. Optimizer state reset on detach.")
                # traceback.print_exc() # Uncomment for detailed debug
                detached_optim_state = {} # Reset on error

        return NeuralMemState(
            seq_index=state.seq_index,
            weights=detached_weights,
            optim_state=detached_optim_state,
        )
    except Exception as e:
        print(f"CRITICAL ERROR in mem_state_detach for state at seq_index {state.seq_index}: {e}")
        traceback.print_exc()
        # Return None to indicate failure, allows caller to handle
        return None


# --- Core NeuralMemory Module (Template & Instance Holder) ---
class NeuralMemoryManager(Module):
    """
    Holds the template architecture and provides methods to manage and update
    individual bot states using that architecture, designed for GPU execution.
    Implements the core learning logic inspired by Titans paper.
    """
    def __init__(
        self,
        dim,
        mem_model_depth=2,
        mem_model_expansion=2.,
        learning_rate=0.01,
        weight_decay=0.01,
        momentum_beta=0.9, # Beta1 for AdamW
        max_grad_norm=1.0,
        target_device='cpu' # The device where computations should happen
    ):
        super().__init__()
        self.dim = dim
        self.target_device = torch.device(target_device) # Ensure it's a torch.device object
        self.max_grad_norm = max_grad_norm if max_grad_norm is not None and max_grad_norm > 0 else None

        print(f"Initializing NeuralMemory Manager: Dim={dim}, Depth={mem_model_depth}, LR={learning_rate}, Device={self.target_device}, MaxGradNorm={self.max_grad_norm}")

        # --- Template Architecture (lives on target_device) ---
        # Memory Model (e.g., MLP) - The "Memory as Parameters" component M
        try:
            self.memory_model_template = MemoryMLP(
                dim=dim,
                depth=mem_model_depth,
                expansion_factor=mem_model_expansion
            ).to(self.target_device)

            # Projection for calculating target value 'v' from input 'x' (which acts as key 'k')
            # This represents the expected processing of the input stream element
            # In the paper's terms, W_v * x_t = v_t
            self.to_value_target_template = nn.Linear(dim, dim).to(self.target_device)
            nn.init.xavier_uniform_(self.to_value_target_template.weight)
            if self.to_value_target_template.bias is not None:
                nn.init.zeros_(self.to_value_target_template.bias)

            # In this simplified setup, the input 'x' itself acts as the key 'k'
            # If separate key generation was needed (e.g., k_t = x_t @ W_k), add another template here.
            # For simplicity, k_t = x_t.

        except Exception as e:
            print(f"FATAL ERROR during NNM template model initialization on {self.target_device}: {e}")
            traceback.print_exc()
            raise # Re-raise to prevent server from starting incorrectly

        # --- State Management Info ---
        # Get parameter names from the memory model template ONLY
        self.mem_param_names = list(dict(self.memory_model_template.named_parameters()).keys())
        if not self.mem_param_names:
             print("Warning: Memory model template has no parameters!")

        # --- Optimizer Configuration (used to create optimizer per bot for the memory model) ---
        self.optimizer_config = {
            'lr': learning_rate,
            'betas': (momentum_beta, 0.999), # AdamW betas (beta1=momentum)
            'weight_decay': weight_decay      # L2 regularization
        }

        # --- Loss Function (Associative Memory Loss as per paper eq 12) ---
        # loss = || M(k) - v ||^2, implemented as MSELoss
        self.loss_fn = nn.MSELoss(reduction='mean') # Use mean for stable gradients per step

        print(f"NeuralMemory Manager Templates Initialized on {self.target_device}")
        # print(f"Memory Model Param Names: {self.mem_param_names}") # Reduce noise

    def get_initial_state(self) -> NeuralMemState:
        """ Returns a new, initial state dictionary for a bot, with tensors on the target_device """
        # Clone initial weights from the template memory model
        initial_weights = {name: p.clone().detach().to(self.target_device)
                           for name, p in self.memory_model_template.named_parameters()}
        initial_optim_state = {} # Optimizer state starts empty
        return NeuralMemState(
            seq_index=0,
            weights=initial_weights,
            optim_state=initial_optim_state,
        )

    def _apply_state_to_model(self, model_instance: Module, state_weights: dict):
        """ Loads weights from the state dictionary into the provided model instance """
        # Ensure weights from state are on the same device as the model
        weights_on_device = {k: v.to(self.target_device) for k, v in state_weights.items()}
        try:
            # Make sure the state_dict keys match the model's parameter names
            current_model_keys = set(dict(model_instance.named_parameters()).keys())
            state_keys = set(weights_on_device.keys())
            if current_model_keys != state_keys:
                print(f"Warning: Mismatch between model keys ({current_model_keys}) and state keys ({state_keys}). Attempting non-strict load.")
                missing_keys, unexpected_keys = model_instance.load_state_dict(weights_on_device, strict=False)
                if missing_keys or unexpected_keys:
                     print(f"  Non-strict load result: Missing={missing_keys}, Unexpected={unexpected_keys}")
            else:
                model_instance.load_state_dict(weights_on_device, strict=True)

        except Exception as e:
            print(f"ERROR loading model state: {e}. Check architecture match.")
            traceback.print_exc()
            raise ValueError("Failed to load model state due to architecture mismatch or other error.") from e


    def _create_or_load_optimizer(self, model_instance: Module, state_optim_data: dict):
        """ Creates an AdamW optimizer for the model instance and loads state """
        # IMPORTANT: Optimizer must be created for the *specific parameters* of the instance
        optimizer = torch.optim.AdamW(model_instance.parameters(), **self.optimizer_config)
        if state_optim_data and 'state' in state_optim_data: # Check if state exists and has the 'state' key
            try:
                # Deepcopy state_dict, ensuring tensors are correctly handled for the device
                optim_state_to_load = copy.deepcopy(state_optim_data)

                # Recursively move tensors in the state dict to the target device
                optim_state_to_load = _recursive_detach_clone_to_device(optim_state_to_load, self.target_device)

                optimizer.load_state_dict(optim_state_to_load)
                # print(f"Optimizer state loaded successfully.") # Debug
            except Exception as e:
                print(f"Warning: Failed to load optimizer state, reinitializing. Error: {e}")
                # traceback.print_exc() # More detailed debug if needed
                optimizer = torch.optim.AdamW(model_instance.parameters(), **self.optimizer_config) # Recreate on error
        # else:
            # print("No optimizer state provided or state key missing, initializing new optimizer.") # Debug
        return optimizer

    @torch.no_grad() # Ensure weight diff calculation doesn't track gradients
    def _calculate_weight_change(self, old_weights, new_weights):
        """ Calculates the L2 norm of the difference between two weight dictionaries (on GPU). """
        total_diff_sq_sum = torch.tensor(0.0, device=self.target_device)
        # Iterate over the keys of the *memory model only*
        for name in self.mem_param_names:
            if name in old_weights and name in new_weights:
                diff = new_weights[name] - old_weights[name]
                total_diff_sq_sum += torch.sum(diff * diff)
            else:
                 print(f"Warning: Param '{name}' missing in old or new weights during diff calc.")
        return torch.sqrt(total_diff_sq_sum) # Return scalar tensor on GPU

    def forward_step(self, x: Tensor, current_state: NeuralMemState, detach_next_state=True):
        """
        Processes one step for a bot using its specific state on the GPU, implementing the
        Titans-inspired "Learning to Memorize at Test Time" concept.

        Args:
            x (Tensor): Input stream tensor [1, 1, dim], MUST be on self.target_device.
                        This tensor acts as the 'key' (k_t) in the associative memory context.
            current_state (NeuralMemState): The bot's current state. Tensors MUST be on self.target_device.
            detach_next_state (bool): If True, the returned next_state will have detached tensors.

        Returns:
            tuple[Tensor, NeuralMemState, Tensor, Tensor, Tensor]:
                - retrieved_value (Tensor): Value retrieved M*(k_t) *before* the update [1, 1, dim], on target_device.
                - next_state (NeuralMemState): Updated state after learning. Tensors on target_device.
                - anomaly_score (Tensor): Scalar loss tensor ||M(k_t) - v_t||^2 calculated for this step, on target_device.
                - weight_change_metric (Tensor): Scalar L2 norm of weight difference ||W_t - W_{t-1}||, on target_device.
                - input_stream_vector (Tensor): Reference to the input tensor x (k_t), on target_device.
        """
        if x.dim() != 3 or x.shape[0] != 1 or x.shape[1] != 1 or x.shape[2] != self.dim:
             raise ValueError(f"Input shape error: Expected [1, 1, {self.dim}], got {x.shape}")
        if x.device != self.target_device:
            x = x.to(self.target_device)

        # Use temporary model instances on the target device for this step's computation
        # Deepcopy ensures templates are not modified and avoids interference between bot steps if run concurrently (though current loop is sequential)
        mem_model_instance = copy.deepcopy(self.memory_model_template).to(self.target_device)
        value_target_proj = self.to_value_target_template # Use the shared template directly (it's just inference)

        # 1. Apply Bot's Current Weights (W_{t-1}) to the temporary memory model instance
        try:
            self._apply_state_to_model(mem_model_instance, current_state.weights)
        except ValueError as e:
             print(f"FATAL: Cannot apply state to model for bot at seq_index {current_state.seq_index}. Aborting step.")
             raise e # Propagate error

        # --- Store old weights (W_{t-1}) for calculating the change metric ---
        old_weights_clone = {name: p.clone() for name, p in mem_model_instance.named_parameters()}

        # 2. Retrieval Phase (Inference M*(k_t)) - Get output BEFORE learning
        # This represents the memory's current prediction for the input 'x' (acting as key k_t)
        mem_model_instance.eval() # Set to evaluation mode
        with torch.no_grad():
            # Input x is k_t, shape [1, 1, dim]. Squeeze for model input [1, dim].
            query_key = x.squeeze(0) # Shape [1, dim]
            retrieved_value = mem_model_instance(query_key) # M*(k_t), Shape [1, dim]
            retrieved_for_return = retrieved_value.unsqueeze(0) # Shape [1, 1, dim] for consistent output

        # 3. Learning Phase - Update the memory model's weights based on prediction error
        mem_model_instance.train() # Set back to training mode
        # Create an optimizer instance specifically for *this step* acting on the *temporary model*
        optimizer = self._create_or_load_optimizer(mem_model_instance, current_state.optim_state)
        optimizer.zero_grad(set_to_none=True) # Reset gradients for this learning step

        # 4. Calculate Loss (Prediction Error = Anomaly Score)
        # Input x is the key, k_t. Shape [1, 1, dim] -> [1, dim]
        key = x.squeeze(0) # Shape [1, dim]

        # Calculate the TARGET value v_t = W_v * k_t (using the shared projection layer)
        # This represents what the memory *should* have output for this input, according to the projection layer
        with torch.no_grad(): # Target calculation should not affect gradients of the memory model
             value_target = value_target_proj(key) # Target v_t, Shape [1, dim]

        # Calculate the memory model's output M(k_t) *again*, this time tracking gradients
        memory_output_for_loss = mem_model_instance(key) # M(k_t), Shape [1, dim]

        # Calculate the loss: ||M(k_t) - v_t||^2 (Equation 12 from Titans paper)
        loss = self.loss_fn(memory_output_for_loss, value_target)
        anomaly_score_tensor = loss.detach().clone() # Store scalar loss value before backward pass destroys it

        # 5. Backpropagation - Compute Gradients based on the anomaly/loss
        loss.backward()

        # 6. Gradient Clipping (Optional but recommended)
        if self.max_grad_norm is not None:
            try:
                # Clip gradients in-place on the temporary model's parameters
                torch.nn.utils.clip_grad_norm_(mem_model_instance.parameters(), self.max_grad_norm)
            except Exception as clip_err:
                print(f"Warning: Gradient clipping failed: {clip_err}")

        # 7. Optimizer Step - Update Weights (W_t = W_{t-1} - lr * gradient)
        optimizer.step()
        # Now mem_model_instance holds the updated weights W_t

        # --- Calculate Weight Change Metric ||W_t - W_{t-1}|| ---
        new_weights_clone = {name: p.clone() for name, p in mem_model_instance.named_parameters()}
        weight_change_metric_tensor = self._calculate_weight_change(old_weights_clone, new_weights_clone)

        # 8. Capture New State (W_t and optimizer state) from the *updated* temporary model
        # Detach the updated weights before storing them in the state
        new_weights = {name: p.clone().detach().to(self.target_device)
                       for name, p in mem_model_instance.named_parameters()}

        # Get the optimizer state *after* the step
        new_optim_state_raw = optimizer.state_dict()
        # Deep copy and ensure all tensors in optimizer state are detached and on the correct device
        new_optim_state_detached = copy.deepcopy(new_optim_state_raw) # Start with deepcopy
        # Recursively detach/clone/move tensors within the copied state
        new_optim_state = _recursive_detach_clone_to_device(new_optim_state_detached, self.target_device)


        next_seq_index = current_state.seq_index + 1

        # Create the final next_state tuple
        next_state_final = NeuralMemState(
            seq_index=next_seq_index,
            weights=new_weights,
            optim_state=new_optim_state,
        )

        # Cleanup temporary resources explicitly (helps garbage collection)
        del mem_model_instance
        del optimizer
        del old_weights_clone
        del new_weights_clone

        # 9. Optionally detach the final state before returning
        if detach_next_state:
            next_state_final_detached = mem_state_detach(next_state_final)
            if next_state_final_detached is None: # Handle detachment failure
                 print("CRITICAL: Detachment of next state failed. Returning current state detached.")
                 # Fallback: return the original state detached, but signal error via metrics
                 original_detached = mem_state_detach(current_state)
                 zero_scalar = torch.tensor(0.0, device=self.target_device)
                 # Return original retrieved value, original state detached, zero metrics
                 return retrieved_for_return, original_detached, zero_scalar, zero_scalar, x
            else:
                 next_state_final = next_state_final_detached


        # Return: retrieved value (before update), new state, anomaly score, weight change, input vector
        # All tensors returned should be on self.target_device
        return retrieved_for_return, next_state_final, anomaly_score_tensor, weight_change_metric_tensor, x

print("Neural Memory Library Loaded Successfully.")

