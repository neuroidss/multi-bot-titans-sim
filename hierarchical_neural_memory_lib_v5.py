# Filename: hierarchical_neural_memory_lib_v5.py
# coding: utf-8
# Version: 5.0.8 (Refined external signal role logic)
# Description: Manages a hierarchy of NMM_V5 modules, handling BU/TD flow and external inputs.

import torch
from torch import nn, Tensor, is_tensor, cat, stack
import torch.nn.functional as F
from torch.nn import Module, ModuleList, Linear, Parameter, ParameterList
from torch.utils._pytree import tree_map
from collections import namedtuple
import copy
import traceback
import math
import time
from typing import Optional, Dict, Literal, Union, List, Tuple, Any

print("Hierarchical Neural Memory Library (Version 5.0.8) Loading...")

def exists(v): return v is not None
def default(*args):
    for arg in args:
        if exists(arg): return arg
    return None

class LayerNorm(Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__(); self.gamma = Parameter(torch.ones(dim)); self.beta = Parameter(torch.zeros(dim)); self.eps = eps
    def forward(self, x: Tensor) -> Tensor:
        if not x.is_floating_point(): x = x.float()
        mean = x.mean(-1, keepdim=True); std = x.std(-1, keepdim=True, unbiased=False)
        return self.gamma * ((x - mean) / (std.clamp(min=self.eps))) + self.beta

class MemoryMLP(Module):
    def __init__(self, dim: int, depth: int, expansion_factor: float = 2.0):
        super().__init__()
        if depth < 1: self.net = nn.Identity(); return
        dim_hidden = int(dim * expansion_factor); layers = []; current_dim = dim
        for i in range(depth):
            is_last = i == (depth - 1); out_dim = dim if is_last else dim_hidden
            layers.append(nn.Linear(current_dim, out_dim))
            if not is_last: layers.append(nn.GELU())
            current_dim = out_dim
        self.net = nn.Sequential(*layers); self._initialize_weights()
    def _initialize_weights(self):
        for m in self.net.modules():
             if isinstance(m, nn.Linear): nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias) if m.bias is not None else None
    def forward(self, x: Tensor) -> Tensor: return self.net(x)

NeuralMemState = namedtuple('NeuralMemState', ['seq_index', 'weights', 'optim_state'])

def _recursive_detach_clone_to_device(data: Any, device: Union[str, torch.device]) -> Any:
    target_device = device if isinstance(device, torch.device) else torch.device(device)
    if is_tensor(data):
        try: return data.detach().clone().to(target_device)
        except Exception: return data 
    if isinstance(data, dict): return {k: _recursive_detach_clone_to_device(v, target_device) for k, v in data.items()}
    if isinstance(data, list): return [_recursive_detach_clone_to_device(item, target_device) for item in data]
    if isinstance(data, tuple): return tuple(_recursive_detach_clone_to_device(item, target_device) for item in data)
    if isinstance(data, (int, float, bool, str)) or data is None: return data
    try: return copy.deepcopy(data)
    except Exception: return data

def mem_state_detach(state: NeuralMemState) -> NeuralMemState:
    if not isinstance(state, NeuralMemState): return state
    target_device = torch.device('cpu')
    try:
        if state.weights: first_weight = next((v for v in state.weights.values() if is_tensor(v)), None)
        if first_weight is not None: target_device = first_weight.device
    except Exception: pass
    try:
        detached_weights = _recursive_detach_clone_to_device(state.weights or {}, target_device)
        detached_optim_state = _recursive_detach_clone_to_device(state.optim_state or {}, target_device)
        return NeuralMemState(seq_index=state.seq_index, weights=detached_weights, optim_state=detached_optim_state)
    except Exception as e: print(f"CRITICAL ERROR in mem_state_detach: {e}\n{traceback.format_exc()}"); return state

class NeuralMemoryManagerTD_V5(Module):
    def __init__(self, dim: int, bu_input_dims: Dict[str, int], td_input_dims: Dict[str, int], level_name: str = "Unknown", mem_model_depth: int = 2, mem_model_expansion: float = 2.0, learning_rate: float = 0.01, weight_decay: float = 0.01, momentum_beta: float = 0.9, max_grad_norm: Optional[float] = 1.0, target_device: Union[str, torch.device] = 'cpu', td_modulates_input: bool = False, external_signal_dim: Optional[int] = None, external_signal_role: Literal['add_to_bu', 'add_to_td', 'add_to_target', 'none'] = 'add_to_bu', verbose: bool = False):
        super().__init__(); self.dim = dim; self.level_name = level_name; self.target_device = torch.device(target_device)
        self.max_grad_norm = max_grad_norm if max_grad_norm and max_grad_norm > 0 else None
        self.external_signal_dim = external_signal_dim if external_signal_dim and external_signal_dim > 0 else None
        self.external_signal_role = external_signal_role if self.external_signal_dim else 'none'; self.verbose = verbose
        self.memory_model_template = MemoryMLP(dim, mem_model_depth, mem_model_expansion)
        self.to_value_target_template = nn.Linear(dim, dim); nn.init.xavier_uniform_(self.to_value_target_template.weight, gain=0.1); nn.init.zeros_(self.to_value_target_template.bias) if self.to_value_target_template.bias is not None else None
        self.bu_projections = nn.ModuleDict(); self.td_projections = nn.ModuleDict()
        for name, s_dim in bu_input_dims.items(): lyr = nn.Linear(s_dim, dim); nn.init.xavier_uniform_(lyr.weight); nn.init.zeros_(lyr.bias) if lyr.bias is not None else None; self.bu_projections[name] = lyr
        for name, s_dim in td_input_dims.items(): lyr = nn.Linear(s_dim, dim); nn.init.normal_(lyr.weight, mean=0.0, std=0.02); nn.init.zeros_(lyr.bias) if lyr.bias is not None else None; self.td_projections[name] = lyr
        self.external_signal_projection = None
        if self.external_signal_dim: lyr = nn.Linear(self.external_signal_dim, dim); nn.init.normal_(lyr.weight, mean=0.0, std=0.15); nn.init.zeros_(lyr.bias) if lyr.bias is not None else None; self.external_signal_projection = lyr
        self.memory_model_template.to(self.target_device); self.to_value_target_template.to(self.target_device)
        self.bu_projections.to(self.target_device); self.td_projections.to(self.target_device)
        if self.external_signal_projection: self.external_signal_projection.to(self.target_device)
        self.optimizer_config = {'lr': learning_rate, 'betas': (momentum_beta, 0.999), 'weight_decay': weight_decay}
        self.loss_fn = nn.MSELoss(reduction='mean')
        if self.verbose: print(f"  NMM-TD V5 ({self.level_name}): Dim={dim}, ExtDim={self.external_signal_dim or 'N/A'}, Role={self.external_signal_role}")

    def get_initial_state(self) -> NeuralMemState:
        w = {f"net.{n}":p.clone().detach() for n,p in self.memory_model_template.named_parameters()}
        w.update({f"value_proj.{n}":p.clone().detach() for n,p in self.to_value_target_template.named_parameters()})
        for proj_n,proj_l in self.bu_projections.items(): w.update({f"bu_proj_{proj_n}.{n}":p.clone().detach() for n,p in proj_l.named_parameters()})
        for proj_n,proj_l in self.td_projections.items(): w.update({f"td_proj_{proj_n}.{n}":p.clone().detach() for n,p in proj_l.named_parameters()})
        if self.external_signal_projection: w.update({f"external_proj.{n}":p.clone().detach() for n,p in self.external_signal_projection.named_parameters()})
        return NeuralMemState(seq_index=0, weights={k:v.to(self.target_device) for k,v in w.items()}, optim_state={})

    def _apply_state_to_model(self, model_instance: Module, state_weights: dict, prefix: str = ""):
        if not state_weights or not model_instance: return
        prefix_dot = prefix + '.' if prefix else ""; prefix_len = len(prefix_dot)
        relevant_weights = {k[prefix_len:]: v for k, v in state_weights.items() if k.startswith(prefix_dot)}
        if not relevant_weights: return
        try:
            inst_dev = next(model_instance.parameters()).device; weights_on_dev = {k:v.to(inst_dev) for k,v in relevant_weights.items()}
            missing, unexpected = model_instance.load_state_dict(weights_on_dev, strict=False)
            if self.verbose and (missing or unexpected): print(f"Warning ({self.level_name}) Load '{prefix}': Missing={missing}, Unexpected={unexpected}")
        except StopIteration: pass 
        except Exception as e: print(f"CRITICAL ERROR ({self.level_name}) applying state to '{prefix}': {e}")

    def _create_or_load_optimizer(self, trainable_params: List[Parameter], state_optim_data: dict) -> Optional[torch.optim.Optimizer]:
        if not trainable_params: return None
        optimizer = torch.optim.AdamW(trainable_params, **self.optimizer_config)
        if state_optim_data and 'state' in state_optim_data and state_optim_data['state']:
            try: optimizer.load_state_dict(_recursive_detach_clone_to_device(state_optim_data, self.target_device))
            except Exception as e: print(f"Warning ({self.level_name}): Optimizer load failed: {e}")
        return optimizer

    @torch.no_grad()
    def _calculate_weight_change(self, old_weights: Dict[str, Tensor], new_weights: Dict[str, Tensor], prefix: str = "") -> Tensor:
        diff_sq_sum = torch.tensor(0.0, device=self.target_device, dtype=torch.float32)
        prefix_dot = prefix + '.' if prefix else ""
        keys_to_compare = {k for k in new_weights if k.startswith(prefix_dot)}
        for key in keys_to_compare:
            if key in old_weights and is_tensor(new_weights[key]) and is_tensor(old_weights[key]):
                 diff_sq_sum += torch.sum((new_weights[key] - old_weights[key].to(new_weights[key].device)).pow(2))
        return torch.sqrt(diff_sq_sum)

    def forward_step(self, bu_inputs: Dict[str, Tensor], td_signals: Dict[str, Tensor], current_state: NeuralMemState, external_signal: Optional[Tensor] = None, detach_next_state: bool = True) -> Tuple[Tensor, NeuralMemState, Tensor, Tensor, Tensor, Tensor, Tensor]:
        mem_model_inst = copy.deepcopy(self.memory_model_template); val_target_proj = copy.deepcopy(self.to_value_target_template)
        bu_proj_insts = copy.deepcopy(self.bu_projections); td_proj_insts = copy.deepcopy(self.td_projections)
        ext_proj_inst = copy.deepcopy(self.external_signal_projection) if self.external_signal_projection else None
        self._apply_state_to_model(mem_model_inst, current_state.weights, "net"); self._apply_state_to_model(val_target_proj, current_state.weights, "value_proj")
        for n, l in bu_proj_insts.items(): self._apply_state_to_model(l, current_state.weights, f"bu_proj_{n}")
        for n, l in td_proj_insts.items(): self._apply_state_to_model(l, current_state.weights, f"td_proj_{n}")
        if ext_proj_inst: self._apply_state_to_model(ext_proj_inst, current_state.weights, "external_proj")
        
        with torch.no_grad(): 
            old_w_clone = {f"net.{n}":p.detach().clone() for n,p in mem_model_inst.named_parameters()}
            if ext_proj_inst: old_w_clone.update({f"external_proj.{n}": p.detach().clone() for n, p in ext_proj_inst.named_parameters()})

        proj_bu = [bu_proj_insts[n](bu_t.to(self.target_device).squeeze(0).squeeze(0)) for n, bu_t in bu_inputs.items() if n in bu_proj_insts and is_tensor(bu_t)]
        comb_bu = torch.stack(proj_bu).sum(dim=0) if proj_bu else torch.zeros(self.dim, device=self.target_device)
        
        proj_td = [td_proj_insts[n](td_t.to(self.target_device).squeeze(0).squeeze(0)) for n, td_t in td_signals.items() if n in td_proj_insts and is_tensor(td_t)]
        comb_td = torch.stack(proj_td).sum(dim=0) if proj_td else torch.zeros(self.dim, device=self.target_device)
        
        proj_ext_sig = None
        proj_ext_norm = torch.tensor(0.0, device=self.target_device)
        if self.external_signal_role != 'none' and external_signal is not None and ext_proj_inst and is_tensor(external_signal) and external_signal.ndim==3 and external_signal.shape[:2]==(1,1) and external_signal.shape[-1]==self.external_signal_dim:
            proj_ext_sig_val = ext_proj_inst(external_signal.to(self.target_device).squeeze(0).squeeze(0))
            if proj_ext_sig_val is not None: # Ensure projection didn't return None
                 proj_ext_sig = proj_ext_sig_val
                 with torch.no_grad(): proj_ext_norm = torch.linalg.norm(proj_ext_sig.float()).detach()
        
        with torch.no_grad(): comb_bu_norm = torch.linalg.norm(comb_bu.float()).detach(); comb_td_norm = torch.linalg.norm(comb_td.float()).detach()

        key_base_for_prediction_target = comb_bu
        if self.external_signal_role == 'add_to_bu' and proj_ext_sig is not None:
            key_base_for_prediction_target = key_base_for_prediction_target + proj_ext_sig

        mem_input = comb_bu + comb_td 
        if self.external_signal_role == 'add_to_bu' and proj_ext_sig is not None:
             mem_input = mem_input + proj_ext_sig # Add to BU stream for mem_input as well
        elif self.external_signal_role == 'add_to_td' and proj_ext_sig is not None:
            mem_input = mem_input + proj_ext_sig
        
        with torch.no_grad():
            final_val_target = val_target_proj(key_base_for_prediction_target)
            if self.external_signal_role == 'add_to_target' and proj_ext_sig is not None:
                final_val_target = final_val_target + proj_ext_sig
            final_val_target_detached = final_val_target.detach()
            
        mem_model_inst.eval(); retrieved_val = mem_model_inst(mem_input).unsqueeze(0).unsqueeze(0)
        mem_model_inst.train()
        trainable_ps = list(p for p in mem_model_inst.parameters() if p.requires_grad)
        if ext_proj_inst and self.external_signal_role != 'none' and proj_ext_sig is not None: 
            trainable_ps.extend(list(p for p in ext_proj_inst.parameters() if p.requires_grad))
        
        optimizer = self._create_or_load_optimizer(trainable_ps, current_state.optim_state)
        
        new_w_final = {k: v.clone() for k, v in current_state.weights.items()}
        new_opt_final = current_state.optim_state 
        loss = torch.tensor(0.0, device=self.target_device)

        if optimizer and trainable_ps:
            optimizer.zero_grad(set_to_none=True); mem_out_for_loss = mem_model_inst(mem_input); loss = self.loss_fn(mem_out_for_loss, final_val_target_detached)
            if loss.requires_grad:
                try:
                    loss.backward()
                    if self.max_grad_norm: torch.nn.utils.clip_grad_norm_([p for p in trainable_ps if p.grad is not None], self.max_grad_norm)
                    optimizer.step()
                    with torch.no_grad():
                        for n, p in mem_model_inst.named_parameters(): new_w_final[f"net.{n}"] = p.detach().clone()
                        if ext_proj_inst and self.external_signal_role != 'none' and proj_ext_sig is not None: # Only update if it was trained
                            for n, p in ext_proj_inst.named_parameters(): new_w_final[f"external_proj.{n}"] = p.detach().clone()
                        new_opt_final = _recursive_detach_clone_to_device(optimizer.state_dict(), self.target_device)
                except Exception as e: 
                    print(f"ERROR ({self.level_name}) optim step: {e}"); loss = torch.tensor(float('inf'),device=self.target_device); 
                    new_opt_final = _recursive_detach_clone_to_device(current_state.optim_state, self.target_device)
        
        anomaly_score_t = loss.detach().clone()
        with torch.no_grad(): 
            weight_change_t = torch.tensor(0.0, device=self.target_device, dtype=torch.float32)
            wc_net = self._calculate_weight_change(old_w_clone, new_w_final, "net")
            wc_ext = torch.tensor(0.0, device=self.target_device, dtype=torch.float32)
            if ext_proj_inst and self.external_signal_role != 'none' and proj_ext_sig is not None:
                 wc_ext = self._calculate_weight_change(old_w_clone, new_w_final, "external_proj")
            weight_change_t = torch.sqrt(wc_net.pow(2) + wc_ext.pow(2))

        next_state_interm = NeuralMemState(current_state.seq_index + 1, new_w_final, new_opt_final)
        del mem_model_inst, val_target_proj, bu_proj_insts, td_proj_insts, ext_proj_inst, optimizer, old_w_clone
        next_state_final = mem_state_detach(next_state_interm) if detach_next_state else next_state_interm
        return (retrieved_val, next_state_final, anomaly_score_t, weight_change_t, comb_bu_norm, comb_td_norm, proj_ext_norm)

class HierarchicalSystemV5(Module):
    def __init__(self, level_configs: List[Dict[str, Any]], target_device: Union[str, torch.device] = 'cpu', verbose: bool = False):
        super().__init__(); self.level_configs = level_configs; self.num_levels = len(level_configs)
        self.target_device = torch.device(target_device); self.verbose = verbose; self.levels = ModuleList()
        self.level_name_to_index: Dict[str, int] = {}; self.dims: Dict[str, int] = {}
        # Store config for external inputs more directly for each level
        self.level_external_signal_configs: Dict[str, List[Dict[str, Any]]] = {}


        print(f"--- Initializing Hierarchical System (V5 - {self.num_levels} Levels) ---"); start_t = time.time()
        for i, cfg in enumerate(level_configs):
            lvl_name = cfg.get('name'); lvl_dim = cfg.get('dim')
            if not isinstance(lvl_name,str) or not lvl_name: raise ValueError(f"Lvl {i}: Invalid 'name'")
            if not isinstance(lvl_dim,int) or lvl_dim <= 0: raise ValueError(f"Lvl '{lvl_name}': Invalid 'dim'")
            if lvl_name in self.level_name_to_index: raise ValueError(f"Duplicate level name: '{lvl_name}'")
            self.level_name_to_index[lvl_name] = i; self.dims[lvl_name] = lvl_dim

            # Process external_input_config: it can be a list of dicts or a single dict
            ext_conf_raw = cfg.get('external_input_config')
            current_level_ext_signals = []
            if isinstance(ext_conf_raw, list): # If it's a list, iterate through it
                for item_conf in ext_conf_raw:
                    if isinstance(item_conf, dict):
                        src_name = item_conf.get('source_signal_name')
                        ext_dim = item_conf.get('dim')
                        # role = item_conf.get('role', 'add_to_bu') # Default role if not specified
                        if isinstance(src_name,str) and src_name and isinstance(ext_dim,int) and ext_dim > 0:
                            current_level_ext_signals.append({'source_signal_name': src_name, 'dim': ext_dim}) # Role handled by NMM
            elif isinstance(ext_conf_raw, dict): # If it's a single dict
                src_name = ext_conf_raw.get('source_signal_name')
                ext_dim = ext_conf_raw.get('dim')
                # role = ext_conf_raw.get('role', 'add_to_bu')
                if isinstance(src_name,str) and src_name and isinstance(ext_dim,int) and ext_dim > 0:
                     current_level_ext_signals.append({'source_signal_name': src_name, 'dim': ext_dim})
            
            if current_level_ext_signals:
                self.level_external_signal_configs[lvl_name] = current_level_ext_signals
                if self.verbose: print(f"  Level '{lvl_name}' configured for external signals: {current_level_ext_signals}")
        
        # Create NMM instances
        for i, cfg in enumerate(level_configs):
            lvl_name = cfg['name']; lvl_dim = self.dims[lvl_name]
            bu_srcs = cfg.get('bu_source_level_names',[]) or []; td_srcs = cfg.get('td_source_level_names',[]) or []
            nmm_params = cfg.get('nmm_params',{})
            
            for sl_name, sl_val in [("BU", bu_srcs),("TD",td_srcs)]:
                for src in sl_val:
                    if src not in self.level_name_to_index: raise ValueError(f"Lvl '{lvl_name}': Unknown {sl_name} src '{src}'")
            
            bu_dims_map = {}
            if not bu_srcs: 
                 raw_sensory_dim = cfg.get('raw_sensory_input_dim')
                 if not isinstance(raw_sensory_dim, int) or raw_sensory_dim <=0:
                     raise ValueError(f"Lvl '{lvl_name}' is a sensory level (no bu_source_level_names) but lacks a valid 'raw_sensory_input_dim' in its config.")
                 bu_dims_map[lvl_name] = raw_sensory_dim # Key by level name for sensory input
            else: 
                 bu_dims_map = {src_lvl_name: self.dims[src_lvl_name] for src_lvl_name in bu_srcs}

            td_dims_map = {src_lvl_name: self.dims[src_lvl_name] for src_lvl_name in td_srcs}
            
            current_nmm_params = nmm_params.copy()
            
            # NMM now handles multiple external signals internally based on its 'external_input_config'
            # We pass the *combined* dimension if NMM expects a single summed external input,
            # OR NMM needs to be adapted to take a dict of external signals.
            # For V5, NMM_TD_V5 expects a single `external_signal_dim` and `external_signal_role`.
            # This implies that if multiple signals target one level, they must be combined *before* NMM,
            # or the NMM logic needs to change to accept multiple named external inputs.
            # The current HNS config structure (list of external_input_config per level) suggests
            # that NMM_TD_V5 might need to be adapted.
            # For now, let's assume NMM_TD_V5 takes ONE external signal.
            # If multiple are configured, this will be an issue. The `configure_hns_external_inputs`
            # in server.py seems to add them as a list, which NMM_TD_V5 is not set up for.

            # Simplification: Assume NMM_TD_V5 is adapted to take external_input_config list from HNS.
            # However, NMM_TD_V5's __init__ still uses singular external_signal_dim & role.
            # This is a mismatch. The server.py's `configure_hns_external_inputs` creates a list of configs.
            # NMM_TD_V5's `external_signal_projection` is singular.

            # Let's adhere to NMM_TD_V5's current design: it takes one external signal.
            # The server's `configure_hns_external_inputs` should ensure only one signal (or a combined one)
            # is configured for a level if it uses the current NMM_TD_V5.
            # The current code in server.py *appends* to `external_input_config` list.
            # This implies NMM *should* handle a list, but it doesn't.

            # For this fix, let's assume server.py's HIERARCHY_LEVEL_CONFIGS is the source of truth
            # and NMM will eventually be updated, or server.py will only pass one.
            # The NMM params passed here will use what's in the HIERARCHY_LEVEL_CONFIGS for `external_signal_dim`
            # which `configure_hns_external_inputs` populates. If it's a list, NMM_TD_V5 will take the first one if not adapted.
            
            # Let's assume `external_input_config` on the level config *is* what NMM_TD_V5 will use directly.
            # NMM_TD_V5 currently has `external_signal_dim` and `external_signal_role` in its init.
            # The server.py sets `external_input_config` as potentially a list of dicts.
            
            # The simplest path is to ensure NMM_TD_V5 is initialized with what it expects based on the *first*
            # valid external signal config found for that level, if any.
            
            level_ext_configs = self.level_external_signal_configs.get(lvl_name, [])
            ext_dim_for_nmm = None
            # Role for NMM is now taken from nmm_params within level_configs,
            # which server.py's configure_hns_external_inputs might set.
            # Default role if not set.
            
            if level_ext_configs: # If there are external signals configured for this level
                # NMM_TD_V5 current design takes one external signal.
                # We'll use the first one specified in the list.
                # The 'role' should be part of nmm_params in HIERARCHY_LEVEL_CONFIGS.
                ext_dim_for_nmm = level_ext_configs[0]['dim'] # Use the dim of the first configured external signal
                # The role comes from current_nmm_params.external_signal_role (set by server.py if EEG, or default)
            
            current_nmm_params['external_signal_dim'] = ext_dim_for_nmm
            if 'external_signal_role' not in current_nmm_params: # if server.py didn't set it
                 current_nmm_params.setdefault('external_signal_role', 'add_to_bu' if ext_dim_for_nmm else 'none')

            if 'verbose' in current_nmm_params: # NMM's verbose, not HNS's
                del current_nmm_params['verbose']

            try: 
                self.levels.append(NeuralMemoryManagerTD_V5(
                    dim=lvl_dim, 
                    bu_input_dims=bu_dims_map, 
                    td_input_dims=td_dims_map, 
                    level_name=lvl_name, 
                    target_device=self.target_device, 
                    verbose=self.verbose, # Pass HNS verbose to NMM verbose
                    **current_nmm_params 
                ))
            except Exception as e: print(f"FATAL ERROR init NMM '{lvl_name}': {e}\n{traceback.format_exc()}"); raise
        print(f"--- Hierarchical System Initialized (V5) in {time.time()-start_t:.3f}s ---")

    def get_initial_states(self) -> List[NeuralMemState]: return [level.get_initial_state() for level in self.levels]
    
    def reset(self) -> None: 
        print("HierarchicalSystemV5.reset() called. Note: Bot-specific states are managed externally.")

    def step(self,
             current_bot_level_states: List[NeuralMemState],
             current_bot_last_step_outputs: Dict[str, Dict[str, Tensor]],
             sensory_inputs: Dict[str, Tensor], # Keyed by level name (for root sensory levels)
             external_inputs: Optional[Dict[str, Tensor]] = None, # Keyed by source_signal_name
             detach_next_states_memory: bool = True
            ) -> Tuple[
                Dict[str, Tensor],        
                List[NeuralMemState],     
                Dict[str, Tensor],        
                Dict[str, Tensor],        
                Dict[str, Tensor],        
                Dict[str, Tensor],        
                Dict[str, Tensor]         
            ]:
        
        next_bot_level_states_list = [None] * self.num_levels
        newly_retrieved_values_for_all_levels_dict: Dict[str, Tensor] = {}
        step_anomalies: Dict[str, Tensor] = {}
        step_weight_changes: Dict[str, Tensor] = {}
        step_bu_norms: Dict[str, Tensor] = {}
        step_td_norms: Dict[str, Tensor] = {}
        step_external_norms: Dict[str, Tensor] = {}

        # Prepare external inputs for each level based on its configuration
        # NMM_TD_V5 expects a single 'external_signal' tensor.
        # If a level is configured to receive multiple external signals (e.g. EEG + Rules),
        # they need to be combined here OR NMM_TD_V5 needs to be adapted.
        # For now, we'll pass the *first* configured external signal that is available.
        
        prepared_external_inputs_for_levels: Dict[str, Tensor] = {}
        if external_inputs:
            for lvl_idx, level_cfg in enumerate(self.level_configs):
                lvl_name = level_cfg['name']
                # Get the external signal config for this specific level from HNS's stored version
                level_specific_ext_configs = self.level_external_signal_configs.get(lvl_name, [])
                
                # NMM_TD_V5 expects ONE external signal. Find the first one available.
                # The actual signal to pass to NMM.forward_step
                actual_ext_signal_for_nmm_step: Optional[Tensor] = None

                if level_specific_ext_configs: # If this level is configured to receive any external signals
                    # The NMM instance for this level was initialized with a specific external_signal_dim
                    # which corresponds to ONE of these (usually the first, or one determined by server.py logic)
                    # We need to find which signal from `external_inputs` (keyed by source_signal_name)
                    # matches the one this NMM expects.
                    
                    # The NMM was configured based on HIERARCHY_LEVEL_CONFIGS[lvl_idx]['external_input_config']
                    # which might be a list. NMM_TD_V5 itself currently takes a single `external_signal_dim`.
                    # The `NeuralMemoryManagerTD_V5` was initialized with `external_signal_dim` corresponding
                    # to one of these. We need to find that one.
                    
                    # Let's assume the NMM's `external_signal_dim` and `external_signal_role` were set based on
                    # the *first* signal in its `external_input_config` list during HNS init.
                    
                    nmm_instance: NeuralMemoryManagerTD_V5 = self.levels[lvl_idx]
                    if nmm_instance.external_signal_dim and nmm_instance.external_signal_dim > 0 :
                        # Find which of the configured external signals for this level matches the NMM's expected dim
                        # This relies on the server.py correctly setting up HIERARCHY_LEVEL_CONFIGS so that
                        # nmm_params.external_signal_dim corresponds to a unique signal source name.
                        # Typically, server.py's `configure_hns_external_inputs` handles this by ensuring
                        # external_input_config on the HLC is updated.
                        
                        # The `level_cfg['external_input_config']` (from self.level_configs) reflects what NMM was built with.
                        # This might be a list. NMM_TD_V5 itself uses `self.external_signal_dim`.
                        
                        configured_ext_sources_for_this_nmm = level_cfg.get('external_input_config', [])
                        if not isinstance(configured_ext_sources_for_this_nmm, list):
                            configured_ext_sources_for_this_nmm = [configured_ext_sources_for_this_nmm] if configured_ext_sources_for_this_nmm else []

                        for nmm_ext_cfg_item in configured_ext_sources_for_this_nmm:
                            if isinstance(nmm_ext_cfg_item, dict):
                                source_name_for_nmm = nmm_ext_cfg_item.get('source_signal_name')
                                expected_dim_for_nmm = nmm_ext_cfg_item.get('dim')
                                
                                if source_name_for_nmm in external_inputs and expected_dim_for_nmm == nmm_instance.external_signal_dim:
                                    provided_tensor = external_inputs[source_name_for_nmm]
                                    if is_tensor(provided_tensor) and provided_tensor.ndim == 3 and \
                                       provided_tensor.shape[:2] == (1,1) and provided_tensor.shape[-1] == expected_dim_for_nmm:
                                        actual_ext_signal_for_nmm_step = provided_tensor.to(self.target_device)
                                        break # Found the one NMM is expecting for its single slot

                if actual_ext_signal_for_nmm_step is not None:
                     prepared_external_inputs_for_levels[lvl_name] = actual_ext_signal_for_nmm_step


        for i in range(self.num_levels):
            lvl_mgr: NeuralMemoryManagerTD_V5 = self.levels[i]
            cfg = self.level_configs[i]; lvl_n = cfg['name']
            bu_src_ns = cfg.get('bu_source_level_names', []) or []; td_src_ns = cfg.get('td_source_level_names', []) or []
            current_level_specific_state = current_bot_level_states[i]
            lvl_bu_in: Dict[str, Tensor] = {}; lvl_td_in: Dict[str, Tensor] = {}

            if not bu_src_ns: # This is a root sensory level
                sens_t = sensory_inputs.get(lvl_n) 
                # NMM expects bu_inputs to be keyed by source name. For sensory, source name IS level name.
                raw_sens_dim_cfg = cfg.get('raw_sensory_input_dim', self.dims[lvl_n]); exp_sh = (1, 1, raw_sens_dim_cfg)
                lvl_bu_in[lvl_n] = sens_t.to(self.target_device) if is_tensor(sens_t) and sens_t.shape == exp_sh else torch.zeros(exp_sh, device=self.target_device, dtype=torch.float32)
            else: # This level gets BU input from other levels
                for src_n in bu_src_ns: 
                    lvl_bu_in[src_n] = current_bot_last_step_outputs.get(src_n, {}).get('retrieved', 
                                     torch.zeros((1, 1, self.dims.get(src_n, 0)), device=self.target_device, dtype=torch.float32))
            
            for src_n in td_src_ns: 
                lvl_td_in[src_n] = current_bot_last_step_outputs.get(src_n, {}).get('retrieved', 
                                 torch.zeros((1, 1, self.dims.get(src_n, 0)), device=self.target_device, dtype=torch.float32))
            
            # Get the specific external signal for this level, if any was prepared
            lvl_ext_in_for_nmm_step = prepared_external_inputs_for_levels.get(lvl_n, None)

            try:
                ret_v, nxt_st, anom_s, wt_chg, bu_n, td_n, ext_n = lvl_mgr.forward_step(
                    lvl_bu_in, 
                    lvl_td_in, 
                    current_level_specific_state, 
                    lvl_ext_in_for_nmm_step, # Pass the single (or None) external signal
                    detach_next_state=detach_next_states_memory
                )
            except Exception as e:
                print(f"FATAL ERROR processing Lvl '{lvl_n}': {e}\n{traceback.format_exc()}")
                ret_v = torch.zeros((1, 1, self.dims[lvl_n]), device=self.target_device, dtype=torch.float32)
                nxt_st = mem_state_detach(current_level_specific_state) if detach_next_states_memory else current_level_specific_state
                anom_s, wt_chg, bu_n, td_n, ext_n = [torch.tensor(v, device=self.target_device, dtype=torch.float32) for v in [float('inf'), 0., 0., 0., 0.]]

            next_bot_level_states_list[i] = nxt_st; newly_retrieved_values_for_all_levels_dict[lvl_n] = ret_v
            step_anomalies[lvl_n] = anom_s.detach(); step_weight_changes[lvl_n] = wt_chg.detach()
            step_bu_norms[lvl_n] = bu_n.detach(); step_td_norms[lvl_n] = td_n.detach(); step_external_norms[lvl_n] = ext_n.detach()
            
        return (newly_retrieved_values_for_all_levels_dict, next_bot_level_states_list, step_anomalies, step_weight_changes, step_bu_norms, step_td_norms, step_external_norms)

print("Hierarchical Neural Memory Library (Version 5.0.8) Loaded Successfully.")

