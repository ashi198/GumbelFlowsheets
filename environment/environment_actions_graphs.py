# environment_actions_graph.py

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
from core.abstract import BaseTrajectory
from core.utils import softmax
import numpy as np
from environment.flowsheet_simulation_graph import FlowsheetSimulationGraph
import copy, torch
from torch import nn
import traceback, random, string
from experts.experts import DistillationColumn, Decanter, Mixer, Split, Recycler, AddSolvent, EdgeFlowExpert, OpenStreamExpert, FlowExpert, ComponentsExpert



class FlowsheetDesign:

    """
    Graph-based action environment:

      Level 0: choose OPEN stream (or index 0 = TERMINATE)
      Level 1: choose UNIT type 
      Level 2: choose parameters for the following UNITS <distillation_column, add_solvent, decanter, split, recycle, mixer>
      Level 3: choose further parameters for <add_solvent, mixer>

    After an action completes successfully:
      - placement/wiring happens (add_unit / add_recycle)
      - simulate() runs
      - NPV is recomputed
      - state resets to level 0

    On any failure:
      - we roll back partial changes (including recycle edges)
      - reset to level 0 (no loops)
    """

    def __init__(self, random_instance: Dict[str, Any], gen_config, env_config):
        self.gen_config = gen_config
        self.env_config = env_config 
        self.sim = FlowsheetSimulationGraph(random_instance, self.env_config)
        self.num_units = len(self.env_config.unit_types)
        self.problem_instance = random_instance

        # add feed nodes
        self.sim.feed_nodes = []
        for feed in random_instance["list_feed_streams"]:
            self.sim.feed_nodes.append(self.sim.add_feed(feed))

        # current action state
        self.level = 0 
        self.failed_simulator_call = 0
        self.valid_nodes: torch.Tensor = None

        # counters (limits)
        self.counts = {
            "distillation_column": 0,
            "decanter": 0,
            "split": 0,
            "mixer": 0,
            "recycle": 0,
            "add_solvent": 0,
        }

        # define all unit experts to get embeddings 
        self.unit_experts = nn.ModuleDict({
            "distillation_column": DistillationColumn(self.gen_config), 
            "decanter": Decanter(self.gen_config),   
            "mixer": Mixer(self.gen_config),
            "split": Split(self.gen_config),
            "recycle": Recycler(self.gen_config), 
            "add_solvent": AddSolvent(self.gen_config, self.env_config),      
            "flow_expert": FlowExpert(self.gen_config, self.env_config),
            "component_expert": ComponentsExpert(self.gen_config, self.env_config)
            }).to(self.gen_config.training_device)

        self.edge_expert = EdgeFlowExpert(self.gen_config, self.unit_experts["flow_expert"]).to(self.gen_config.training_device)
        self.open_stream_expert = OpenStreamExpert(self.gen_config, self.env_config, self.unit_experts["flow_expert"]).to(self.gen_config.training_device)
        self.total_units_placed = 0
        self.history: List[int] = []
        self.level_list: List[int] = []
        self._action_seq_start = 0

        self.current_action_mask: Optional[np.array] = None # The action mask indicates before each action what is feasible at the current level.

        # limits on units 
        self.max_total_units = self.env_config.action_limits[self.problem_instance['system_name']]['max_total_units'] # overall cap on placed units (excluding feed)
        self.min_total_units = self.env_config.action_limits[self.problem_instance['system_name']]['min_total_units']
        self.max_distillation_columns = self.env_config.action_limits[self.problem_instance['system_name']]['max_distillation_columns']
        self.max_decanters = self.env_config.action_limits[self.problem_instance['system_name']]['max_decanters']
        self.max_split = 0
        self.max_mixer = 0
        self.max_recycle = self.env_config.action_limits[self.problem_instance['system_name']]['max_recycle']
        self.max_solvent = self.env_config.action_limits[self.problem_instance['system_name']]['max_solvent']

        # initial simulate to populate open streams/NPV
        self.sim.simulate()
        self.objective = None        
        self.identifier: str = None
        self.current_state = self.get_current_state()
        self.get_feasible_actions()

    def get_current_state(self) -> Dict[str, Any]:
        state = {
            "current_level": self.level,
            "open_streams": self._enumerate_open_streams(), # # (node_id, label) of exisiting streams 
            "chosen_open_stream": None, # Optional[Tuple[int, str]] # (node_id, label)
            "chosen_unit": None, # Optional[Tuple[int, str]] 

            "pending_params": {
            "distillation_column": None,
            "split": None,
            "recycle": None, 
            "add_solvent": {
                "index_for_comp": None,
                "name_comp": None,
                "index_for_amount": None,
                "amount_value": None},
            "mixer": None}, #Optional[Dict] = (int, int) # tuple of index, value of the 2nd o/p stream
            
            "npv_raw": None, #raw values of nvp simulation
            "npv_norm": None, #norm values 
            "completed_design": None, # True only when termination is selected
            "second_open_stream_dest_node": None, #Optional[int] 
            "second_open_stream": None, #Optional[Tuple[int, str]]
            "recycle_dest_unit": None, #Optional[int]
            "current_action_mask": self.current_action_mask,
                            }
        return state

    def get_feasible_actions(self) -> np.ndarray:

        """
        Return a 0/1 vector mask feasible actions for the current level:

        Level 0: indices enumerate all open streams, + 1 for "terminate"
        Level 1: indices enumerate unit choices (0...num_units-1) 
        Level 2: select parameters values for the following units (distillation column: DF value (100), split: split_ratio (100), 
        mixer: 2nd open node (N = num of nodes within the graph), add_solvent: select a compound (from 5 options)
        recycler: destination node (N = num of nodes within the graph))
        Level 3: select amount for selected compound for add_solvent (100) or select open stream for the selected node for mixer 

        0: the action is masked. 1 = action is allowed

        """
        if self.level == 0:
            total_limit_reached = self.total_units_placed >= getattr(self, "max_total_units", 9999) 
            open_streams = self._enumerate_open_streams()
            mask = np.zeros(len(open_streams) + 1, dtype=int)
            
            if total_limit_reached:
                # only recycler add mode
                for i, stream in enumerate(open_streams):
                    if self._stream_has_valid_recycle(stream):
                        mask[i + 1] = 1
            else:
                mask[1:] = 1

            # masking condition for lvl 0 (terminate)
            if self.forced_termination():   
                mask[0] = 1
                mask[1:] = 0

            elif self.optional_termination():
                if total_limit_reached:
                    mask[0] = 1
                else:
                    mask[:] = 1 #terminate only becomes valid after reaching a minimum num of units placed

            # enable all available stream slots (everything is available)
            self.current_state["open_streams"] = open_streams
            self.current_action_mask = mask 
            return mask   

        # select units now 
        elif self.level == 1:
            unit_params_mask = np.zeros(self.env_config.num_units, dtype=int)
            open_streams = self._enumerate_open_streams()
            for idx, unit_name in enumerate(self.env_config.units_map_indices_type):
                # allow only avail units
                avail_unit = self._unit_available(unit_name, idx)
                if unit_name not in ["mixer", "recycle"] and avail_unit: 
                    unit_params_mask[idx] = 1

                if unit_name == "mixer" and avail_unit:
                    # need at least 2 open streams
                    if len(self._enumerate_open_streams()) < 2:
                        continue
                    else:
                        src_node, _ = self.current_state["chosen_open_stream"]
                        candidates = self._enumerate_open_streams_excluding(exclude = self.current_state["chosen_open_stream"])
                        candidate_nodes = sorted({i for i, _ in candidates if i != src_node})
                        if len(candidate_nodes) < 1:
                            continue 
                        else:
                            unit_params_mask[idx] = 1
                if unit_name == "recycle" and avail_unit:
                    # need a chosen stream (comes from level 0) and at least one eligible des

                    '''eligible_recycle_destinations = self._eligible_recycle_destinations()
                    self.eligible_recycle_destinations = eligible_recycle_destinations
                    if self.current_state["chosen_open_stream"] is None or len(eligible_recycle_destinations) == 0:
                        continue
                    else:
                        unit_params_mask[idx] = 1'''
                    
                    if self._stream_has_valid_recycle(self.current_state["chosen_open_stream"]) or self._is_pure_solvent_stream(self.current_state["chosen_open_stream"]):
                        unit_params_mask[idx] = 1

            self.current_action_mask = unit_params_mask
            return unit_params_mask

        elif self.level == 2:
            open_streams = self.current_state["open_streams"]
            _, chosen_unit_name = self.current_state["chosen_unit"]

            if chosen_unit_name in ["distillation_column", "split"]:
                params_mask = np.ones(100, dtype=int)
            elif chosen_unit_name == "add_solvent":
                params_mask = np.zeros(len(self.env_config.component_names), dtype= int)
                for i in self.problem_instance['possible_ind_add_comp']:
                    params_mask[i] = 1
            elif chosen_unit_name == "recycle":
                node_ids = list(self.sim.graph.nodes)
                source_node, _ = self.current_state["chosen_open_stream"]
                id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
                params_mask = np.zeros(len(node_ids), dtype=int) 

                if self._stream_has_valid_recycle(self.current_state["chosen_open_stream"]):
                    recycle_dests = self._eligible_recycle_destinations()
                    for i in recycle_dests:
                        if i != source_node:
                            idx = id_to_idx[i]
                            params_mask[idx] = 1
                    if not np.any(params_mask):
                        print('goofup here')

            elif chosen_unit_name == "mixer":
                src_node, _ = self.current_state["chosen_open_stream"]
                candidates = self._enumerate_open_streams_excluding(exclude = self.current_state["chosen_open_stream"])
                node_ids = list(self.sim.graph.nodes)
                id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
                params_mask = np.zeros(len(node_ids), dtype=int) # number of nodes present in the graph
                for i, _ in candidates:
                    if i != src_node:
                        idx = id_to_idx[i]
                        params_mask[idx] = 1 
            
            self.current_action_mask = params_mask
            return params_mask

        elif self.level == 3:
            _, unit_name = self.current_state["chosen_unit"]
            if unit_name == "add_solvent":
                params_mask =  np.ones(100, dtype=int)

            if unit_name == "mixer":
                src_node, _ = self.current_state["chosen_open_stream"]
                candidates = self._enumerate_open_streams_excluding(exclude = self.current_state["chosen_open_stream"])
                params_mask = np.zeros(self.env_config.max_outlets, dtype=int)
                for i, name in candidates:
                    if i != src_node and i == self.current_state['second_open_stream_dest_node']:
                        if name == 'out0':
                            params_mask[0] = 1
                        elif name == 'out1':
                            params_mask[1] = 1
            
            self.current_action_mask = params_mask # decide which outlet to select for available nodes

            return params_mask
        
        return np.array([], dtype=int)

    def take_action(self, action_index: int, next_level: None) -> Tuple[bool, float, bool]:

        """
        A action index that agent takes across different levels. 
        Actions can be taken across 4 levels in a hierarchical way. 

        Level 0: choose either to terminate design <index 0> or select an open stream 
        Level 1: For a given stream, select a unit type: <distillation_column, add_solvent, decanter, split, recycle, mixer>
        Level 2: Select parameters values for the following units 
         -- distillation column: select a distillation fraction value 
         -- decanter: nothing 
         -- split: select split ratio
         -- recycle: select the destination node, given a source node 
         -- mixer: select the destination node with open streams 
         -- add_solvent: select a compound between 5 global available compounds 
        
        Level 3: Further parameter selection for units add_solvent and mixer 
        -- select amount for selected compound for add_solvent 
        -- select open stream for the selected node for mixer 

        Returns:
          finished_design (bool) = true if termination chosen or max units reached
          reward (float) = current (raw) NPV after completing an action
          move_worked (bool) = false if placement failed (e.g., PEQ failure / convergence fail)

        """

        assert not self.current_state['completed_design'], "Taking action on an already terminated design!"
        
        assert self.current_action_mask[action_index] == 1, \
            f"Trying to take action {action_index} on level {self.level}, but it is set to infeasible"
        
        if action_index >= len(self.current_action_mask):
            raise ValueError(f"Invalid action {action_index}, mask size {len(self.current_action_mask)}")

        try:
            action_index = int(action_index)
            self.current_state['current_level'] = self.level 
            self.current_state['current_action_mask'] = self.current_action_mask
            open_streams = self._enumerate_open_streams()
            self.current_state['open_streams'] = open_streams

            if self.level == 0:
                self._action_seq_start = len(self.history)
                if action_index == 0: 
                    self.current_state['completed_design'] = True 
                    self.level_list.append(self.level)
                    self.history.append(action_index)
                    self.identifier = self.generate_custom_flowsheet_id()
                    self.literature_flowsheet_similarity_check()

                    return True, self.objective, True

                # if not terminate index, then open_stream selected 
                selected_stream = open_streams[action_index - 1], 
                self.current_state['chosen_open_stream'] = selected_stream[0]
                self.level_list.append(self.level)
                self.level = 1 if next_level == None else next_level
                self.history.append(action_index)
                self.get_feasible_actions()
                return False, -1000.0, True

            elif self.level == 1:
                unit_idx = action_index
                if unit_idx < 0 or unit_idx >= self.env_config.num_units:
                    raise ValueError("Illegal unit index.")
                unit_name = self.env_config.units_map_indices_type[unit_idx]
                if not self._unit_available(unit_name, unit_idx):
                    raise ValueError("Unit not available due to limits or feasibility.")
            
                if unit_name == "distillation_column":
                    # add a new category for DF ratio
                    #self.current_state["pending_params"]["distillation_column"] = None # tuple of index, value of the chosen DF 
                    self.level_list.append(self.level)
                    self.current_state['chosen_unit'] = (unit_idx, unit_name)
                    self.history.append(action_index)
                    self.level = 2 if next_level == None else next_level
                    self.get_feasible_actions()
                    return False, -1000.0, True
                 
                if unit_name == "split":
                    # add a new category for DF ratio
                    #self.current_state["pending_params"]["split"] = None # tuple of index, value of the chosen ratio
                    self.level_list.append(self.level)
                    self.current_state['chosen_unit'] = (unit_idx, unit_name)
                    self.history.append(action_index)
                    self.level = 2 if next_level == None else next_level
                    self.get_feasible_actions()
                    return False, -1000.0, True

                if unit_name == "add_solvent":
                    #self.current_state["pending_params"]["add_solvent"] = {}
                    self.level_list.append(self.level)
                    self.current_state['chosen_unit'] = (unit_idx, unit_name)
                    self.history.append(action_index)
                    self.level = 2 if next_level == None else next_level
                    self.get_feasible_actions()
                    return False, -1000.0, True

                # recycle (destination selection)
                if unit_name == "recycle":
                    self.level_list.append(self.level)
                    self.current_state['chosen_unit'] = (unit_idx, unit_name)
                    self.history.append(action_index)
                    self.level = 2 if next_level == None else next_level
                    self.get_feasible_actions()
                    return False, -1000.0, True
                
                if unit_name == "mixer":
                    self.level_list.append(self.level)
                    self.current_state['chosen_unit'] = (unit_idx, unit_name)
                    self.history.append(action_index)
                    self.level = 2 if next_level == None else next_level
                    self.get_feasible_actions()
                    return False, -1000.0, True 

                if unit_name == "decanter":
                    # immediate place (no continuous param, no second stream)
                    self.level_list.append(self.level)
                    self.history.append(action_index)
                    self.current_state['chosen_unit'] = (unit_idx, unit_name)                  
                    done, reward, worked = self._complete_action_place_and_simulate()
                    self.current_state["completed_design"] = done 
                    self.objective = reward 
                    self.current_state['open_streams'] = self._enumerate_open_streams()
                    self.level = 0 if next_level == None else next_level
                    self.get_feasible_actions()
                    return done, reward, worked

            elif self.level == 2:
                # for distillation, decide on which DF ratio value to choose 
                if self._chosen_unit_name() == "distillation_column":
                    if action_index > len (self.env_config.DF_distillation_map):
                        raise ValueError("Distillation fraction value selected more than the permissible limit.")
                    else:
                        self.current_state["pending_params"]["distillation_column"] = (action_index, self.env_config.DF_distillation_map[action_index])
                        self.history.append(action_index)
                        self.level_list.append(self.level)
                        done, reward, worked = self._complete_action_place_and_simulate()
                        self.current_state["completed_design"] = done 
                        self.objective = reward
                        self.current_state['open_streams'] = self._enumerate_open_streams()

                        self.level = 0 if next_level == None else next_level
                        self.get_feasible_actions()
                        return done, reward, worked

                # for selecting split ratio 
                elif self._chosen_unit_name() == "split":
                    if action_index > len(self.env_config.split_ratio_map):
                        raise ValueError("Split ratio value selected more than the permissible limit.")
                    else:
                        self.level_list.append(self.level)
                        self.current_state["pending_params"]["split"] = (action_index, self.env_config.split_ratio_map[action_index])
                        self.history.append(action_index)
                        done, reward, worked = self._complete_action_place_and_simulate()
                        self.current_state["completed_design"] = done 
                        self.objective = reward 
                        self.current_state['open_streams'] = self._enumerate_open_streams()
                        self.level = 0 if next_level == None else next_level
                        self.get_feasible_actions()
                        return done, reward, worked
                    
                elif self._chosen_unit_name() == "add_solvent":
                    self.chosen_add_solvent_comp = self.env_config.component_names[action_index]
                    self.current_state["pending_params"]["add_solvent"] = {
                        "index_for_comp": action_index,
                        "name_comp": self.chosen_add_solvent_comp,
                        "index_for_amount": None,
                        "amount_value": None,
                    }
                    self.level_list.append(self.level)
                    self.history.append(action_index)

                    self.level = 3 if next_level == None else next_level
                    self.get_feasible_actions()
                    return False, -1000.0, True
                        
                # choose destination stream for recycle
                elif self._chosen_unit_name() == "recycle":
                    dests = self._eligible_recycle_destinations()
                    node_ids = list(self.sim.graph.nodes)

                    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
                    idx_to_id = {v: k for k, v in id_to_idx.items()}
                    if len(dests) == 0:
                        raise ValueError("No destination stream available for recycle ")
                    else:
                        # select destination and simulate 
                        if idx_to_id[action_index] in dests:
                            self.current_state["pending_params"]["recycle"] = idx_to_id[action_index] 
                            self.current_state["recycle_dest_unit"] = idx_to_id[action_index] 
                        else:
                            raise RuntimeError("Recycle requires a destination unit to be chosen.")
                        
                        self.history.append(idx_to_id[action_index])
                        self.level_list.append(self.level)
                        done, reward, worked = self._complete_action_place_and_simulate()
                        self.current_state["completed_design"] = done 
                        self.objective = reward
                        self.level = 0 if next_level == None else next_level
                        self.get_feasible_actions()
                        self.current_state['open_streams'] = self._enumerate_open_streams()
                        return done, reward, worked

                # mixer: choose second stream
                elif self._chosen_unit_name() == "mixer":
                    candidates = self._enumerate_open_streams_excluding(exclude = self.current_state["chosen_open_stream"])
                    src_node, _ = self.current_state["chosen_open_stream"]
                    candidate_nodes = sorted({i for i, _ in candidates if i != src_node})
                    node_ids = list(self.sim.graph.nodes)
                    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
                    idx_to_id = {v: k for k, v in id_to_idx.items()}
                    
                    if idx_to_id[action_index] in candidate_nodes:
                        self.current_state["second_open_stream_dest_node"] = idx_to_id[action_index] 

                    if self.current_state["second_open_stream_dest_node"] is None:
                        raise RuntimeError("Selection for mixer node not valid")
                    
                    self.level_list.append(self.level)
                    self.history.append(idx_to_id[action_index]) 
                    self.level = 3 if next_level == None else next_level
                    self.get_feasible_actions()
                    return False, -1000.0, True

            elif self.level == 3:
                # select amount for the select parameter of add solvent 
                if self._chosen_unit_name() == "add_solvent":
                    _, component, _, _ = self.current_state["pending_params"]["add_solvent"].values()
                    if component:
                        selected_amount = self.env_config.add_solvent_comp_map[component][action_index]
                        self.current_state["pending_params"]["add_solvent"]["index_for_amount"] = action_index
                        self.current_state["pending_params"]["add_solvent"]["amount_value"] = selected_amount
                        self.history.append(action_index)

                        self.level_list.append(self.level)
                        done, reward, worked = self._complete_action_place_and_simulate()
                        self.current_state["completed_design"] = done 
                        self.objective = reward
                        self.level = 0 if next_level == None else next_level
                        self.get_feasible_actions()
                
                elif self._chosen_unit_name() == "mixer":
                    index_value = self.current_state["second_open_stream_dest_node"] 
                    all_candidates = self._enumerate_open_streams_excluding(exclude = self.current_state["chosen_open_stream"])
                    src_node, _ = self.current_state["chosen_open_stream"]
                    true_cands = []
                    for i, out in all_candidates:
                        if i == index_value and i != src_node:
                            true_cands.append((i, out))

                    out_value = None
                    for i, name in true_cands:
                        if action_index == 1 and name == 'out1':
                            out_value = 'out1'
                        elif action_index == 0 and name == 'out0':
                            out_value = 'out0'

                    if out_value is None:
                        raise RuntimeError("Selected mixer outlet is not valid for the chosen destination node.")

                    self.current_state["pending_params"]["mixer"] = (index_value, out_value)
                    self.history.append(action_index) 
                    self.current_state["second_open_stream"] = (index_value, out_value)

                    done, reward, worked = self._complete_action_place_and_simulate()
                    self.current_state["completed_design"] = done 
                    self.objective = reward
                    
                    self.level_list.append(self.level)
                    self.level = 0 if next_level == None else next_level
                    self.get_feasible_actions()
                    self.current_state['open_streams'] = self._enumerate_open_streams()
 
                else:
                    if self._chosen_unit_name() not in ["add_solvent", "mixer"]:
                        raise RuntimeError("Only add_solvent or mixer related decisions allowed on Level 3")
                
                return done, reward, worked
            

        except Exception as e:
            # failed move (e.g., PEQ fail, convergence fail)
            print("take_action error:")
            print("  exception type:", type(e))
            print("  exception value:", e)
            traceback.print_exc()
            # reset to prevent level loops
            self._reset_action_state()
            self._rollback_action_history()
            return False, -1000.0, False

        return False, -1000.0, True

    # ------------- internals -------------#

    def _unit_available(self, unit_name: str, unit_idx: int) -> bool:

        if unit_name != 'recycle':
            if self.total_units_placed >= getattr(self, "max_total_units", 9999) :
                return False

        cap_map = {
            "distillation_column": self.max_distillation_columns,
            "decanter": self.max_decanters,
            "split": self.max_split,
            "mixer": self.max_mixer,
            "recycle": self.max_recycle,
            "add_solvent": self.max_solvent,
        }
        if unit_name in cap_map and self.counts[unit_name] >= cap_map[unit_name]:
            return False

        # add_solvent: only allowed components (global indices)
        '''if unit_name == "add_solvent":
            start = self.env_config.add_solvent_start_index
            if start is None or unit_idx < start:
                return False
            comp_global_idx = unit_idx - start
            allowed = self.sim.feed_stream_information.get("possible_ind_add_comp", [])
            if comp_global_idx not in allowed:
                return False'''

        # mixer needs >=2 open streams
        if unit_name == "mixer":
            if len(self._enumerate_open_streams()) < 2:
                return False

        # recycle needs:
        #  - >=2 open streams
        #  - a chosen source stream (comes from level 0 first)
        #  - at least one eligible destination unit with single input
        if unit_name == "recycle":
            if len(self._enumerate_open_streams()) < 2:
                return False
            if self.current_state["chosen_open_stream"] is None:
                return False
            if len(self._eligible_recycle_destinations()) == 0:
                return False

        return True

    def _chosen_unit_name(self) -> Optional[str]:
        chosen = self.current_state["chosen_unit"]
        if chosen is None:
            return None
        chosen_unit_index, _ = chosen
        if chosen_unit_index is None:
            return None
        return self.env_config.units_map_indices_type[chosen_unit_index]

    def _enumerate_open_streams(self) -> List[Tuple[int, str]]:
        return self.sim.get_open_streams()

    def _enumerate_open_streams_excluding(self, exclude: Optional[Tuple[int, str]]) -> List[Tuple[int, str]]:
        all_ops = self.sim.get_open_streams()
        if exclude is None:
            return all_ops
        return [(n, l) for (n, l) in all_ops if not (n == exclude[0] and l == exclude[1])]
    

    def generate_custom_flowsheet_id (self) -> str:
        
        """
        Generates a custom identifier for a newly generated flowsheez 

        """

        random_chars = ''.join(random.choices(string.ascii_uppercase + string.digits + string.ascii_lowercase, k=4))
        return f"{random_chars}"

    
    def _all_units_at_max_capacity(self) -> bool:
        cap_map = {
            "distillation_column": self.max_distillation_columns,
            "decanter": self.max_decanters,
            "split": self.max_split,
            "mixer": self.max_mixer,
            "recycle": self.max_recycle,
            "add_solvent": self.max_solvent,
        }

        for unit, max_cap in cap_map.items():
            # if a unit has no cap, it never blocks termination
            if max_cap is None:
                return False

            if self.counts.get(unit, 0) < max_cap:
                return False

        return True
    
    def _stream_is_recyclable(self, stream, purity_cutoff=0.90, eps=1e-12):
        node_id, lab = stream
        flow = np.asarray(self.sim.graph.nodes[node_id]["output_flows"][lab], dtype=float)
        total = flow.sum()

        if total <= eps:
            return False

        y = flow / total

        # if already pure enough, do not recycle it
        num_comps_in_feed = len(self.problem_instance["indices_components_in_feeds"])
        if np.max(y[:num_comps_in_feed]) >= purity_cutoff:
            return False
        
        return True and not self._node_already_has_recycle(stream)
    
    def _is_pure_solvent_stream(self, stream, eps=1e-12):
        node_id, lab = stream
        flow = np.asarray(self.sim.graph.nodes[node_id]["output_flows"][lab], dtype=float)
        total = flow.sum()
        if total <= eps:
            return False

        y = flow / total
        solvent_idx = 2
        return y[solvent_idx] >= 0.99
    
    def _stream_has_valid_recycle(self, stream) -> bool:
        dests = self._eligible_recycle_destinations(stream=stream)
        dests_valid = len(dests) > 0
        if_recyclable = self._stream_is_recyclable(stream)

        return if_recyclable and dests_valid
    
    def _node_already_has_recycle(self, node_id):
        for u, v, data in self.sim.graph.edges(data=True):
            if u == node_id and data.get("is_recycle", False):
                return True
        return False

    def _assert_stream_is_open(self, stream: Tuple[int, str]) -> None:
        opens = set(self._enumerate_open_streams())
        if stream not in opens:
            raise RuntimeError(f"Chosen stream {stream} is no longer open.")

    def _complete_action_place_and_simulate(self) -> Tuple[bool, float, bool]:
        
        
        """
        Place the chosen unit (or recycle), wire edges, run simulate(), compute NPV.
        Resets to level 0 when done (unless terminated).
        
        """
        chosen_unit_index, unit_name = self.current_state["chosen_unit"]
        if self.current_state["chosen_open_stream"] is None or chosen_unit_index is None or unit_name is None:
            raise RuntimeError("Action incomplete (missing stream or unit).")

        # Ensure the source stream is still open
        self._assert_stream_is_open(self.current_state["chosen_open_stream"])
        src_node, src_label = self.current_state["chosen_open_stream"]

        # Build params per unit
        params: Dict[str, Any] = {}
        created_node_id: Optional[int] = None
        snap = None
        edge_snap = None

        try:
            # Continuous param (if any)
            if unit_name == "distillation_column":
                _, cont_val = self.current_state["pending_params"]["distillation_column"]
                params["df"] = cont_val
            elif unit_name == "split":
                _, cont_val = self.current_state["pending_params"]["split"]
                params["split_ratio"] = cont_val

            elif unit_name == "add_solvent":
                index, component, index_for_amount, amount_value = self.current_state["pending_params"][
                    "add_solvent"].values()
                params = {
                    "index_new_component": index,
                    "solvent_amount": float(amount_value),
                    "component_name": component
                }

            snap = self.sim.snapshot(include_phase=(unit_name == "add_solvent"))

            # Actually place
            if unit_name == "mixer":
                index, second_o_str_name = self.current_state["pending_params"]['mixer']
                if second_o_str_name is None:
                    raise RuntimeError("Mixer requires a second open stream.")
                n2, l2 = index, second_o_str_name
                created_node_id = self.sim.add_unit(
                    [(src_node, src_label), (n2, l2)],
                    "mixer",
                    params={},
                    num_outputs=1
                )

            elif unit_name == "recycle":
                recycle_dest = self.current_state["pending_params"]["recycle"]
                edge_snap = self.sim.snapshot_edges()
                feed_nodes = set(getattr(self.sim, "feed_nodes", []))
                if recycle_dest in feed_nodes:
                    raise ValueError(f"Cannot recycle into feed node {recycle_dest}")
                if recycle_dest is None:
                    raise RuntimeError("Recycle requires a destination unit to be chosen.")
                if recycle_dest == src_node:
                    raise ValueError("Cannot recycle a stream back into its own producing unit.")
                
                # Add recycle edge (transactional)
                self.sim.add_recycle(src_node, src_label, recycle_dest)
                created_node_id = None  # no new node

            elif unit_name == "add_solvent":
                created_node_id = self.sim.add_unit(
                    [(src_node, src_label)],
                    "add_solvent",
                    params=params,
                    num_outputs=1
                )

            elif unit_name == "distillation_column":
                created_node_id = self.sim.add_unit(
                    [(src_node, src_label)],
                    "distillation_column",
                    params=params,
                    num_outputs=2
                )

            elif unit_name == "decanter":
                created_node_id = self.sim.add_unit(
                    [(src_node, src_label)],
                    "decanter",
                    params={},
                    num_outputs=2
                )

            elif unit_name == "split":
                created_node_id = self.sim.add_unit(
                    [(src_node, src_label)],
                    "split",
                    params=params,
                    num_outputs=2
                )

            else:
                raise ValueError(f"Unknown unit type: {unit_name}")

            # simulate + NPV
            self.sim.simulate()
            self.failed_simulator_call = 0

        except Exception as e:
            print("Placement/simulation failed:", e)
            if unit_name == "recycle" and edge_snap is not None:
                try:
                    self.sim.restore_edges(edge_snap)
                except Exception:
                    pass
            elif created_node_id is not None:
                try:
                    self.sim.remove_node_and_restore_upstream_open(created_node_id)
                except Exception:
                    pass
            if snap is not None:
                try:
                    self.sim.restore(snap)
                except Exception:
                    pass
            self._reset_action_state()
            self._rollback_action_history()
            self.failed_simulator_call += 1
            return False, 0.0, False

        # update counts if worked and a *new* node was placed (recycle places no node)
        if unit_name == "recycle":
            self.counts["recycle"] += 1
        else:
            self.counts[unit_name] += 1
            self.total_units_placed += 1

        # reward = normalized NPV
        self.current_state['npv_raw'] = self.sim.current_net_present_value
        self.current_state['npv_norm'] = self.sim.current_net_present_value_normed 
        reward = self.sim.current_net_present_value_normed #or 0.0
        
        return False, reward, True

    def _rollback_action_history(self):
        start = getattr(self, "_action_seq_start", 0)
        self.history = self.history[:start]
        self.level_list = self.level_list[:start]

    def _reset_action_state(self):
        """
        Reset all current state variables in case the simulation fails
        """
        self.level = 0
        self.current_state["current_level"] = 0
        self.current_state["open_streams"] = self._enumerate_open_streams()
        self.current_state["chosen_unit"] = None
        self.current_state["chosen_open_stream"] = None
        self.current_state["pending_params"] = {
            "distillation_column": None,
            "split": None,
            "recycle": None,
            "add_solvent": {
                "index_for_comp": None,
                "name_comp": None,
                "index_for_amount": None,
                "amount_value": None,
            },
            "mixer": None,
        }
        self.current_state["npv_raw"] = None
        self.current_state["npv_norm"] = None
        self.current_state["completed_design"] = False
        self.current_state["second_open_stream_dest_node"] = None
        self.current_state["second_open_stream"] = None
        self.current_state["recycle_dest_unit"] = None
        self.current_state["current_action_mask"] = None
        self.get_feasible_actions()


    def _eligible_recycle_destinations(self, stream = None) -> List[int]:
        """
        Return the exact list of destination unit node_ids that are legal for the
        currently chosen open stream:
          - must have available input capacity per simulator
          - must not be a feed
          - must not be the origin (producer) unit of the chosen stream
          - if env_config.allow_forward_recycles is False, destination must be
            upstream/older than the source node, i.e. dest_node_id < origin_node_id
        """
        if self.current_state["chosen_open_stream"] is None:
            return []

        origin_node_id = self.current_state["chosen_open_stream"][0] if stream == None else stream[0] 
        dests = list(self.sim.get_units_with_available_input(exclude=origin_node_id, max_inputs=2))
        #print(f"origin node: {origin_node_id}")

        # Cannot recycle into feeds.
        feed_nodes = set(getattr(self.sim, "feed_nodes", []))
        dests = [nid for nid in dests if nid not in feed_nodes]

        # Optional true-recycle restriction: block forward pseudo-recycles such as
        # split out1 -> later unit that was already fed by split out0.
        if not bool(getattr(self.env_config, "allow_forward_recycles", True)):
            dests = [nid for nid in dests if nid < origin_node_id] 
        return dests


    def compute_recycler_masks(flowsheets) -> torch.Tensor:

        """
        Make attention masks for recycler expert unit. 
        0 = not allowed, 1 = allowed 

        mask: torch.tensor of shape (B, max_nodes, max_nodes)

        """

        max_nodes = max(fs.sim.graph.number_of_nodes() for fs in flowsheets)
        attention_mask = torch.zeros(len(flowsheets), max_nodes, max_nodes, device=flowsheets[0].gen_config.training_device)

        for fs_num, fs in enumerate(flowsheets):
            node_ids = list(fs.sim.graph.nodes)
            id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
            N = len(node_ids)

            # chosen source node id (graph-space)
            if fs.current_state["chosen_open_stream"] == None: 
                continue 
            else:
                source_node = fs.current_state["chosen_open_stream"][0]
                if source_node not in id_to_idx:
                    continue
                src_idx = id_to_idx[source_node]

            # candidate destination node ids (graph-space)
            cand_node_ids = fs._eligible_recycle_destinations()
            if len(cand_node_ids) == 0:
                continue

            for dst_node_id in cand_node_ids:
                if dst_node_id == source_node:
                    continue
                if dst_node_id not in id_to_idx:
                    continue

                dst_idx = id_to_idx[dst_node_id] # do this only when source node is not equal to destination node 
                attention_mask[fs_num, src_idx, dst_idx] = 1 # avoid self connections and only allow valid dest candidates corresponding to the row of chosen unit''' 
        
        return attention_mask

    
    def compute_mixer_masks(flowsheets) -> torch.Tensor:

        """
        Make attention masks for mixer expert unit. 
        0 = not allowed, 1 = allowed 
        mask: torch.tensor of shape (B, max_nodes, max_nodes)

        """
        max_nodes = max(fs.sim.graph.number_of_nodes() for fs in flowsheets)
        attention_mask = torch.zeros(len(flowsheets), max_nodes, max_nodes, device=flowsheets[0].gen_config.training_device)

        for fs_num, fs in enumerate(flowsheets):
            open_streams = fs._enumerate_open_streams()
            open_nodes_id = [node_id for node_id, _ in open_streams]

            node_ids = list(fs.sim.graph.nodes)
            id_to_idx = {nid: i for i, nid in enumerate(node_ids)}

            src_node = fs.current_state["chosen_open_stream"] 
            if src_node is None:
                continue

            src_node_id, _ = src_node  
            src_idx = id_to_idx[src_node_id]

            for dst_node_id in open_nodes_id:
                dst_idx = id_to_idx[dst_node_id]
                if dst_node_id != src_node_id:
                    attention_mask[fs_num, src_idx, dst_idx] = 1 # avoid self connections and only allow valid dest candidates corresponding to the row of chosen stream        
        
        return attention_mask
    
    def masked_log_probs_for_current_action_level(self, logits: np.ndarray) -> np.ndarray:
        
        """
        Apply current_action_mask to logits and return normalized log-probs.
        """

        mask = self.current_action_mask.astype(bool)
        logits = logits.copy()
        logits[~mask] = -np.inf
        with np.errstate(divide="ignore", invalid="ignore"):
            log_probs = np.log(softmax(logits))

        return log_probs

    def forced_termination(self):
        unit_budget_full = (self.total_units_placed >= getattr(self, "max_total_units", 9999))
        recycle_budget_full = (self.counts["recycle"] >= getattr(self, "max_recycle", 9999)) 
        no_capacity_left = self._all_units_at_max_capacity() and recycle_budget_full
        too_many_failures = (self.failed_simulator_call >= self.env_config.max_simulator_tries)

        return self.level == 0 and (too_many_failures or no_capacity_left or (unit_budget_full and recycle_budget_full))
    
    def optional_termination(self) -> bool:
        min_units = getattr(self, "min_total_units", 0)
        return self.level == 0 and self.total_units_placed >= min_units

    
    def is_terminable(self):
        return self.forced_termination() or self.optional_termination()
    

    @staticmethod
    def get_open_stream_mask_padded(flowsheets: List['FlowsheetDesign']):
        
        batch_open_streams = []
        max_nodes = max(fs.sim.graph.number_of_nodes() for fs in flowsheets)
        batch_node_outlet_masks = []

        for fs in flowsheets:
            open_streams = fs.current_state["open_streams"]
            batch_open_streams.append(open_streams)
        
        
        for streams, fs in zip(batch_open_streams, flowsheets):
            
            #canonical node ordering
            node_ids = list(fs.sim.graph.nodes)
            id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
            outlet_mask = torch.zeros(max_nodes, 2, dtype=torch.bool)

            for node_id, outlet in streams:
                outlet_mask[id_to_idx[node_id], fs.env_config.outlet_to_idx[outlet]] = True
            
            batch_node_outlet_masks.append(outlet_mask)

        return torch.stack(batch_node_outlet_masks, dim=0).to(flowsheets[0].gen_config.training_device)


    @staticmethod
    def get_embeddings_from_experts(flowsheets: List['FlowsheetDesign']):

        """
        Get embeddings in latent dim for each node and edge using predefined units and edge experts 
        Already padded and stacked 

        """

        max_nodes = max(fs.sim.graph.number_of_nodes() for fs in flowsheets)
        valid_node_mask = torch.zeros(len(flowsheets), max_nodes + 1, dtype=torch.bool, device=flowsheets[0].gen_config.training_device)
        node_embeds = torch.zeros(len(flowsheets), max_nodes, flowsheets[0].gen_config.latent_dim, device=flowsheets[0].gen_config.training_device) #this is also considering virtual node already 
        open_stream_embeds = torch.zeros(len(flowsheets), max_nodes, flowsheets[0].env_config.max_outlets, flowsheets[0].gen_config.latent_dim, device=flowsheets[0].gen_config.training_device)
        flow_embeds = torch.zeros(len(flowsheets), max_nodes, flowsheets[0].env_config.max_outlets, flowsheets[0].gen_config.latent_dim, device=flowsheets[0].gen_config.training_device)
        edge_embeds = torch.zeros(len(flowsheets), max_nodes + 1 , max_nodes + 1, flowsheets[0].gen_config.latent_dim, device=flowsheets[0].gen_config.training_device) # +1 for handling edges for virtual node 

        for fs_num, fs in enumerate (flowsheets): 
            node_ids = list(fs.sim.graph.nodes)
            id_to_idx = {nid: i for i, nid in enumerate(node_ids)}

            for nid in node_ids:
                node_data = fs.sim.graph.nodes[nid]
                if node_data["unit_type"] == "feed":
                    interaction_params = node_data['params']['system_gammas_inf']
                    system_pure_crit = node_data['params']['system_pure_crit']
                    compon_emb, interaction_emb = fs.unit_experts["component_expert"](system_pure_crit, interaction_params, node_data)
                    flow_emb, open_emb = fs.open_stream_expert(node_data, compon_emb, interaction_emb)

                    node_embeds[fs_num, id_to_idx[nid]] =  flow_emb[0]
                    open_stream_embeds[fs_num, id_to_idx[nid]] = open_emb
                    flow_embeds[fs_num, id_to_idx[nid]] = flow_emb

                else:
                    if node_data["unit_type"] in ["mixer", "decanter"]:
                        emb = fs.unit_experts[node_data["unit_type"]].embed(batch_size=1)
                        flow_emb, open_emb = fs.open_stream_expert(node_data, compon_emb, interaction_emb)

                    else:
                        emb = fs.unit_experts[node_data["unit_type"]].embed(node_data)
                        flow_emb, open_emb = fs.open_stream_expert(node_data, compon_emb, interaction_emb)
                    
                    node_embeds[fs_num, id_to_idx[nid]] =  emb
                    open_stream_embeds[fs_num, id_to_idx[nid]] = open_emb
                    flow_embeds[fs_num, id_to_idx[nid]] = flow_emb
                    
                valid_node_mask[fs_num, 0] = True # virtual node is always valid
                valid_node_mask[fs_num, id_to_idx[nid] + 1] = True #all other nodes get shifted by +1 
            
            # now overwrite where real edges exist
            for u, v, key, edge_data in fs.sim.graph.edges(keys= True, data=True):
                ui = id_to_idx[u]
                vi = id_to_idx[v]               
                if 'out0' in edge_data["output_label"]:
                    edge_out0_src = fs.edge_expert(edge=edge_data, edge_exists=True, is_recycle = edge_data.get("is_recycle"), feed_emb = flow_embeds[fs_num][ui][0])
                    edge_embed = edge_out0_src #+ edge_out0_dest
                else:
                    edge_out1_src = fs.edge_expert(edge=edge_data, edge_exists=True, is_recycle = edge_data.get("is_recycle"), feed_emb = flow_embeds[fs_num][ui][1])
                    edge_embed = edge_out1_src #+ edge_out1_dest

                edge_embeds[fs_num, ui, vi] += edge_embed

        return dict(batch_latent_nodes_embeds = node_embeds, 
                batch_latent_edges_embeds = edge_embeds, 
                batch_latent_open_streams_embeds = open_stream_embeds,
                valid_node_mask = valid_node_mask,
                open_stream_mask = FlowsheetDesign.get_open_stream_mask_padded(flowsheets))
    

    def literature_flowsheet_match(self, desired_nodes, desired_edges):
    
        # same number of nodes
        if len(self.sim.graph.nodes) != len(desired_nodes):
            return False

        # check node types
        for nid, utype in desired_nodes.items():
            if nid not in self.sim.graph.nodes:
                return False
            if self.sim.graph.nodes[nid]["unit_type"] != utype:
                return False
            
        self.literature_bonus = max(self.literature_bonus, 0.10)

        # extract edges
        graph_edges = set()
        for u, v, data in self.sim.graph.edges(data=True):
            graph_edges.add(
                (u, v, data.get("is_recycle", False))
            )

        edges_match = graph_edges == desired_edges

        return edges_match
    

    def literature_flowsheet_similarity_check(self):
        self.literature_bonus = 0.0
        sys_name = self.problem_instance["system_name"]
        motifs = self.env_config.literature_motifs.get(sys_name, [])

        for motif in motifs:
            desired_nodes = motif["desired_nodes"]
            desired_edges = motif["desired_edges"]

            if self.literature_flowsheet_match(desired_nodes, desired_edges):
                self.literature_bonus = 0.20
                break
    
    # ---- Implementation of abstract methods from `BaseTrajectory`
    def transition_fn(self, action: int) -> Tuple['BaseTrajectory', bool]:

        expert_list = self.remove_units_before_deepcopy()
        copied_fs= copy.deepcopy(self)

        self.add_experts_back(expert_list, copied_fs)
        copied_fs.take_action(action, None)
        return copied_fs, copied_fs.current_state["completed_design"]
    
    def remove_units_before_deepcopy(self):
        
        # Temporarily remove GPU modules before deepcopy
        unit_experts = self.unit_experts
        edge_expert = self.edge_expert
        open_stream_expert = self.open_stream_expert

        self.unit_experts = None
        self.edge_expert = None
        self.open_stream_expert = None

        expert_list = [unit_experts, edge_expert, open_stream_expert]
        return expert_list
    
    def add_experts_back(self, expert_list, copied_fs):
        self.unit_experts = expert_list[0]
        self.edge_expert = expert_list[1]
        self.open_stream_expert = expert_list[2]

        # Attach same shared experts to copy, do NOT deepcopy them
        copied_fs.unit_experts = expert_list[0]
        copied_fs.edge_expert = expert_list[1]
        copied_fs.open_stream_expert = expert_list[2]

    
    def to_max_evaluation_fn(self) -> float:
        if self.objective is None:
            raise ValueError("Objective is `None`. Check if Flowsheet Simulator really works")
        return self.objective

    
    @staticmethod
    def log_probability_fn(trajectories: List['FlowsheetDesign'], network: nn.Module, device: torch.device, config= None) -> List[np.array]:
        
        """
        Given a list of trajectories and a policy network,
        returns a list of numpy arrays, each having length num_actions, where each numpy array is a log-probability
        distribution over the next action level.

        Parameters:
            trajectories [List[BaseTrajectory]]
            network [torch.nn.Module]: Policy network
        Returns:
            List of numpy arrays, where i-th entry corresponds to the log-probabilities for i-th trajectory.

        """
        log_probs_to_return: List[np.array] = []
        device = torch.device("cpu") if device is None else device
        network.eval()
        with torch.no_grad():
            with torch.amp.autocast(enabled=False, device_type=config.training_device):
                batch = FlowsheetDesign.list_to_batch(flowsheets=trajectories, device=network.device)
                lvl_0_logits, unit_predictions = network(batch)
                padded_open_stream_masks = batch['open_stream_mask']
                valid_nodes = batch['valid_nodes']
                for i, fs in enumerate(trajectories):
                    # get logits for this sequence and corresponding level
                    if fs.level == 0:
                        terminate_logits_per_batch = lvl_0_logits['terminate_logits'][i, :]
                        open_stream_logits_per_batch= lvl_0_logits['open_stream_logits'][i, :, :]
                        open_stream_valid_logits = open_stream_logits_per_batch[padded_open_stream_masks[i]] #isolate non padded nodes 
                        logits = torch.cat([terminate_logits_per_batch, open_stream_valid_logits],dim=0)
                        logits = np.array(logits.float().cpu())
                    
                    if fs.level == 1:
                        node_id, _ = fs.current_state["chosen_open_stream"]
                        node_ids = list(fs.sim.graph.nodes)
                        id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
                        node_id = id_to_idx[node_id]

                        # collect logits for units corresponding to ONLY this open stream 
                        logits = []
                        for _, unit_name in enumerate(fs.env_config.units_map_indices_type):
                            logit = unit_predictions[unit_name]["picked_logit"][i, node_id, 0] # scalar logit per unit for ith batch, at node node_id and removing any singleton dimension
                            logits.append(logit.float().detach().cpu())
                        logits = np.array(logits) # (length of all possible units, _)
                    
                    if fs.level == 2: # extract parameter predictions
                        node_ids = list(fs.sim.graph.nodes)
                        id_to_idx = {nid: i for i, nid in enumerate(node_ids)} 
                        _, unit_name = fs.current_state["chosen_unit"]
                        node_id, _ = fs.current_state["chosen_open_stream"]
                        node_id = id_to_idx[node_id]

                        logits = []
                        if unit_name == "distillation_column":
                            logits = unit_predictions[unit_name]["distillate_fraction_categorical"][i, node_id, :]
                        if unit_name == "mixer":
                            target_scores = unit_predictions[unit_name]["target_scores"][i, node_id, :]
                            logits = target_scores[valid_nodes[i][1:]]
                        if unit_name == "recycle":
                            target_scores = unit_predictions[unit_name]["target_scores"][i, node_id, :]
                            logits = target_scores[valid_nodes[i][1:]]
                        if unit_name == "split":
                            logits = unit_predictions[unit_name]["split_ratio_categorical"][i, node_id, :]
                        if unit_name == "add_solvent":
                            logits = unit_predictions[unit_name]["component_logit"][i, node_id, :] 
                        
                        logits = np.array(logits.float().detach().cpu())

                    if fs.level == 3:
                        logits = []
                        node_ids = list(fs.sim.graph.nodes)
                        id_to_idx = {nid: i for i, nid in enumerate(node_ids)} 
                        _, unit_name = fs.current_state["chosen_unit"]
                        node_id, _ = fs.current_state["chosen_open_stream"]
                        node_id = id_to_idx[node_id]

                        if unit_name == "add_solvent":
                            index_comp, comp_name, _, _ = fs.current_state["pending_params"]["add_solvent"].values()
                            logits = unit_predictions[unit_name]["component_amount"][i, node_id, index_comp, :]
                        elif unit_name == "mixer":
                            dest_node = fs.current_state["second_open_stream_dest_node"]
                            outlet_logits = unit_predictions[unit_name]["destinate_node_outlets"][i, id_to_idx[dest_node], :]
                            logits = outlet_logits

                        logits = np.array(logits.float().detach().cpu())
                    
                    log_probs_to_return.append(fs.masked_log_probs_for_current_action_level(logits))
        return log_probs_to_return
    
    @staticmethod
    def batch_to_device(batch: dict, device: torch.device):
        """
        Takes batch as returned from `list_to_batch` and moves it onto the given device.
        """
        return {k: v.to(device) for k, v in batch.items()}
    
    @staticmethod
    def design_flowsheets(random_instance: Dict[str, Any], gen_config, env_config) -> List['FlowsheetDesign']:

        """
        Returns list of flowsheet designs based on a starting problem instance 

        """
        instance_list = []
        flowsheet_traj = FlowsheetDesign(random_instance, gen_config, env_config)
        #instance_list.append(flowsheet_traj)
        return flowsheet_traj
    
    @staticmethod
    def list_to_batch(flowsheets: List['FlowsheetDesign'], include_feasibility_masks: bool = False, device: torch.device = None) -> dict:
        
        """
        Given a list of flowsheet graphs, prepares a batch that can be passed through the network.

        The batch is given as a dictionary with the following keys and values:
        * "level_idx": tensor containing information about the current level that the agent is on 
        * "embedding_dict": a dict containing stacked tensors of node and edge embeddings, as processed from unit/open_stream experts and masks for open_stream and valid nodes 
        * "recycler_masks": stacked tensors of padded recycler_masks, which will be passed to recycle unit expert
        * "mixer_masks": stacked tensors of padded mixer_masks, which will be passed to mixer unit expert

        if `include_feasibility_masks` is set to True, we also return feaisbility masks 
        """

        assert len(flowsheets) > 0, "Empty batch of flowsheets"

        # Calculate latent dimension from experts  
        embeddings_dict = FlowsheetDesign.get_embeddings_from_experts(flowsheets = flowsheets)

        recycler_masks = FlowsheetDesign.compute_recycler_masks(flowsheets= flowsheets)
        mixer_masks = FlowsheetDesign.compute_mixer_masks(flowsheets= flowsheets)

        # levels info
        batch_levels_idx = [fs.level for fs in flowsheets]

        return_dict = dict(
            batch_latent_nodes_embeds = embeddings_dict['batch_latent_nodes_embeds'], 
            batch_latent_edges_embeds = embeddings_dict['batch_latent_edges_embeds'], 
            batch_latent_open_streams_embeds =  embeddings_dict['batch_latent_open_streams_embeds'], 
            valid_nodes = embeddings_dict['valid_node_mask'],
            open_stream_mask = embeddings_dict['open_stream_mask'],
            recycler_masks = recycler_masks,        
            mixer_masks = mixer_masks, 
            levels = torch.tensor(batch_levels_idx, dtype=torch.long, device=device), 
        )

        if include_feasibility_masks:

            # Build per-level feasibility masks, padded across the batch to each level's max action count.
            feasibility_mask_per_level = []

            num_actions_per_level_and_flowsheet = [
                [len(fs.current_state['open_streams']) + 1 for fs in flowsheets],  # lvl 0 
                [fs.env_config.num_units for fs in flowsheets],  # lvl 1
                [100 if fs.current_state['chosen_unit'] is not None and fs.current_state['chosen_unit'][1] in ('distillation_column', 'split')
                else len(fs.sim.graph.nodes) if fs.current_state['chosen_unit'] != None and fs.current_state['chosen_unit'][1] in ('recycle', 'mixer')
                else 5 if fs.current_state['chosen_unit'] is not None and fs.current_state['chosen_unit'][1] == 'add_solvent'
                else 100 for fs in flowsheets], #lvl 2 
                [100 if fs.current_state['chosen_unit'] is not None and fs.current_state['chosen_unit'][1] == 'add_solvent'
                else 2 if fs.current_state['chosen_unit'] is not None and fs.current_state['chosen_unit'][1] == 'mixer' 
                else 100 for fs in flowsheets],  # lvl 3
                
            ]

            for lvl, num_actions_per_fs in enumerate(num_actions_per_level_and_flowsheet):
                max_num_actions = max(num_actions_per_fs)
                max_num_actions = 100 if lvl == 3 and max_num_actions == 0 else max_num_actions
                feasibility_mask_per_level.append(
                torch.from_numpy(
                    np.stack([
                        np.pad(
                            ~fs.current_action_mask.astype(bool),
                            (0, max_num_actions - fs.current_action_mask.shape[0]),
                            mode='constant', constant_values=True
                        ) if fs.level  == lvl 
                        else np.ones(max_num_actions, dtype=bool)
                        for i, fs in enumerate(flowsheets)
                    ])
                ).bool().to(device)
            )
        
            # Add to return_dict
            return_dict["feasibility_mask_level_zero"] = feasibility_mask_per_level[0] 
            return_dict["feasibility_mask_level_one"] = feasibility_mask_per_level[1]  
            return_dict["feasibility_mask_level_two"] = feasibility_mask_per_level[2]  
            return_dict["feasibility_mask_level_three"] = feasibility_mask_per_level[3]  

        return return_dict
