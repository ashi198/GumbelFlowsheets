import os, mlflow 
import numpy as np 
import torch


def set_mlflow_connection():
    #os.environ["AWS_ACCESS_KEY_ID"] = "minio_id"
    #os.environ["AWS_SECRET_ACCESS_KEY"] = "XaFC8sHiRHr5uQTdfVQQ"
    #os.environ["MLFLOW_S3_IGNORE_TLS"] = "true"
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://0.0.0.0:5000" 
    #os.environ['MLFLOW_TRACKING_USERNAME'] = "hilbert" #"fhaselbeck"
    #os.environ['MLFLOW_TRACKING_PASSWORD'] = "schneckenfunktion"
    #os.environ['GIT_PYTHON_REFRESH'] ="quiet"
    remote_server_uri = "http://0.0.0.0:5000"  # "http://10.154.6.32:5100"  # set to MLFlow server URI (host ip and PORT in .env)
    mlflow.set_tracking_uri(remote_server_uri)


def build_logit_tensors_per_level(dataset, input_data, terminate_or_open_streams_logits, unit_predictions, indices_for_tracking):

    '''

    Takes logits across the entire space and build its per level. Returns a padded tensor per level, 
    with size (num_flowsheet_dataset, max_possible_action_per_lvl)

    Args:
        dataset: derived from RandomDataset function, which contains information about all flowsheets and action sequences derived from .pickle file. 
        input_data: dict containing information about masking for valid nodes and open streams. 
        terminate_or_open_streams_logits: dict of logits across lvl0 derived from the network. 
        unit_predictions; dict of logits across lvls 1, 2, and 3 derived from the network.
        indices_for_tracking: list of indices for flowsheets that are actually loaded in the dataloader. 

    '''

    indices_for_tracking = [t.squeeze().item() for t in indices_for_tracking]
    actual_flowsheets_dataset = [dataset._flat_sequences[i] for i in indices_for_tracking]

    num_actions_per_level_and_flowsheet = [
                [len(fs.current_state['open_streams']) + 1 for fs in actual_flowsheets_dataset],  # lvl 0 
                [fs.env_config.num_units for fs in actual_flowsheets_dataset],  # lvl 1
                [100 if fs.current_state['chosen_unit'] is not None and fs.current_state['chosen_unit'][1] in ('distillation_column', 'split')
                else len(fs.sim.graph.nodes) if fs.current_state['chosen_unit'] != None and fs.current_state['chosen_unit'][1] in ('recycle', 'mixer')
                else 5 if fs.current_state['chosen_unit'] is not None and fs.current_state['chosen_unit'][1] == 'add_solvent'
                else 100 for fs in actual_flowsheets_dataset], #lvl 2 
                [100 if fs.current_state['chosen_unit'] is not None and fs.current_state['chosen_unit'][1] == 'add_solvent'
                else 2 if fs.current_state['chosen_unit'] is not None and fs.current_state['chosen_unit'][1] == 'mixer' 
                else 100 for fs in actual_flowsheets_dataset],  # lvl 3
                
            ]
    
    all_lvl_zero_logits = torch.zeros(len(indices_for_tracking), max(num_actions_per_level_and_flowsheet[0])).to(actual_flowsheets_dataset[0].gen_config.training_device)
    all_lvl_one_logits = torch.zeros(len(indices_for_tracking), max(num_actions_per_level_and_flowsheet[1])).to(actual_flowsheets_dataset[0].gen_config.training_device)
    all_lvl_two_logits = torch.zeros(len(indices_for_tracking), max(num_actions_per_level_and_flowsheet[2])).to(actual_flowsheets_dataset[0].gen_config.training_device)
    all_lvl_three_logits = torch.zeros(len(indices_for_tracking), max(num_actions_per_level_and_flowsheet[3])).to(actual_flowsheets_dataset[0].gen_config.training_device)

    for i, fs in enumerate(actual_flowsheets_dataset):
        
        if fs.level == 0:
            terminate_logits_fs = terminate_or_open_streams_logits['terminate_logits'][i, :]
            open_stream_logits_fs= terminate_or_open_streams_logits['open_stream_logits'][i, :, :]
            open_stream_valid_logits = open_stream_logits_fs[input_data['open_stream_mask'][i]] #isolate non padded nodes 
            lvl_zero_logits = torch.cat([terminate_logits_fs, open_stream_valid_logits],dim=0)
            all_lvl_zero_logits[i, :lvl_zero_logits.shape[0]] = lvl_zero_logits

        if fs.level == 1:
            node_id, _ = fs.current_state["chosen_open_stream"]
            node_ids = list(fs.sim.graph.nodes)
            id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
            node_id = id_to_idx[node_id]

            # collect logits for units corresponding to ONLY this open stream 
            lvl_one_logits = []
            for _, unit_name in enumerate(fs.env_config.units_map_indices_type):
                logit = unit_predictions[unit_name]["picked_logit"][i, node_id, 0] 
                lvl_one_logits.append(logit)

            lvl_one_logits = torch.stack(lvl_one_logits, dim=0)
            all_lvl_one_logits[i, :lvl_one_logits.shape[0]] = lvl_one_logits

        if fs.level == 2:
            node_ids = list(fs.sim.graph.nodes)
            id_to_idx = {nid: i for i, nid in enumerate(node_ids)} 
            _, unit_name = fs.current_state["chosen_unit"]
            node_id, _ = fs.current_state["chosen_open_stream"]
            node_id = id_to_idx[node_id]

            if unit_name == "distillation_column":
                lvl_two_logits = unit_predictions[unit_name]["distillate_fraction_categorical"][i, node_id, :]
            if unit_name == "mixer":
                target_scores = unit_predictions[unit_name]["target_scores"][i, node_id, :]
                lvl_two_logits = target_scores[input_data['valid_nodes'][i][1:]]
            if unit_name == "recycle":
                target_scores = unit_predictions[unit_name]["target_scores"][i, node_id, :]
                lvl_two_logits = target_scores[input_data['valid_nodes'][i][1:]]
            if unit_name == "split":
                lvl_two_logits = unit_predictions[unit_name]["split_ratio_categorical"][i, node_id, :]
            if unit_name == "add_solvent":
                lvl_two_logits = unit_predictions[unit_name]["component_logit"][i, node_id, :] 

            all_lvl_two_logits[i, :lvl_two_logits.shape[0]] = lvl_two_logits

        if fs.level == 3:
            lvl_three_logits = []
            node_ids = list(fs.sim.graph.nodes)
            id_to_idx = {nid: i for i, nid in enumerate(node_ids)} 
            _, unit_name = fs.current_state["chosen_unit"]
            node_id, _ = fs.current_state["chosen_open_stream"]
            node_id = id_to_idx[node_id]

            if unit_name == "add_solvent":
                index_comp, _, _, _ = fs.current_state["pending_params"]["add_solvent"].values()
                lvl_three_logits = unit_predictions[unit_name]["component_amount"][i, node_id, index_comp, :]
            elif unit_name == "mixer":
                dest_node = fs.current_state["second_open_stream_dest_node"]
                outlet_logits = unit_predictions[unit_name]["destinate_node_outlets"][i, id_to_idx[dest_node], :]
                lvl_three_logits = outlet_logits

            all_lvl_three_logits[i, :lvl_three_logits.shape[0]] = lvl_three_logits

    return all_lvl_zero_logits, all_lvl_one_logits, all_lvl_two_logits, all_lvl_three_logits
    

