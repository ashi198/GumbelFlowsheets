import argparse, copy, os, time, ray, torch, mlflow, datetime

from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.amp import autocast

from logger import Logger
from flowsheet_dataset import RandomDataset

import numpy as np
from config import GeneralConfig, EnvConfig

from core.gumbeldore_dataset import GumbeldoreDataset
from model.policy_arch import FlowsheetNetwork, dict_to_cpu
from utils import set_mlflow_connection, build_logit_tensors_per_level

os.environ["RAY_DEDUP_LOGS"]="0"
os.environ["RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"]="1"


def save_checkpoint(checkpoint: dict, filename: str, gen_config):
    os.makedirs(gen_config.results_path, exist_ok=True)
    path = os.path.join(gen_config.results_path, filename)
    torch.save(checkpoint, path)


def train_for_one_epoch(epoch: int, gen_config, env_config, network: FlowsheetNetwork, network_weights: dict,
                        optimizer: torch.optim.Optimizer, best_objective: float, system_index: int, destination_path: str):

    gumbeldore_dataset = GumbeldoreDataset(gen_config, env_config)
    metrics = gumbeldore_dataset.generate_dataset(
        network_weights,
        best_objective=best_objective,
        memory_aggressive=False, 
        system_index = system_index, 
        destination_path= destination_path
    )

    print("Generated Flowsheets")
    print(f"Mean obj. over fresh best flowsheets: {metrics['mean_best_gen_obj']:.3f}")
    print(f"Best / worst obj. over fresh best flowsheets: {metrics['best_gen_obj']:.3f}, {metrics['worst_gen_obj']:.3f}")
    print(f"Mean obj. over all time top 20 flowsheets: {metrics['mean_top_20_obj']:.3f}")
    print(f"All time best flowsheet: {list(metrics['top_20_flowsheets'][0].values())[0]:.3f}")

    torch.cuda.empty_cache()
    time.sleep(1)

    print("---- Loading dataset")
    dataset = RandomDataset(gen_config=gen_config, env_config=env_config, path_to_pickle=gen_config.gumbeldore_config["destination_path"], batch_size=gen_config.batch_size_training,
                                    custom_num_batches=gen_config.num_batches_per_epoch, no_random= True)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0,
                            pin_memory=False, persistent_workers=False)

    # Train for one epoch
    network.train()  

    accumulated_loss_lvl_zero = 0
    accumulated_loss_lvl_one = 0
    accumulated_loss_lvl_two = 0
    accumulated_loss_lvl_three = 0
    accumulated_loss = 0

    num_batches = len(dataloader)
    progress_bar = tqdm(range(num_batches))
    data_iter = iter(dataloader)
    scaler = torch.amp.GradScaler()

    for _ in progress_bar:
        data = next(data_iter)
        indices_for_tracking = data.pop('indices_for_tracking')
        input_data = {k: v[0].to(network.device) for k, v in data["input"].items()}
        
        # targets for the logit levels
        target_zero = data["target_zero"][0].to(network.device)
        target_one = data["target_one"][0].to(network.device)
        target_two = data["target_two"][0].to(network.device)
        target_three = data["target_three"][0].to(network.device)

        with autocast(enabled=False, device_type=gen_config.training_device): 

            terminate_or_open_streams_logits, unit_predictions = network(input_data)
            
            #build logits based on selected units and streams 
            lvl_zero_logits, lvl_one_logits, lvl_two_logits, lvl_three_logits = build_logit_tensors_per_level(dataset= dataset, 
                                                                                input_data=input_data, terminate_or_open_streams_logits=terminate_or_open_streams_logits,
                                                                                unit_predictions=unit_predictions, indices_for_tracking = indices_for_tracking)
            
            lvl_zero_logits[input_data["feasibility_mask_level_zero"]] = float("-inf")
            lvl_one_logits[input_data["feasibility_mask_level_one"]] = float("-inf")
            lvl_two_logits[input_data["feasibility_mask_level_two"]] = float("-inf")
            lvl_three_logits[input_data["feasibility_mask_level_three"]] = float("-inf")

            criterion = CrossEntropyLoss(reduction="mean", ignore_index=-1)
            loss_zero = criterion(lvl_zero_logits, target_zero)
            loss_zero = torch.tensor(0.) if torch.isnan(loss_zero) else loss_zero
            
            loss_one = criterion(lvl_one_logits, target_one)
            loss_one = torch.tensor(0.) if torch.isnan(loss_one) else loss_one

            loss_two = criterion(lvl_two_logits, target_two)
            loss_two = torch.tensor(0.) if torch.isnan(loss_two) else loss_two
            
            loss_three = criterion(lvl_three_logits, target_three)
            loss_three = torch.tensor(0.) if torch.isnan(loss_three) else loss_three

            loss = loss_zero + loss_one + loss_two + loss_three

        # Optimization step
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()

        if gen_config.optimizer["gradient_clipping"] > 0:
            torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=gen_config.optimizer["gradient_clipping"])

        scaler.step(optimizer)

        # Updates the scale for next iteration.
        scaler.update()

        batch_loss = loss.item()
        accumulated_loss += loss.item()
        accumulated_loss_lvl_zero += loss_zero.item()
        accumulated_loss_lvl_one += loss_one.item()
        accumulated_loss_lvl_two += loss_two.item()
        accumulated_loss_lvl_three += loss_three.item()

        progress_bar.set_postfix({"batch_loss": batch_loss})

        del data 

    metrics["loss_level_zero"] = accumulated_loss_lvl_zero / num_batches
    metrics["loss_level_one"] = accumulated_loss_lvl_one / num_batches
    metrics["loss_level_two"] = accumulated_loss_lvl_two / num_batches
    metrics["loss_level_three"] = accumulated_loss_lvl_three / num_batches
    metrics["total_loss"] = accumulated_loss/ num_batches

    top_20_flowsheets = metrics["top_20_flowsheets"]
    del metrics["top_20_flowsheets"]
    return metrics, top_20_flowsheets

def evaluate(eval_type: str, gen_config, env_config, network: FlowsheetNetwork, sys_ind: int, destination_path):
    #gen_config.gumbeldore_config["destination_path"] = None

    gumbeldore_dataset = GumbeldoreDataset(gen_config, env_config)

    metrics = gumbeldore_dataset.generate_dataset(copy.deepcopy(network.get_weights()), memory_aggressive=False, system_index= sys_ind, destination_path = destination_path)
    top_20_flowsheets = metrics["top_20_flowsheets"]

    metrics = {
        f"{eval_type}_mean_top_20_obj": metrics["mean_top_20_obj"],
        f"{eval_type}_best_obj": metrics['best_gen_obj'],
    }

    print("Evaluation done")
    print(f"Eval ({eval_type}) best obj: {metrics[f'{eval_type}_best_obj']:.3f}")
    print(f"Eval ({eval_type}) mean top 20 obj: {metrics[f'{eval_type}_mean_top_20_obj']:.3f}")

    return metrics, top_20_flowsheets

if __name__ == '__main__':
    print(">> Flowsheet Design using Gumbeldore")

    parser = argparse.ArgumentParser(description='Experiment')
    parser.add_argument('--config', help="Path to optional config relative to main.py")
    args = parser.parse_args()

    gen_config = GeneralConfig()
    env_config = EnvConfig()

    os.environ["CUDA_VISIBLE_DEVICES"] = gen_config.CUDA_VISIBLE_DEVICES
    num_gpus = len(gen_config.CUDA_VISIBLE_DEVICES.split(","))
    ray.init(num_gpus=num_gpus, log_to_driver=False, logging_level="info", num_cpus=16) 
    print(ray.available_resources())

    logger = Logger(args, gen_config.results_path, gen_config.log_to_file)
    logger.log_hyperparams(gen_config)

    # set up mlflow connection 
    set_mlflow_connection() 
    model_start_time = f'test' + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if gen_config.mlflow_experiment is None:
        mlflow.set_experiment('test' + '_' + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))                        
    else:
        mlflow.set_experiment(gen_config.mlflow_experiment)

    # Fix random number generator seed for better reproducibility
    np.random.seed(gen_config.seed)
    torch.manual_seed(gen_config.seed)

    # Setup the neural network for training
    network = FlowsheetNetwork(gen_config, env_config, gen_config.training_device)

    # Load checkpoint if needed
    if gen_config.load_checkpoint_from_path is not None:
        print(f"Loading checkpoint from path {gen_config.load_checkpoint_from_path}")
        checkpoint = torch.load(gen_config.load_checkpoint_from_path)
        print(f"{checkpoint['epochs_trained']} episodes have been trained in the loaded checkpoint.")
    else:
        checkpoint = {
            "model_weights": None,
            "best_model_weights": None,
            "optimizer_state": None,
            "epochs_trained": 0,
            "validation_metric": float("-inf"),   # objective of the best sequence designed during validation.
            "best_validation_metric": float("-inf")  # corresponding to best model weights
        }
    if checkpoint["model_weights"] is not None:
        network.load_state_dict(checkpoint["model_weights"])

    print(f"Policy network is on device {gen_config.training_device}")
    network.to(network.device)
    network.eval()

    if gen_config.num_epochs > 0:

        # Training loop
        print(f"Starting training for {gen_config.num_epochs} epochs.")

        best_model_weights = checkpoint["best_model_weights"]  # can be None
        best_validation_metric = checkpoint["best_validation_metric"]

        print("Setting up optimizer.")
        optimizer = torch.optim.Adam(
            network.parameters(),
            lr=gen_config.optimizer["lr"],
            weight_decay=gen_config.optimizer["weight_decay"]
        )
        if checkpoint["optimizer_state"] is not None and gen_config.load_optimizer_state:
            print("Loading optimizer state from checkpoint.")
            optimizer.load_state_dict(
                checkpoint["optimizer_state"]
            )
        print("Setting up LR scheduler")
        _lambda = lambda epoch: gen_config.optimizer["schedule"]["decay_factor"] ** (
                    checkpoint["epochs_trained"] // gen_config.optimizer["schedule"]["decay_lr_every_epochs"])
        scheduler = LambdaLR(optimizer, lr_lambda=_lambda)

        start_time_counter = None
        if gen_config.wall_clock_limit is not None:
            print(f"Wall clock limit of training set to {gen_config.wall_clock_limit / 3600} hours")
            start_time_counter = time.perf_counter()

        # uniformaly sample system indices for making problem instances
        system_index = np.random.choice([0, 1, 2, 3], size=gen_config.num_epochs)
            
        with mlflow.start_run(run_name = model_start_time):
            for epoch in range(gen_config.num_epochs):
                print("------")
                print(f"Generating dataset.")
                network_weights = copy.deepcopy(network.get_weights())

                mlflow.log_params({k: v for k, v in vars(gen_config).items() if isinstance(v, (int, float, str, bool))})

                generated_loggable_dict, generated_text_to_save = train_for_one_epoch(
                    epoch, gen_config, env_config, network, network_weights, optimizer, best_validation_metric, system_index[epoch], gen_config.gumbeldore_config['destination_path']
                )

                # Save model
                checkpoint["model_weights"] = copy.deepcopy(network.get_weights())
                checkpoint["optimizer_state"] = copy.deepcopy(
                    dict_to_cpu(optimizer.state_dict())
                )
                val_metric = generated_loggable_dict["best_gen_obj"]   # measure by best objective found during sampling
                checkpoint["validation_metric"] = val_metric
                save_checkpoint(checkpoint, "last_model.pt", gen_config)

                # log metrics per epoch 
                for key, val in generated_loggable_dict.items():
                    mlflow.log_metric(key, val, step=epoch)

                if val_metric > best_validation_metric:
                    print(">> Got new best model.")
                    checkpoint["best_model_weights"] = copy.deepcopy(checkpoint["model_weights"])
                    checkpoint["best_validation_metric"] = val_metric
                    best_model_weights = checkpoint["best_model_weights"]
                    best_validation_metric = val_metric
                    save_checkpoint(checkpoint, "best_model.pt", gen_config)

                if start_time_counter is not None and time.perf_counter() - start_time_counter > gen_config.wall_clock_limit:
                    print("Time exceeded. Stopping training.")
                    break

    if gen_config.num_epochs == 0:
        print(f"Testing with loaded model.")
    else:
        print(f"Testing with best model.")
        checkpoint = torch.load(os.path.join(gen_config.results_path, "best_model.pt"), weights_only= False)
        network.load_state_dict(checkpoint["model_weights"])

    if checkpoint["model_weights"] is None and gen_config.num_epochs == 0:
        print("WARNING! No training was performed, but also no checkpoint to load was given. "
              "Evaluating with random model.")

    torch.cuda.empty_cache()
    destinaton_path_results = gen_config.results_path

    # uniformaly sample system indices for making problem instances
    system_index = [0, 1, 2, 3]
    
    for sys_ind in system_index: 
        with torch.no_grad():
            file_name_pickle = f'test_20_top_flowsheets' + '_' + 'sys_index' + '_' + str(sys_ind) + '.pickle'
            results_path = os.path.join(gen_config.results_path, file_name_pickle)

            test_loggable_dict, test_text_to_save = evaluate('test', gen_config, env_config, network, sys_ind, results_path)
            print(">> TEST")
            print(test_loggable_dict)
            logger.log_metrics(test_loggable_dict, step=0, step_desc="test")
            print(test_text_to_save)
            file_name_txt = f'test_20_top_flowsheets' + '_' + 'sys_index' + '_' + str(sys_ind) + '.txt'
            logger.text_artifact(os.path.join(gen_config.results_path, file_name_txt), test_text_to_save)

    print("Finished. Shutting down ray.")
    ray.shutdown()

