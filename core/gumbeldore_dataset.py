import copy, os, pickle, sys, ray, torch 
from model.policy_arch import FlowsheetNetwork
from environment.environment_actions_graphs import FlowsheetDesign
import numpy as np
from ray.thirdparty_files import psutil
from tqdm import tqdm
from core.abstract import Instance
import core.stochastic_beam_search as sbs
from typing import List, Tuple, Any, Optional
from core.incremental_sbs import IncrementalSBS

os.environ["RAY_DEDUP_LOGS"] = "0"

@ray.remote
class JobPool:
    def __init__(self, problem_instances: List[Instance]):
        self.jobs = [(i, instance) for i, instance in enumerate(problem_instances)]
        self.job_results = []

    def get_jobs(self, n_items: int):
        if len(self.jobs) > 0:
            items = self.jobs[:n_items]
            self.jobs = self.jobs[n_items:]
            return items
        else:
            return None

    def push_results(self, results: List[Tuple[int, Any]]):
        self.job_results.extend(results)

    def fetch_results(self):
        results = self.job_results
        self.job_results = []
        return results
    

class GumbeldoreDataset:
    def __init__(self, gen_config, env_config):
        self.gen_config = gen_config
        self.gumbeldore_config = gen_config.gumbeldore_config
        self.env_config = env_config
        self.devices_for_workers: List[str] = self.gumbeldore_config["devices_for_workers"]

    def generate_dataset(self, network_weights: dict, best_objective: Optional[float] = None, memory_aggressive: bool = False, system_index: int = None, destination_path: str = None):
        
        """
        Parameters:
            network_weights: [dict] Network weights to use for generating data.
            memory_aggressive: [bool] If True, IncrementalSBS is performed "memory aggressive" meaning that
                intermediate states in the search tree are not stored after transitioning from them, only their
                policies.
        """

        batch_size_gpu, batch_size_cpu = (self.gumbeldore_config["batch_size_per_worker"],
                                            self.gumbeldore_config["batch_size_per_cpu_worker"])

        random_instance = self.env_config.create_random_problem_instance(system_index)
        problem_instances = FlowsheetDesign.design_flowsheets(random_instance, self.gen_config, self.env_config)

        job_pool = JobPool.remote(copy.deepcopy(problem_instances))
        results = [None] * len(problem_instances)

        # Check if we should pin the workers to core
        cpu_cores = [None] * len(self.devices_for_workers)
        if self.gumbeldore_config["pin_workers_to_core"] and sys.platform == "linux":
            # Get available core IDs
            affinity = list(os.sched_getaffinity(0))
            cpu_cores = [affinity[i % len(cpu_cores)] for i in range(len(self.devices_for_workers))]

        # Kick off workers
        future_tasks = [
            async_sbs_worker.remote(
            #async_sbs_worker(
                self.gen_config, self.env_config, job_pool, network_weights, device,
                batch_size_gpu if device != "cpu" else batch_size_cpu,
                cpu_cores[i], best_objective, memory_aggressive
            )
            for i, device in enumerate(self.devices_for_workers)
        ] 

        with tqdm(total=len(problem_instances)) as progress_bar:
            while True:
                # Check if all workers are done. If so, break after this iteration
                do_break = len(ray.wait(future_tasks, num_returns=len(future_tasks), timeout=0.5)[1]) == 0
                fetched_results = ray.get(job_pool.fetch_results.remote()) 
                for (i, result) in fetched_results:
                    results[i] = result
                if len(fetched_results):
                    progress_bar.update(len(fetched_results))
                if do_break:
                    break

        ray.get(future_tasks)
        del job_pool
        del network_weights
        torch.cuda.empty_cache()

        return self.process_results(problem_instances, results, destination_path)
    

    def process_results(self, problem_instances, results, destination_path):
            """
            Processes the results from Gumbeldore search and save it to a pickle. Each trajectory will be represented as a dict with the
            following keys and values
            "action_seq": List[List[int]] Actions which need to be taken on each index to create the molecule
            "obj": [float] NPV value

            Then:
            1. If the dataset already exists at the path where to save, we load it, merge them and take the best from the
                merged dataset.

            Then returns the following dictionary:
            - "mean_best_gen_obj": Mean best generated obj. -> over the unmerged best flowsheets generated
            - "best_gen_obj": Best generated obj. -> Best obj. of the unmerged flowsheets generated
            - "worst_gen_obj": Worst generated obj. -> Worst obj. of the unmerged flowsheets generated
            - "mean_top_20_obj": Mean top 20 obj. -> over the merged best flowsheets
            - "top_20_flowsheets": A list of flowsheets with obj. of the top 20 obj.
            """
            instances = []
            metrics_return = dict()
            seen = set()
            #instances_dict = dict()  # Use a dict to directly avoid duplicates

            for i, _ in enumerate(problem_instances):
                for flowsheet in results[i]:  
                    if flowsheet.objective > float("-inf"):
                        hist_key = tuple(flowsheet.history)
                        if hist_key in seen:
                            continue
                        seen.add(hist_key)
                        
                        instances.append(dict(
                            problem_instance = flowsheet.problem_instance,
                            identifier = flowsheet.identifier, 
                            action_seq=flowsheet.history,
                            obj=flowsheet.objective,
                            graph = flowsheet.sim.graph, 
                            levels = flowsheet.level_list,
                            status = flowsheet.current_state['completed_design']
                        ))
            #generated_fs = list(instances_dict.values())
            generated_fs = instances
            generated_fs = sorted(generated_fs, key=lambda x: x["obj"], reverse=True)[:self.gumbeldore_config["num_trajectories_to_keep"]]
            generated_objs = np.array([x["obj"] for x in generated_fs])
            metrics_return["mean_best_gen_obj"] = generated_objs.mean()
            metrics_return["best_gen_obj"] = generated_objs[0]
            metrics_return["worst_gen_obj"] = generated_objs[-1]

            # Now check if there already is a data file, and if so, load it and merge it.
            #destination_path = self.gumbeldore_config["destination_path"]
            merged_fs = generated_fs
            feed_index = generated_fs[0]["problem_instance"]["feed_situation_index"]
            if destination_path is not None:
                if os.path.isfile(destination_path):
                    with open(destination_path, "rb") as f:
                        existing_fs = pickle.load(f)  # list of dicts
                    
                    existing_by_key = {(x["problem_instance"]["feed_situation_index"], x["identifier"]): x for x in existing_fs}
                    for x in merged_fs: 
                        key = (x["problem_instance"]["feed_situation_index"], x["identifier"])
                        existing_by_key[key] = x
                    merged_fs = list(existing_by_key.values())
                    merged_fs = [x for x in merged_fs if x["problem_instance"]["feed_situation_index"] == feed_index]
                
                merged_fs = sorted(merged_fs, key=lambda x: x["obj"], reverse=True)[
                                    :self.gumbeldore_config["num_trajectories_to_keep"]]
                # Pickle the generated data again
                with open(destination_path, "wb") as f:
                    pickle.dump(merged_fs, f)

            # Get overall best metrics and flowsheets
            metrics_return["mean_top_20_obj"] = np.array([x["obj"] for x in merged_fs[:20]]).mean()
            metrics_return["mean_kept_obj"] = np.array([x["obj"] for x in merged_fs]).mean()
            metrics_return["top_20_flowsheets"] = [{x["identifier"]: x["obj"] for x in merged_fs[:20]}]

            return metrics_return


@ray.remote(max_calls=1)
def async_sbs_worker(gen_config, env_config, job_pool: JobPool, network_weights: dict,
                     device: str, batch_size: int,
                     cpu_core: Optional[int] = None,
                     best_objective: Optional[float] = None,
                     memory_aggressive: bool = False
                     ):
    
    def child_log_probability_fn(trajectories: List[FlowsheetDesign]) -> [np.array]:
        return FlowsheetDesign.log_probability_fn(config = gen_config, trajectories=trajectories, network=network, device=device)
    
    def batch_leaf_evaluation_fn(trajectories: List[FlowsheetDesign]) -> np.array:
        objs = [traj.objective for traj in trajectories]
        return objs

    def child_transition_fn(trajectory_action_pairs: List[Tuple[FlowsheetDesign, int]]):
        return [traj.transition_fn(action) for traj, action in trajectory_action_pairs]
    
    # Pin worker to core if wanted
    if cpu_core is not None:
        os.sched_setaffinity(0, {cpu_core})
        psutil.Process().cpu_affinity([cpu_core])

    with torch.no_grad():

        if gen_config.CUDA_VISIBLE_DEVICES:
            # override ray's limiting of GPUs
            os.environ["CUDA_VISIBLE_DEVICES"] = gen_config.CUDA_VISIBLE_DEVICES

        device = torch.device(device)
        network = FlowsheetNetwork(gen_config, env_config, device)
        network.load_state_dict(network_weights)
        network.to(network.device)
        network.eval()

        while True:
            batch = ray.get(job_pool.get_jobs.remote(batch_size))

            if batch is None:
                break

            idx_list = [i for i, _ in batch]
            root_nodes = [instance for _, instance in batch]

            if gen_config.gumbeldore_config["search_type"] == "beam_search":

                # Deterministic beam search.
                beam_leaves_batch: List[List[sbs.BeamLeaf]] = sbs.stochastic_beam_search(
                    child_log_probability_fn=child_log_probability_fn,
                    child_transition_fn=child_transition_fn,
                    root_states=root_nodes,
                    beam_width=gen_config.gumbeldore_config["beam_width"],
                    deterministic=True
                )
            else:
                inc_sbs = IncrementalSBS(root_nodes, child_log_probability_fn, child_transition_fn,
                                         leaf_evaluation_fn=FlowsheetDesign.to_max_evaluation_fn,
                                         batch_leaf_evaluation_fn=batch_leaf_evaluation_fn,
                                         memory_aggressive=False)

                if gen_config.gumbeldore_config["search_type"] == "tasar":
                    beam_leaves_batch: List[List[sbs.BeamLeaf]] = inc_sbs.perform_tasar(
                        beam_width=gen_config.gumbeldore_config["beam_width"],
                        deterministic=gen_config.gumbeldore_config["deterministic"],
                        nucleus_top_p=gen_config.gumbeldore_config["nucleus_top_p"],
                        replan_steps=gen_config.gumbeldore_config["replan_steps"],
                        sbs_keep_intermediate=gen_config.gumbeldore_config["keep_intermediate_trajectories"]
                    )
                elif gen_config.gumbeldore_config["search_type"] == "wor":
                    beam_leaves_batch: List[List[sbs.BeamLeaf]] = inc_sbs.perform_incremental_sbs(
                        beam_width=gen_config.gumbeldore_config["beam_width"],
                        num_rounds=gen_config.gumbeldore_config["num_rounds"],
                        nucleus_top_p=gen_config.gumbeldore_config["nucleus_top_p"],
                        sbs_keep_intermediate=gen_config.gumbeldore_config["keep_intermediate_trajectories"],
                        best_objective=best_objective
                    )

            results_to_push = []
            for j, result_idx in enumerate(idx_list):
                result: List[FlowsheetDesign] = [x.state for x in beam_leaves_batch[j][:gen_config.gumbeldore_config["num_trajectories_to_keep"]]]

                # Check if they need objective evaluation (this will only be true for deterministic beam search
                if result[0].objective is None:
                    batch_leaf_evaluation_fn(result)
                results_to_push.append((result_idx, result))

            ray.get(job_pool.push_results.remote(results_to_push)) 

            if device != "cpu":
                torch.cuda.empty_cache()

    del network
    del network_weights
    torch.cuda.empty_cache()
