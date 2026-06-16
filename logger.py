import os
import inspect
import json, pickle
from typing import Optional
import numpy as np


class Logger:
    def __init__(self, args, results_path: str, log_to_file: bool):
        self.results_path = results_path
        self.log_to_file = log_to_file

        self.file_log_path = os.path.join(self.results_path, "log.txt")
        if self.log_to_file:
            os.makedirs(self.results_path, exist_ok=True)

    def log_hyperparams(self, config_object):
        attributes = inspect.getmembers(config_object, lambda a: not (inspect.isroutine(a)))
        attributes = [a for a in attributes if not (a[0].startswith('__') and a[0].endswith('__'))]
        attribute_dict = {}

        def add_to_attribute_dict(a):
            for key, value in a:
                key = key.replace("+", "_plus")
                key = key.replace("@", "_at")
                if isinstance(value, dict):
                    add_to_attribute_dict([(f"{key}.{k}", v) for k, v in value.items()])
                else:
                    if key not in ["devices_for_eval_workers"] and len(str(value)) <= 500:
                        attribute_dict[key] = value

        add_to_attribute_dict(attributes)

        if self.log_to_file:
            with open(self.file_log_path, "a+") as f:
                f.write(json.dumps({"hyperparameters": attribute_dict}))
                f.write("\n")

    def log_metrics(self, metrics: dict, step: Optional[int] = None, step_desc: Optional[str] = "epoch"):
        if self.log_to_file:
            if step is not None:
                metrics[step_desc] = step
            with open(self.file_log_path, "a+") as f:
                f.write(json.dumps(metrics, default=str))
                f.write("\n")

    def text_artifact(self, dest_text: str):

        pickle_file_path = f"{dest_text}/test_flowsheets.pickle"
        final_logger_path = f"{dest_text}/final_test_logger.txt"
        
        with open(pickle_file_path, "rb") as f:
            existing_fs = pickle.load(f)
        
        top_k = sorted(existing_fs, key=lambda x: x["obj"], reverse=True)[:20]     

        # Get overall best metrics and flowsheets
        mean_top_20_obj = np.array([x["obj"] for x in top_k]).mean()
        top_20_flowsheets = [{x["identifier"]: x["obj"] for x in top_k}]
        best_gen_obj = top_k[0]["obj"]

        with open(final_logger_path, "w") as f:
            f.write(f"Mean obj top 20 flowsheets: {mean_top_20_obj}, ")
            f.write(f"Top 20 flowsheets identifiers: {top_20_flowsheets}, ")
            f.write(f"Best objective: {best_gen_obj}\n")
                    
            for x in top_k:
                pi = x["problem_instance"]
                graph = x["graph"]
                identifier = x["identifier"]
                f.write(f"identifier: {identifier}, ")
                f.write(f"situation index: {pi.get('feed_situation_index')}, ")
                f.write(f"components in feed: {pi.get('indices_components_in_feeds')}, ")
                f.write(f"feeds: {pi.get('list_feed_streams')}, ")
                f.write(f"npv_normed: {x.get('obj')}, ")
                f.write(f"per_ratio: {x.get('per_ratio')}, ")
                f.write(f"npv_wo_app_cost: {x.get('npv_wo_app_cost')}, ")
                f.write(f"npv_raw: {x.get('npv_raw')}, ")
                f.write(f"total_units_placed: {x.get('total_units_placed')}, ")
                f.write("units:\n")
                for node_idx, node_data in graph._node.items():
                    f.write(f"  node {node_idx}:\n")
                    for k, v in node_data.items():
                        f.write(f"    {k}: {v}\n")
                f.write("\n\n")