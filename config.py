import numpy as np 
import os, datetime, copy
import environment.phase_equilibria.phase_eq_handling as phase_eq_generation
from environment import units 
import torch

class GeneralConfig:

    """
    General configurations for the policy archiecture, training, and sampling 

    """
        

    def __init__(self):
        self.seed = 42

        # Network and environment
        self.latent_dim = 256 #latent dimension for Core transformer 
        self.num_transformer_blocks = 10 # Number of layers in the stack of transformer blocks for the architecture
        self.num_heads = 16 # Number of heads in the multihead attention.
        self.dropout = 0. # Dropout for feedforward layer in a transformer block.
        self.num_trf_flow_blocks = 3 #num of transformer blocks for flow expert 
        self.flow_latent_dim = 64

        # Loading trained checkpoints to resume training or evaluate
        self.load_checkpoint_from_path = None  # If given, model checkpoint is loaded from this path.
        self.load_optimizer_state = False  # If True, the optimizer state is also loaded.

        # Training
        self.num_dataloader_workers = 3  # Number of workers for creating batches for training
        self.CUDA_VISIBLE_DEVICES = "0,1"  # Must be set, as ray can have problems detecting multiple GPUs
        self.training_device = "cuda:0"  # Device on which to perform the supervised training
        self.num_epochs = 100 # Number of epochs (i.e., passes through training set) to train
        self.batch_size_training = 32 #Batch size to use for the supervised training during finetuning. 
        self.num_batches_per_epoch = None  # Can be None, then we just do one pass through generated dataset

        self.wall_clock_limit = None
        self.mlflow_experiment = 'test'
        

        # Optimizer
        self.optimizer = {
            "lr": 1e-4,  # learning rate
            "weight_decay": 0,
            "gradient_clipping": 1.,  # Clip gradient to given L2-norm. Set to 0 if no clipping should be performed.
            "schedule": {
                "decay_lr_every_epochs": 1,
                "decay_factor": 1
            }
        }

        # Self-improvement sequence decoding
        self.gumbeldore_config = {

            # Number of trajectories with the the highest objective function evaluation to keep for training
            "num_trajectories_to_keep": 30,
            "keep_intermediate_trajectories": True,  # if True, we consider all intermediate, terminable trajectories
            "devices_for_workers": ["cuda:0"] * 1,
            "destination_path": "./data/generated_flowsheets.pickle",
            "batch_size_per_worker": 1, 
            "batch_size_per_cpu_worker": 1,
            "search_type": "tasar",
            "beam_width": 32,
            "replan_steps": 12,
            "num_rounds": 1,  # if it's a tuple, then we sample as long as it takes to obtain a better trajectory, but for a minimum of first entry rounds and a maximum of second entry rounds
            "deterministic": False,  # Only use for gumbeldore_eval=True below, switches to regular beam search.
            "nucleus_top_p": 1.,
            "pin_workers_to_core": False
        }

        # Results and logging
        self.results_path = os.path.join("./results",
                                         datetime.datetime.now().strftime(
                                             "%Y-%m-%d--%H-%M-%S"))  # Path to store the model weights
        self.log_to_file = True


class EnvConfig:

    """
    Graph-first configuration for the flowsheet environment.
    - Loads phase equilibrium data (VLE/LLE) for allowed systems
    - Defines unit catalog, discretization of continuous specs, and action limits
    - Provides economics knobs for NPV calculation
    - Generates random feed problem instances
    """

    def __init__(self):

        # ----- Core dimensional settings -----

        # maximum number of components present simultaneously in a flowsheet
        self.max_number_of_components = 3
        self.max_simulator_tries = 30

        # ----- Phase equilibrium / property data -----
        self.systems_allowed = {
            "acetone_chloroform": True,
            "ethanol_water": True,
            "n-butanol_water": True,
            "water_pyridine": True
        }
        self.dicretization_parameter_lle = 5       # LLE simplex discretization
        self.curvature_parameter_vle = 0.001       # VLE curvature fitting

        self.phase_eq_generator = phase_eq_generation.PhaseEqHandling(
            directory=os.path.join(os.getcwd(), "environment", "phase_equilibria"),
            systems_allowed=self.systems_allowed
        )
        self.phase_eq_generator.load_phase_eqs(
            num_comp_lle=self.max_number_of_components,
            disc_para_lle=self.dicretization_parameter_lle,
            curvature_parameter=self.curvature_parameter_vle
        )

        # dict with pure component data (e.g., molar masses "M") used by literature NPV path
        self.dict_pure_component_data = self.phase_eq_generator.load_pure_component_data()

        # make a components tensor for Add solvent 
        self.component_names = list(self.dict_pure_component_data.keys())

        self.components_tensor = torch.tensor([self.dict_pure_component_data[name]["critical_data"] for name in self.component_names],
            dtype=torch.float32
        )

        # Shuffle option for feed component order (usually keep False for stable tests)
        self.shuffle_order_of_components = False

        # ----- NPV & PRICING/COSTS -----
        # Choose NPV variant: "generic" (per-mole pricing) or "literature" (per-kg pricing)
        self.npv_version = "literature"  # or "literature"
        self.norm_npv = True  # also compute a normalized NPV

        # Build dynamic price/cost maps that always match the global component list
        names = self.phase_eq_generator.names_components
        self.num_components = len(names)

        # Uniform defaults (edit these two numbers to tune all components at once)
        _uniform_product_price_per_mol = 100.0  # value for pure product streams (per mole) in "generic" mode
        _uniform_solvent_cost_per_mol = 10.0  # cost when a component is used as solvent (per mole) in "generic" mode

        _uniform_product_value_per_kg = 0.5  # value per kg in "literature" mode (set >0 to activate economics there)
        _uniform_solvent_cost_per_kg = 0.05  # cost per kg of solvent in "literature" mode

        # Generic (per-mole) pricing/costs: component-indexed dicts
        self.product_price_per_component = {idx: _uniform_product_price_per_mol for idx in range(self.num_components)}
        self.solvent_cost_per_component_mol = {idx: _uniform_solvent_cost_per_mol for idx in range(self.num_components)}

        # Literature (per-kg) pricing/costs
        self.lit_product_value_per_kg = _uniform_product_value_per_kg
        self.solvent_cost_per_component_kg = {idx: _uniform_solvent_cost_per_kg for idx in range(self.num_components)}

        self.steam_cost_per_kg = 0.04  # €/kg steam, used in literature NPV calc

        # Per-unit costs (used by compute_npv in the simulator)
        self.unit_costs_generic = {
            "add_solvent": 2.0,
            "distillation_column": 10.0,
            "decanter": 2.0,
            "split": 1.0,
            "mixer": 1.0
        }
        self.unit_costs_literature = {
            "add_solvent": 200000,
            "distillation_column": 1000000,
            "decanter": 200000,
            "split": 100000,
            "mixer": 100000,
            "steam_cost_per_kg": 0.04 # Energy/steam cost for distillation operating expenses
        }

        # Optional: override normalization scale for NPV (otherwise simulator uses total effective feed)
        # self.npv_normalization_scale = 1.0

        # ----- Unit catalog + discretization (RL action semantics) -----
        # Unit definitions: how many outputs, whether they need a continuous spec (range),
        # and which level to go to next.
        self.unit_types = {
            "distillation_column": {"num": 1, "output_streams": 2, "cont_range": [0.01, 0.99]},
            "decanter":            {"num": 1, "output_streams": 2, "cont_range": None},
            "split":               {"num": 1, "output_streams": 2, "cont_range": [0.01, 0.99]},
            "mixer":               {"num": 1, "output_streams": 1, "cont_range": None},
            "recycle":             {"num": 1, "output_streams": 1, "cont_range": None},
            "add_solvent":         {"num": 1,"output_streams": 1, "cont_range": [0.01, 10]},
        }

        self.outlet_to_idx = {"out0": 0, "out1": 1}
        self.max_outlets = 2

        self.distillation_column = units.distillation_column()

        # Stable index -> unit type mapping (flat catalog)
        self.units_map_indices_type = []
        for key in self.unit_types.keys():
            for _ in range(self.unit_types[key]["num"]):
                self.units_map_indices_type.append(key)

        self.num_units = len(self.unit_types)

        # index where "add_solvent" block starts
        self.add_solvent_start_index = None
        for i, key in enumerate(self.units_map_indices_type):
            if key == "add_solvent":
                self.add_solvent_start_index = i
                break
            
        # Action limits
        self.max_total_units = 10 # overall cap on placed units (excluding feed)
        self.max_distillation_columns = 5
        self.max_decanters = 5
        self.max_split = 5
        self.max_mixer = 5
        self.max_recycle = 2
        self.max_solvent = 1

        # ----- Recycle solver config -----
        # recycle guesses; see original env for semantics
        self.random_guesses_root_iteration = 0
        self.max_num_root_finding_interactions = 50
        self.use_wegstein = False
        self.wegstein_constant = 0.5
        self.wegstein_steps = 500
        self.epsilon = 0.001

        # size limits for recycle
        self.limit_recycle_size = 25

        # ----- Mass-balance tolerances (graph MB check) -----
        # feed-scaled tolerance (1% of total feed), with floors
        self.mb_relative_percent = 0.01
        self.mb_atol = 1e-6 # absolute tolerance
        self.mb_severe_atol = 1e-3
        self.mb_rtol = 1e-8 #relative tolerabce

        
        # List of mappings for params for distillation, split, add_solvent
        self.DF_distillation_map = np.linspace(0.01, 0.99, 100)
        self.split_ratio_map = np.linspace(0.01, 0.99, 100)
        self.acetone_conc_map = np.linspace(0.01, 9.99, 100)
        self.benzene_conc_map = np.linspace(0.01, 9.99, 100)
        self.butanol_conc_map = np.linspace(0.01, 9.99, 100)
        self.tol_conc_map = np.linspace(0.01, 9.99, 100)
        self.water_conc_map = np.linspace(0.01, 9.99, 100)

        self.add_solvent_comp_map = {
            "acetone": self.acetone_conc_map,
            "benzene": self.benzene_conc_map, 
            "n-butanol": self.butanol_conc_map, 
            "toluene": self.tol_conc_map, 
            "water": self.water_conc_map
                } 


    # Random feed generator
    def create_random_problem_instance(self, sampled_index):

        """
        Sample a feed situation of format:
          [[global indices for feed], [global indices allowed as solvent], number_of_feed_streams]
        Return:
          {
            "feed_situation_index": int,
            "indices_components_in_feeds": list[int],
            "list_feed_streams": [np.array(len=max_number_of_components)],
            "possible_ind_add_comp": list[int],
            "comp_order_feeds": [names],
            "lle_for_start": None,
            "vle_for_start": None
          }
          
        """
        feed_streams = []

        # sample a feed situation from the PEQ generator
        #sampled_index = np.random.randint(len(self.phase_eq_generator.feed_situations))
        sampled_situation = copy.deepcopy(self.phase_eq_generator.feed_situations[sampled_index])
        # sampled_situation: [[feed_global_indices], [allowed_add_comp_global_indices], num_streams]

        # optionally shuffle order of components in the feed streams
        if self.shuffle_order_of_components:
            np.random.shuffle(sampled_situation[0])

        # get names for readability
        names_in_streams = [
            self.phase_eq_generator.names_components[i] for i in sampled_situation[0]
        ]

        # generate the feed stream(s)
        for _ in range(sampled_situation[-1]):
            base = np.random.rand(len(sampled_situation[0]))
            base = base / (sampled_situation[-1] * np.sum(base))  # normalize and split across streams

            stream = np.zeros(self.max_number_of_components)
            stream[:len(base)] = base
            feed_streams.append(stream)

        return {
            "feed_situation_index": sampled_index,
            "indices_components_in_feeds": sampled_situation[0],
            "list_feed_streams": feed_streams,
            "possible_ind_add_comp": sampled_situation[1],
            "comp_order_feeds": names_in_streams,
            # placeholders; graph simulator determines current PEQ from indices on-the-fly
            "lle_for_start": None,
            "vle_for_start": None
        }