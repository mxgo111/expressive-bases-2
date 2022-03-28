'''
Instructions: Don't delete any configurations, but instead inherit and make modifications as necessary!
'''
import numpy as np

class MultipleRuns():
    """
    Class for running multiple sets of hyperparameters
    """
    def __init__(self, lst):
        self.hyps = lst
        self.length = len(lst)


class MultipleModels():
    """
    Class for storing multiple models based on training epochs
    """
    def __init__(self, lst):
        self.epochs = lst


# Base Configuration - don't change!
class BaseConfig():
    def __init__(self):
        self.hyp = {}

        # experiment parameters
        self.hyp["experiment_name"] = "REPLACE_ME_IN_FUTURE_SUBCLASSES"

        # dataset parameters
        self.hyp["dataset"] = "cubic"
        self.hyp["dataset_min_range"] = -1
        self.hyp["dataset_max_range"] = 1
        self.hyp["gap_min_range"] = -0.2
        self.hyp["gap_max_range"] = 0.2
        self.hyp["train_dataset_size"] = 100
        self.hyp["output_var"] = 0.01

        # model parameters
        self.hyp["basis"] = "FullyConnected" # or Custom, Legendre, Linear, Sine+Cosine
        self.hyp["custom_basis_path"] = None # relevant if using Custom, starting from current dir
        self.hyp["final_layer"] = "FullyConnected"
        self.hyp["model"] = "BayesianRegression" # if GP, ignore almost every other parameter
        self.hyp["activation"] = "ReLU"
        self.hyp["num_bases"] = 20
        self.hyp["layers"] = [1, 50, self.hyp["num_bases"], 1]
        self.hyp["output_activation"] = True
        self.hyp["bias"] = True
        self.hyp["rand_init_mean"] = 0.0
        self.hyp["rand_init_std"] = 1.0
        self.hyp["rand_init_seed"] = MultipleRuns([0, 1]) # goto for multiple runs of the same parameters
        self.hyp["w_prior_var"] = 1.0
        self.hyp["loss"] = "MLE"
        self.hyp["k"] = 0.1 # relevant if using MAP Loss
        self.hyp["learning_rate"] = 1e-3
        self.hyp["optimizer_weight_decay_l2"] = 0.0
        self.hyp["total_epochs"] = 5000

        # sampling and other parameters
        self.hyp["train_print_freq"] = 1000
        self.hyp["posterior_prior_predictive_samples"] = 3000
        self.hyp["add_output_noise_prior_predictive_sampling"] = True
        self.hyp["add_output_noise_posterior_predictive_sampling"] = True
        self.hyp["num_points_linspace_visualize"] = 500


# First Configuration
class FirstConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        # experiment_name
        self.hyp["experiment_name"] = "TestExperiment"
        self.hyp["rand_init_seed"] = MultipleRuns([0, 1])
        self.hyp["total_epochs"] = MultipleModels([0, 100, 1000])
        self.hyp["train_dataset_size"] = 20
        self.hyp["num_bases"] = 40
        self.hyp["layers"] = [1, 50, self.hyp["num_bases"], 1]

# First Configuration
class SecondConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        # experiment_name
        self.hyp["experiment_name"] = "LeakyReLUTest"
        self.hyp["rand_init_seed"] = MultipleRuns([0, 0])
        self.hyp["total_epochs"] = MultipleModels([0, 100])
        self.hyp["train_dataset_size"] = 20
        self.hyp["num_bases"] = 40
        self.hyp["layers"] = [1, 50, self.hyp["num_bases"], 1]
        self.hyp["activation"] = MultipleRuns(["LeakyReLU", "ReLU"])

class ThirdConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        # experiment_name
        self.hyp["experiment_name"] = "LegendreTest"
        self.hyp["basis"] = "Legendre"
        self.hyp["num_bases"] = 7
        self.hyp["dataset_min_range"] = 1.5
        self.hyp["dataset_max_range"] = 6
        self.hyp["gap_min_range"] = 2.5
        self.hyp["gap_max_range"] = 4.5
        self.hyp["dataset"] = "xsinx"

class RetestingLegendreConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        # experiment_name
        self.hyp["experiment_name"] = "LegendreReTest"
        self.hyp["basis"] = "Legendre"
        self.hyp["num_bases"] = 7
        self.hyp["dataset_min_range"] = 0
        self.hyp["dataset_max_range"] = 6
        self.hyp["gap_min_range"] = 2
        self.hyp["gap_max_range"] = 4.5
        self.hyp["dataset"] = "xsinx"


class TestingLinearConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        # experiment_name
        self.hyp["experiment_name"] = "NEWEXPERIMENT"
        self.hyp["basis"] = "OneBasisIsData"
        self.hyp["num_bases"] = 7
        self.hyp["dataset_min_range"] = -1
        self.hyp["dataset_max_range"] = 1
        self.hyp["gap_min_range"] = -0.2
        self.hyp["gap_max_range"] = 0.2
        self.hyp["dataset"] = "cubic"


class FourthConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        # experiment_name
        self.hyp["experiment_name"] = "FCTest"
        self.hyp["basis"] = "FullyConnected"
        self.hyp["activation"] = "LeakyReLU"
        self.hyp["num_bases"] = 20
        self.hyp["dataset_min_range"] = 1.5
        self.hyp["dataset_max_range"] = 6
        self.hyp["gap_min_range"] = 2.5
        self.hyp["gap_max_range"] = 4.5
        self.hyp["dataset"] = "xsinx"
        self.hyp["w_prior_var"] = 3.0
        self.hyp["total_epochs"] = 1

class goodBubble(BaseConfig):
    def __init__(self):
        super().__init__()
        # experiment_name
        # experiment parameters
        self.hyp["experiment_name"] = "goodBubble"

        # dataset parameters
        self.hyp["dataset"] = "xsinx"
        # self.hyp["dataset_min_range"] = -1
        # self.hyp["dataset_max_range"] = 1
        # self.hyp["gap_min_range"] = -0.2
        # self.hyp["gap_max_range"] = 0.2
        self.hyp["dataset_min_range"] = 1.5
        self.hyp["dataset_max_range"] = 6
        self.hyp["gap_min_range"] = 2.5
        self.hyp["gap_max_range"] = 4.5
        self.hyp["train_dataset_size"] = 50
        self.hyp["output_var"] = 0.25

        # model parameters
        self.hyp["basis"] = "Legendre" # or Custom, Legendre, Sine+Cosine
        self.hyp["custom_basis_path"] = None # relevant if using Custom, starting from current dir
        self.hyp["final_layer"] = "FullyConnected"
        self.hyp["model"] = "BayesianRegression"
        self.hyp["activation"] = "LeakyReLU"
        self.hyp["num_bases"] = 4
        self.hyp["layers"] = [1, 50, self.hyp["num_bases"], 1]
        self.hyp["output_activation"] = True
        self.hyp["bias"] = True
        self.hyp["rand_init_mean"] = 0.0
        self.hyp["rand_init_std"] = 1.0
        self.hyp["rand_init_seed"] = 0 # goto for multiple runs of the same parameters
        self.hyp["w_prior_var"] = MultipleRuns([1.0, 5.0])
        self.hyp["loss"] = "MLE"
        self.hyp["k"] = 0.1 # relevant if using MAP Loss
        self.hyp["learning_rate"] = 1e-3
        self.hyp["optimizer_weight_decay_l2"] = 0.0
        self.hyp["total_epochs"] = 6000

        # sampling and other parameters
        self.hyp["train_print_freq"] = 1000
        self.hyp["posterior_prior_predictive_samples"] = 5000
        self.hyp["add_output_noise_prior_predictive_sampling"] = True
        self.hyp["add_output_noise_posterior_predictive_sampling"] = True
        self.hyp["num_points_linspace_visualize"] = 1000
