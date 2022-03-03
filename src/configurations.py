'''
Instructions: Don't delete any configurations, but instead inherit and make modifications as necessary!
'''

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
        self.hyp["basis"] = "FullyConnected"
        self.hyp["final_layer"] = "FullyConnected"
        self.hyp["model"] = "BayesianRegression"
        self.hyp["activation"] = MultipleRuns(["ReLU"])
        self.hyp["num_bases"] = 20
        self.hyp["layers"] = [1, 50, self.hyp["num_bases"], 1]
        self.hyp["output_activation"] = True
        self.hyp["bias"] = True
        self.hyp["rand_init_mean"] = 0.0
        self.hyp["rand_init_std"] = 1.0
        self.hyp["rand_init_seed"] = MultipleRuns([2, 5])
        self.hyp["w_prior_var"] = MultipleRuns([1.0])
        self.hyp["loss"] = MultipleRuns(["MLE"])
        self.hyp["k"] = 0.1 # relevant if using MAP Loss
        self.hyp["learning_rate"] = 1e-3
        self.hyp["optimizer_weight_decay_l2"] = 0.0
        self.hyp["total_epochs"] = MultipleModels([3000, 5000])

        # sampling and other parameters
        self.hyp["train_print_freq"] = 1000
        self.hyp["posterior_predictive_samples"] = 3000
        self.hyp["add_output_noise_prior_predictive_sampling"] = True
        self.hyp["add_output_noise_posterior_predictive_sampling"] = True


# First Configuration
class FirstConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        # experiment_name
        self.hyp["experiment_name"] = "TestExperiment"
