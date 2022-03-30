"""
Instructions: Don't delete any configurations, but instead inherit and make modifications as necessary!
"""
import numpy as np


class MultipleRuns:
    """
    Class for running multiple sets of hyperparameters
    """

    def __init__(self, lst):
        self.hyps = lst
        self.length = len(lst)


class MultipleModels:
    """
    Class for storing multiple models based on training epochs
    """

    def __init__(self, lst):
        self.epochs = lst


# Base Configuration - don't change!
class BaseConfig:
    def __init__(self):
        self.hyp = {}

        # experiment parameters
        self.hyp["experiment_name"] = "REPLACE_ME_IN_FUTURE_SUBCLASSES"

        # dataset parameters
        self.hyp["dataset"] = "cubic"
        self.hyp["dataset_min_range"] = -1
        self.hyp["dataset_max_range"] = 1
        self.hyp["visualize_min_range"] = -1
        self.hyp["visualize_max_range"] = 1
        self.hyp["gap_min_range"] = -0.5
        self.hyp["gap_max_range"] = 0.5
        self.hyp["train_dataset_size"] = 100
        self.hyp["output_var"] = 0.01

        # model parameters
        self.hyp["basis"] = "FullyConnected"  # or Custom, Legendre, Fourier
        self.hyp["final_layer"] = "FullyConnected"
        self.hyp[
            "model"
        ] = "BayesianRegression"  # if GP, ignore almost every other parameter
        self.hyp["length_scale"] = 1  # for GP
        self.hyp["rbf_multiplier"] = 0.1  # for GP
        self.hyp["activation"] = "ReLU"
        self.hyp["num_bases"] = 20
        self.hyp["layers"] = [1, 50, 1]
        self.hyp["output_activation"] = True
        self.hyp["bias"] = True
        self.hyp["rand_init_mean"] = 0.0
        self.hyp["rand_init_std"] = 1.0
        self.hyp["rand_init_seed"] = 0  # goto for multiple runs of the same parameters
        self.hyp["w_prior_var"] = 1.0
        self.hyp["loss"] = "MLE"
        self.hyp["k"] = 0.1  # relevant if using MAP Loss
        self.hyp["learning_rate"] = 1e-3
        self.hyp["optimizer_weight_decay_l2"] = 0.0
        self.hyp["total_epochs"] = 5000

        # sampling and other parameters
        self.hyp["train_print_freq"] = 1000
        self.hyp["posterior_prior_predictive_samples"] = 3000
        self.hyp["add_output_noise_prior_predictive_sampling"] = True
        self.hyp["add_output_noise_posterior_predictive_sampling"] = True
        self.hyp["num_points_linspace_visualize"] = 500


class VaryingBasesNLM(BaseConfig):
    def __init__(self):
        super().__init__()
        # experiment_name
        self.hyp["experiment_name"] = "VaryingBases"
        self.hyp["model"] = "BayesianRegression"
        self.hyp["basis"] = "FullyConnected"
        self.hyp["activation"] = MultipleRuns(["LeakyReLU", "ReLU", "Tanh"])
        self.hyp["num_bases"] = MultipleRuns([5, 10, 20, 40, 80])
        self.hyp["rand_init_seed"] = MultipleRuns([0, 1, 2, 3, 4])
        self.hyp["layers"] = [1, 50, 1]

class VaryingBasesGP(BaseConfig):
    def __init__(self):
        super().__init__()
        # experiment_name
        self.hyp["experiment_name"] = "VaryingBases"
        self.hyp["model"] = "GP"

class VaryingBasesFourier(BaseConfig):
    def __init__(self):
        super().__init__()
        # experiment_name
        self.hyp["experiment_name"] = "VaryingBases"
        self.hyp["model"] = "BayesianRegression"
        self.hyp["basis"] = "Fourier"
        self.hyp["num_bases"] = MultipleRuns([5, 10, 20, 40, 80])
        self.hyp["rand_init_seed"] = MultipleRuns([1, 2, 3, 4])

class VaryingBasesFourierScale(BaseConfig):
    def __init__(self):
        super().__init__()
        # experiment_name
        self.hyp["experiment_name"] = "VaryingBasesScale"
        self.hyp["model"] = "BayesianRegression"
        self.hyp["basis"] = "Fourier"
        self.hyp["num_bases"] = MultipleRuns([5, 10, 20, 40, 80])
        self.hyp["rand_init_seed"] = 5

class VaryingBasesOther(BaseConfig):
    def __init__(self):
        super().__init__()
        # experiment_name
        self.hyp["experiment_name"] = "VaryingBases"
        self.hyp["model"] = "BayesianRegression"
        self.hyp["basis"] = MultipleRuns(["Legendre", "Fourier", "RandomLinear"])
        self.hyp["num_bases"] = MultipleRuns([5, 10, 20, 40, 80])



# First Configuration
class GPConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        # experiment_name
        self.hyp["experiment_name"] = "GP"
        self.hyp["model"] = "GP"
        self.hyp["visualize_min_range"] = -1
        self.hyp["visualize_max_range"] = 1


# VarofVar Configuration
class VarofVar(BaseConfig):
    def __init__(self):
        super().__init__()
        # experiment_name
        self.hyp["experiment_name"] = "VarofVar"
        self.hyp["rand_init_seed"] = MultipleRuns([0, 1])
        self.hyp["total_epochs"] = MultipleModels([1, 100, 1000])
        self.hyp["train_dataset_size"] = 20
        self.hyp["num_bases"] = 20
        self.hyp["layers"] = [1, 40, self.hyp["num_bases"], 1]


# Fourier Configuration
class TestingFourier(BaseConfig):
    def __init__(self):
        super().__init__()
        # experiment_name
        self.hyp["experiment_name"] = "Fourier"
        self.hyp["basis"] = "Fourier"
        self.hyp["dataset_min_range"] = -3
        self.hyp["dataset_max_range"] = 3
        self.hyp["train_dataset_size"] = 10
        self.hyp["gap_min_range"] = -1
        self.hyp["gap_max_range"] = 1
        self.hyp["w_prior_var"] = 5.0
        self.hyp["num_bases"] = 10
        self.hyp["rand_init_seed"] = 0


# testing GPs
class TestingGP(BaseConfig):
    def __init__(self):
        super().__init__()
        # experiment_name
        self.hyp["experiment_name"] = "GP"
        self.hyp["model"] = MultipleRuns(["BayesianRegression", "GP"])
        self.hyp["dataset_min_range"] = -3
        self.hyp["dataset_max_range"] = 3
        self.hyp["train_dataset_size"] = 10
        self.hyp["gap_min_range"] = -1
        self.hyp["gap_max_range"] = 1
        self.hyp["w_prior_var"] = 1.0
        self.hyp["num_bases"] = 10
        self.hyp["rand_init_seed"] = 0


# comparing to GPs
class ComparingGP(BaseConfig):
    def __init__(self):
        super().__init__()
        # experiment_name
        self.hyp = {}

        # experiment parameters
        self.hyp["experiment_name"] = "ComparingGP"

        # dataset parameters
        self.hyp["dataset"] = "cubic"
        self.hyp["dataset_min_range"] = -1
        self.hyp["dataset_max_range"] = 1
        self.hyp["gap_min_range"] = -0.5
        self.hyp["gap_max_range"] = 0.5
        self.hyp["visualize_min_range"] = -1
        self.hyp["visualize_max_range"] = 1
        self.hyp["train_dataset_size"] = 100
        self.hyp["output_var"] = 0.01

        # model parameters
        self.hyp["basis"] = "OneBasisIsDataFourier"  # or Custom, Legendre, Sine+Cosine
        self.hyp["final_layer"] = "FullyConnected"
        self.hyp[
            "model"
        ] = "BayesianRegression"  # if GP, ignore almost every other parameter
        self.hyp["activation"] = "LeakyReLU"
        self.hyp["num_bases"] = MultipleRuns([1, 5, 10, 100, 200, 300])
        self.hyp["layers"] = [1, 50, self.hyp["num_bases"], 1]
        self.hyp["output_activation"] = True
        self.hyp["bias"] = True
        self.hyp["rand_init_mean"] = 0.0
        self.hyp["rand_init_std"] = 1.0
        self.hyp["rand_init_seed"] = 0
        self.hyp["w_prior_var"] = 1.0
        self.hyp["loss"] = "MLE"
        self.hyp["k"] = 0.1  # relevant if using MAP Loss
        self.hyp["learning_rate"] = 1e-3
        self.hyp["optimizer_weight_decay_l2"] = 0.0
        self.hyp["total_epochs"] = 5000

        # sampling and other parameters
        self.hyp["train_print_freq"] = 1000
        self.hyp["posterior_prior_predictive_samples"] = 3000
        self.hyp["add_output_noise_prior_predictive_sampling"] = True
        self.hyp["add_output_noise_posterior_predictive_sampling"] = True
        self.hyp["num_points_linspace_visualize"] = 500


class GeneralComparison(BaseConfig):
    def __init__(self):
        super().__init__()
        # experiment_name
        self.hyp = {}

        # experiment parameters
        self.hyp["experiment_name"] = "GeneralComparison"

        # dataset parameters
        self.hyp["dataset"] = "cubic"
        self.hyp["dataset_min_range"] = -1
        self.hyp["dataset_max_range"] = 1
        self.hyp["gap_min_range"] = -0.5
        self.hyp["gap_max_range"] = 0.5
        self.hyp["visualize_min_range"] = -1.5
        self.hyp["visualize_max_range"] = 1.5
        self.hyp["train_dataset_size"] = 100
        self.hyp["output_var"] = 0.01

        # model parameters
        self.hyp["basis"] = "FullyConnected"  # or Custom, Legendre, Sine+Cosine
        self.hyp["final_layer"] = "FullyConnected"
        self.hyp[
            "model"
        ] = MultipleModels(["BayesianRegression", "GP"])  # if GP, ignore almost every other parameter
        self.hyp["length_scale"] = 1
        self.hyp["rbf_multiplier"] = MultipleRuns([0.1])
        self.hyp["activation"] = "Tanh"
        self.hyp["num_bases"] = MultipleRuns([20])
        self.hyp["layers"] = [1, 5, 1] # second to last will be replaced by num_bases in the FullyConnected case
        self.hyp["output_activation"] = True
        self.hyp["bias"] = True
        self.hyp["rand_init_mean"] = 0.0
        self.hyp["rand_init_std"] = 1.0
        self.hyp["rand_init_seed"] = 0
        self.hyp["w_prior_var"] = 1.0
        self.hyp["loss"] = "MLE"
        self.hyp["k"] = 0.1  # relevant if using MAP Loss
        self.hyp["learning_rate"] = 1e-3
        self.hyp["optimizer_weight_decay_l2"] = 0.0
        self.hyp["total_epochs"] = 5000

        # sampling and other parameters
        self.hyp["train_print_freq"] = 1000
        self.hyp["posterior_prior_predictive_samples"] = 3000
        self.hyp["add_output_noise_prior_predictive_sampling"] = True
        self.hyp["add_output_noise_posterior_predictive_sampling"] = True
        self.hyp["num_points_linspace_visualize"] = 500



class TestingPosteriorContraction(BaseConfig):
    def __init__(self):
        super().__init__()
        # experiment_name
        self.hyp = {}

        # experiment parameters
        self.hyp["experiment_name"] = "TestingPosteriorContraction"

        # dataset parameters
        self.hyp["dataset"] = "cubic"
        self.hyp["dataset_min_range"] = -1
        self.hyp["dataset_max_range"] = 1
        self.hyp["gap_min_range"] = -0.5
        self.hyp["gap_max_range"] = 0.5
        self.hyp["visualize_min_range"] = -1.5
        self.hyp["visualize_max_range"] = 1.5
        self.hyp["train_dataset_size"] = 100
        self.hyp["output_var"] = 0.01

        # model parameters
        self.hyp["basis"] = "FullyConnected"  # or Custom, Legendre, Sine+Cosine
        self.hyp["final_layer"] = "FullyConnected"
        self.hyp[
            "model"
        ] = "BayesianRegression"  # if GP, ignore almost every other parameter
        self.hyp["length_scale"] = 1
        self.hyp["rbf_multiplier"] = MultipleRuns([0.1])
        self.hyp["activation"] = MultipleRuns(["Tanh", "LeakyReLU"])
        self.hyp["num_bases"] = MultipleRuns([5, 10, 20, 40])
        self.hyp["layers"] = [1, 5, 1] # second to last will be replaced by num_bases in the FullyConnected case
        self.hyp["output_activation"] = True
        self.hyp["bias"] = True
        self.hyp["rand_init_mean"] = 0.0
        self.hyp["rand_init_std"] = 1.0
        self.hyp["rand_init_seed"] = 0
        self.hyp["w_prior_var"] = 1.0
        self.hyp["loss"] = "MLE"
        self.hyp["k"] = 0.1  # relevant if using MAP Loss
        self.hyp["learning_rate"] = 1e-3
        self.hyp["optimizer_weight_decay_l2"] = 0.0
        self.hyp["total_epochs"] = 5000

        # sampling and other parameters
        self.hyp["train_print_freq"] = 1000
        self.hyp["posterior_prior_predictive_samples"] = 3000
        self.hyp["add_output_noise_prior_predictive_sampling"] = True
        self.hyp["add_output_noise_posterior_predictive_sampling"] = True
        self.hyp["num_points_linspace_visualize"] = 500

# Base Configuration - don't change!
class NewConfig(BaseConfig):
    def __init__(self):
        super().__init__()

        # experiment parameters
        self.hyp["experiment_name"] = "NewConfig"

        # dataset parameters
        self.hyp["dataset"] = "cubic"
        self.hyp["dataset_min_range"] = -1
        self.hyp["dataset_max_range"] = 1
        self.hyp["gap_min_range"] = -0.5
        self.hyp["gap_max_range"] = 0.5
        self.hyp["train_dataset_size"] = 100
        self.hyp["output_var"] = 0.01

        # model parameters
        self.hyp["basis"] = "FullyConnected"  # or Custom, Legendre, Sine+Cosine
        self.hyp["final_layer"] = "FullyConnected"
        self.hyp[
            "model"
        ] = MultipleRuns(["GP", "BayesianRegression"])  # if GP, ignore almost every other parameter
        self.hyp["length_scale"] = 1  # for GP
        self.hyp["rbf_multiplier"] = 0.1  # for GP
        self.hyp["activation"] = "ReLU"
        self.hyp["num_bases"] = 20
        self.hyp["layers"] = [1, 50, self.hyp["num_bases"], 1]
        self.hyp["output_activation"] = True
        self.hyp["bias"] = True
        self.hyp["rand_init_mean"] = 0.0
        self.hyp["rand_init_std"] = 1.0
        self.hyp["rand_init_seed"] = MultipleRuns(
            [0]
        )  # goto for multiple runs of the same parameters
        self.hyp["w_prior_var"] = 1.0
        self.hyp["loss"] = "MLE"
        self.hyp["k"] = 0.1  # relevant if using MAP Loss
        self.hyp["learning_rate"] = 1e-3
        self.hyp["optimizer_weight_decay_l2"] = 0.0
        self.hyp["total_epochs"] = 5000

        # sampling and other parameters
        self.hyp["train_print_freq"] = 1000
        self.hyp["posterior_prior_predictive_samples"] = 3000
        self.hyp["add_output_noise_prior_predictive_sampling"] = True
        self.hyp["add_output_noise_posterior_predictive_sampling"] = True
        self.hyp["num_points_linspace_visualize"] = 500
