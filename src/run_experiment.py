from util import *
from configurations import *
from generate_data import generate_data
from model import FullyConnected, BayesianRegression, NLM
from analyze import analyze

def run_multiple_models(config_object):
    """
    Given a configuration with MultipleRuns classes, run multiple runs of each of those classes
    """

    config = config_object.hyp

    # create a list of all possible hyperparameters from MultipleRuns objects
    all_hyps = [{}]
    for key in config:
        if isinstance(config[key], MultipleRuns):
            og_all_hyps = copy.deepcopy(all_hyps)
            all_hyps = []
            for instance in config[key].hyps:
                hyps_with_instance = copy.deepcopy(og_all_hyps)
                # set specific instance within MultipleRuns to be hyperparameter
                for hyp in hyps_with_instance:
                    hyp[key] = instance
                all_hyps += hyps_with_instance
        else:
            for hyp in all_hyps:
                hyp[key] = config[key]

    print(f"Number of sets of hyperparameters: {len(all_hyps)}")

    for hyp in all_hyps:
        run_single_model(hyp)


def run_single_model(hyp):
    """
    train model
    analyze results (put into dataframe)
    """
    x_train, y_train = generate_data(hyp)
    model = NLM(hyp)

    # output several models at different levels of training
    all_models = []
    all_epochs = hyp["total_epochs"].epochs
    training_epochs = [all_epochs[0]] + list(np.array(all_epochs[1:]) - np.array(all_epochs[:-1]))

    for epochs in training_epochs:
        model.train(x_train, y_train, epochs=epochs)
        # model.train(x_train, y_train)
        sys.exit()
        all_models.append(copy.deepcopy(model))

    for (model, trained_epochs) in zip(all_models, all_epochs):
        analyze(hyp, model, x_train, y_train, trained_epochs=trained_epochs)


if __name__ == "__main__":
    # testing multiple runs
    firstconfig = FirstConfig()
    run_multiple_models(firstconfig)
