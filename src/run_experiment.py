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
    all_epochs = hyp["total_epochs"].epochs
    training_epochs = [all_epochs[0]] + list(np.array(all_epochs[1:]) - np.array(all_epochs[:-1]))

    for i in range(1, len(training_epochs)):
        assert training_epochs[i] > 0

    # check for 0 training (just random init model) at the beginning
    if training_epochs[0] == 0:
        analyze(hyp, model, x_train, y_train, trained_epochs=0)
        training_epochs = training_epochs[1:]
        all_epochs = all_epochs[1:]

    for i, epochs in enumerate(training_epochs):
        model.train(x_train, y_train, epochs=epochs)
        analyze(hyp, model, x_train, y_train, trained_epochs=all_epochs[i])


if __name__ == "__main__":
    # testing multiple runs
    run_multiple_models(TestingEpoch0())
