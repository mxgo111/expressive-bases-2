from util import *
torch.set_default_tensor_type(torch.DoubleTensor)

def cubic(x):
    return torch.pow(x, 3.0)

def sine(x):
    return torch.sin(x)

def xsinx(x):
    return torch.multiply(x, torch.sin(x))

def generate_data(hyp, random_seed=0, test=False):
    '''
    Generates data according to hyperparameters specified
    '''
    # dataset parameters
    N = hyp["train_dataset_size"]
    var = hyp["output_var"]

    if not test:
        # set random seed for reproducibility
        torch.manual_seed(random_seed)
        random.seed(random_seed)
        np.random.seed(random_seed)

        x = torch.cat(
                (dists.Uniform(hyp["dataset_min_range"], hyp["gap_min_range"]).sample((N//2, 1)),
                 dists.Uniform(hyp["gap_max_range"], hyp["dataset_max_range"]).sample((N//2, 1)))
            )

        if hyp["dataset"] == "cubic":
            y = add_output_noise(cubic(x), var)
        elif hyp["dataset"] == "sine":
            y = add_output_noise(sine(x), var)
        elif hyp["dataset"] == "xsinx":
            y = add_output_noise(xsinx(x), var)

    else:
        # TODO: how do we wanna do test data?
        pass

    return x, y
