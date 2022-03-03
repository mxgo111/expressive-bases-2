from imports import *

torch.set_default_tensor_type(torch.DoubleTensor)


def cuda_available():
    return torch.cuda.is_available()


def ftens_cuda(*args, **kwargs):
    if cuda_available():
        t = torch.cuda.DoubleTensor(*args, **kwargs)
    else:
        t = torch.DoubleTensor(*args, **kwargs)

    return t


def to_np(v):
    if type(v) == float:
        return v

    if v.is_cuda:
        return v.detach().cpu().numpy()
    return v.detach().numpy()


def area(upper, lower, h):
    """
    Calculate the area between f1 and f2 over the interval [x1, x2] using n points in finite estimation
    """
    return np.sum(np.abs(upper - lower)) * h


def get_coverage_bounds(posterior_pred_samples, percentile):
    '''
    Assumes N x samples
    '''
    assert(not (percentile < 0. or percentile > 100.))

    lower_percentile = (100.0 - percentile) / 2.0
    upper_percentile = 100.0 - lower_percentile

    upper_bounds = np.percentile(posterior_pred_samples, upper_percentile, axis=-1)
    lower_bounds = np.percentile(posterior_pred_samples, lower_percentile, axis=-1)

    return lower_bounds, upper_bounds


def get_uncertainty_in_gap(model, basis, x_train, y_train, n_points=1000, picp=95.0):
    """
    Estimates area in uncertainty region of the gap
    """
    assert(len(x_train.shape) == 2 and x_train.shape[-1] == 1)
    assert(len(y_train.shape) == 2 and y_train.shape[-1] == 1)

    # make sure x_train is sorted in ascending order
    x_train_sorted = np.sort(to_np(x_train.squeeze()))
    assert(np.all(x_train_sorted[:-1] <= x_train_sorted[1:]))

    # find gap
    N = len(x_train)
    gap = np.linspace(x_train_sorted.squeeze()[N//2-1], x_train_sorted.squeeze()[N//2], n_points)
    h = gap[1] - gap[0]
    gap = ftens_cuda(gap).unsqueeze(-1)

    # sample from inside gap
    y_pred = model.sample_posterior_predictive(basis(gap), n_points)
    lower, upper = get_coverage_bounds(to_np(y_pred), picp)

    area = area(upper, lower, h)

    return area


def eff_dim(evals, z):
    """
    Computes effective dimensionality of matrix
    """
    assert z > 0
    return np.sum(np.divide(evals, evals+z))


def compute_eff_dim(basis_vals, z=1, visual=False):
    """
    Computes effective dimensionality of basis functions
    """
    # each column is a basis function
    basis_vals_np = basis_vals.detach().cpu().numpy()
    basis_vals_df = pd.DataFrame(basis_vals_np)

    # calculate correlations
    corr = basis_vals_df.corr()

    # drop irrelevant rows/columns
    corr.dropna(axis=0, how='all', inplace=True)
    corr.dropna(axis=1, how='all', inplace=True)

    # eigenvals
    evals, evecs = np.linalg.eig(corr)

    if visual:
        plt.figure(figsize=(10,10))
        # plot the heatmap
        sns.heatmap(corr,
                xticklabels=corr.columns,
                yticklabels=corr.columns)

        # Scree plot
        plt.figure(figsize=(10, 5))
        plt.scatter(np.arange(len(evals)), evals)
        plt.plot(evals)
        plt.title("Eigenvalues of correlation matrix")
        plt.show()

    return eff_dim(evals, z)
