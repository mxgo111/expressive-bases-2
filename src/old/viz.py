from util import *


def plot_1d_posterior_predictive(x_train, y_train, x_viz, y_pred, savefig=None):
    assert(len(x_train.shape) == 2 and x_train.shape[-1] == 1)
    assert(len(y_train.shape) == 2 and y_train.shape[-1] == 1)
    assert(len(x_viz.shape) == 2 and x_viz.shape[-1] == 1)
    assert(len(y_pred.shape) == 2 and y_pred.shape[0] == x_viz.shape[0])

    # make sure x_viz is sorted in ascending order
    x_viz = to_np(x_viz.squeeze())
    assert(np.all(x_viz[:-1] <= x_viz[1:]))

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    # plot predictive intervals
    for picp, alpha in zip([50.0, 68.0, 95.0], [0.4, 0.3, 0.2]):
        lower, upper = get_coverage_bounds(to_np(y_pred), picp)

        ax.fill_between(
            x_viz, lower, upper, label='{}%-PICP'.format(picp), color='steelblue', alpha=alpha,
        )

    # plot predictive mean
    pred_mean = to_np(torch.mean(y_pred, -1))
    ax.plot(x_viz, pred_mean, color='blue', lw=3, label='Predictive Mean')

    # plot training data
    ax.scatter(x_train, y_train, color='red', s=10.0, zorder=10, label='Training Data')

    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_title('Posterior Predictive')
    ax.legend()

    if savefig != None:
        plt.savefig(savefig)

def plot_basis_functions_1d(num_final_layers, x_vals, basis, x_train, posterior_mean, numcols=12, savefig=None):
    basis_vals = basis(torch.tensor(x_vals.reshape(-1, 1)))

    # sort functions
    def argsort(seq):
        # https://stackoverflow.com/questions/3382352/equivalent-of-numpy-argsort-in-basic-python
        return sorted(range(len(seq)), key=lambda x: abs(max(seq[x]) - min(seq[x])))

    functions = [basis_vals[:, i].detach().cpu().numpy() for i in range(num_final_layers)]
    argsorted_basis = argsort(functions)

    # training data
    x_train_np = x_train.detach().cpu().numpy().squeeze()
    basis_train_np = basis(x_train).detach().cpu().numpy()

    fig, axs = plt.subplots(num_final_layers//numcols + 1, numcols, figsize=(40, 15))
    for j in range(num_final_layers):
        i = argsorted_basis[j]
        row, col = j//numcols, j % numcols
        axs[row,col].plot(x_vals, functions[i])
        axs[row,col].scatter(x_train_np, basis_train_np[:,i], c="red") # scatterplot training data
        axs[row,col].set_title(f"w_posterior_mean={np.round(posterior_mean.detach().cpu().numpy()[i], 3)}")
    plt.tight_layout()

    if savefig != None:
        plt.savefig(savefig)

    return basis_vals
