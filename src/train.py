from util import *
from generate_data import *
torch.set_default_tensor_type(torch.DoubleTensor)


def train_objective(params, loss_fn_name, loss_fn, lr=0.01, l2=0.0, epochs=4000, print_freq=100, k=0):

    '''
    Optimizes 'loss_fn' with respect to 'params'
    'loss_fn' return a tuple of two:
    the value of the loss, and the model.

    k is the regularization term, default 0
    '''

    best_model = None
    min_loss = float('inf')


    optimizer = optim.Adam(params, lr=lr, weight_decay=l2)
    try:
        for epoch in range(epochs):
            optimizer.zero_grad()

            # save loss and model if loss is the smallest observed so far
            if loss_fn_name == "MLE":
                loss, model = loss_fn()
            elif loss_fn_name == "MAP":
                loss, model = loss_fn(k)

            if loss.item() < min_loss:
                min_loss = loss.item()
                best_model = copy.deepcopy(model)

            loss.backward()
            optimizer.step()

            if epoch % print_freq == 0:
                print('Epoch {}: loss = {}'.format(epoch, loss.item()))
    except KeyboardInterrupt:
        print('Interrupted...')

    print('Final Loss = {}'.format(min_loss))
    return best_model, min_loss



def train(hyp, model=None, basis=None, data=None):
    """
    Train a given model on data
    """

    # TODO: currently assuming everything are constantly None lol

    if not model:
        if hyp["model"] == "BayesianRegression":
            w_prior_var = hyp["w_prior_var"]
            output_var = hyp["output_var"]
            model = BayesianRegression(w_prior_var, output_var)

    if not data:
        x_train, y_train = generate_data(hyp)
    else:
        x_train, y_train = data

    if not basis:
        layers = hyp["layers"]
        ACTIVATION = hyp["activation"]
        output_activation = hyp["output_activation"]
        w_prior_var = hyp["w_prior_var"]
        if self.hyp["basis"] == "FullyConnected":
            basis = FullyConnected(layers[:-1], activation_module=ACTIVATION, output_activation=output_activation)
            basis.rand_init(math.sqrt(w_prior_var))

    if not final_layer:
        layers = hyp["layers"]
        ACTIVATION = hyp["activation"]
        w_prior_var = hyp["w_prior_var"]
        final_layer = FullyConnected(layers[-2:], activation_module=ACTIVATION)
        final_layer.rand_init(math.sqrt(w_prior_var))

    # define losses
    def mle_loss():
        y_pred = final_layer(basis(x_train))
        loss = torch.mean(torch.sum(torch.pow(final_layer(basis(x_train)) - y_train, 2.0), -1))
        return loss, (basis, final_layer)

    def map_loss(k):
        y_pred = final_layer(basis(x_train))
        loss = torch.mean(torch.sum(torch.pow(final_layer(basis(x_train)) - y_train, 2.0), -1)) + k*(torch.linalg.norm(torch.cat((basis.get_weights(),final_layer.get_weights())), 2))**2
        return loss, (basis, final_layer)

    if hyp["loss"] == "MLE":
        loss_fn = mle_loss
    elif hyp["loss"] == "MAP":
        loss_fn = map_loss
    else:
        raise NameError("loss not defined")

    if not hyp["save_multiple_models"]:
        # optimize loss to learn network
        (basis, final_layer), loss = train_objective(
            list(basis.parameters()) + list(final_layer.parameters()),
            loss_fn_name = hyp["loss"],
            loss_fn = loss_fn,
            lr=hyp["learning_rate"],
            print_freq=hyp["train_print_freq"],
            epochs = hyp["total_epochs"],
            k=hyp["k"]
        )
    else:
        pass

    # NEED TO MOVE THE BELOW STUFF
    
    # infer posterior over the last layer weights given the basis
    model.infer_posterior(basis(x_train), y_train)

    # sample from posterior predictive
    x_viz = ftens_cuda(np.linspace(-2.0, 2.0, 500)).unsqueeze(-1)
    y_pred = model.sample_posterior_predictive(basis(x_viz), 500)

    # visualize posterior predictive
    plot_1d_posterior_predictive(x_train, y_train, x_viz, y_pred)
