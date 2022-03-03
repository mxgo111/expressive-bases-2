from util import *


class FullyConnected(nn.Module):
    '''
    Fully connected neural network
    '''

    def __init__(self, hyp, layers=None, is_final_layer=False):
        super(FullyConnected, self).__init__()

        assert layers != None
        self.layers = layers
        self.output_activation = hyp["output_activation"]
        self.activation_module = getattr(nn, hyp["activation"])
        self.bias = hyp["bias"]
        self.rand_init_mean = hyp["rand_init_mean"]
        self.rand_init_std = hyp["rand_init_std"]
        self.rand_init_seed = hyp["rand_init_seed"]

        # adjust the random seed if using as final layer
        if is_final_layer:
            self.rand_init_seed += 1

        # seq stores layers
        seq = OrderedDict()
        seq['layer_0'] = nn.Linear(self.layers[0], self.layers[1], bias=self.bias)

        idx = 0
        for i in range(len(self.layers[1:-1])):
            seq['layer_{}_activation'.format(i)] = self.activation_module()

            idx = i + 1
            seq['layer_{}'.format(idx)] = nn.Linear(
                self.layers[idx], self.layers[idx + 1], bias=self.bias,
            )

        if self.output_activation:
            seq['layer_{}_activation'.format(idx + 1)] = self.activation_module()

        self.network = nn.Sequential(seq)

    def forward(self, x):
        return self.network(x)

    def get_weights(self):
        return torch.cat([p.view(-1) for p in self.parameters()])

    def rand_init(self):
        random_seed = self.rand_init_seed

        # set random seed for reproducibility
        torch.manual_seed(random_seed)
        random.seed(random_seed)
        np.random.seed(random_seed)

        def init_weights(l):
            if isinstance(l, nn.Linear):
                torch.nn.init.normal_(l.weight, mean=self.rand_init_mean, std=self.rand_init_std)
                if self.bias:
                    torch.nn.init.normal_(l.bias, mean=self.rand_init_mean, std=self.rand_init_std)

        self.network.apply(init_weights)


class BayesianRegression(nn.Module):
    def __init__(self, hyp):
        super(BayesianRegression, self).__init__()

        self.weights_var = hyp["w_prior_var"]
        self.output_var = hyp["output_var"]

        self.posterior = None

    def data_to_features(self, x):
        '''
        Concatenate features x with a column of 1s (for bias)
        '''

        ones = torch.ones(x.shape[0], 1)
        if cuda_available():
            ones = ones.cuda()

        return torch.cat([x, ones], -1)

    def get_posterior(self):
        return self.posterior

    def bayesian_linear_regression_posterior_1d(self, X, y):
        # See this link for derivation: http://cs229.stanford.edu/section/cs229-gaussian_processes.pdf
        # prior is distributed N(0, weights_var^2)
        # output y is distributed N(wx, output_var^2)
        assert(len(X.shape) == 2)
        assert(len(y.shape) == 2)
        assert(y.shape[-1] == 1)

        eye = torch.eye(X.shape[-1])
        if cuda_available():
            eye = eye.cuda()

        posterior_precision = eye / self.weights_var + torch.mm(X.t(), X) / self.output_var
        posterior_cov = torch.pinverse(posterior_precision)
        posterior_mu = torch.mm(posterior_cov, torch.mm(X.t(), y)).squeeze() / self.output_var

        return dists.MultivariateNormal(posterior_mu, precision_matrix=posterior_precision), posterior_mu

    def infer_posterior(self, x, y):
        '''
        Infers posterior and stores it within the class instance
        '''

        phi = self.data_to_features(x)
        assert(len(phi.shape) == 2)

        self.posterior, posterior_mean = self.bayesian_linear_regression_posterior_1d(phi, y)

        return self.posterior, posterior_mean

    def sample_posterior_predictive(self, x, num_samples, add_noise=True):
        assert(self.posterior is not None)

        phi = self.data_to_features(x)

        weights = self.posterior.rsample(torch.Size([num_samples]))
        assert(weights.shape == (num_samples, phi.shape[-1]))

        r = torch.mm(phi, weights.t())
        assert(r.shape == torch.Size([x.shape[0], num_samples]))

        if add_noise:
            return add_output_noise(r, self.output_var)
        else:
            return r

    def sample_prior_predictive(self, x, num_samples, add_noise=True):
        phi = self.data_to_features(x)

        weights = dists.Normal(0.0, math.sqrt(self.weights_var)).sample((num_samples, phi.shape[-1]))
        assert(len(weights.shape) == 2)

        r = torch.mm(phi, weights.t())
        assert(r.shape == torch.Size([x.shape[0], num_samples]))

        if add_noise:
            return add_output_noise(r, self.output_var)
        return r


class NLM(nn.Module):
    def __init__(self, hyp):
        super(NLM, self).__init__()

        self.model = BayesianRegression(hyp)
        self.basis = FullyConnected(hyp, layers=hyp["layers"][:-1])
        self.final_layer = FullyConnected(hyp, layers=hyp["layers"][-2:], is_final_layer=True)

        # randomly initialize weights
        self.w_prior_var = hyp["w_prior_var"]
        self.basis.rand_init()
        self.final_layer.rand_init()

        # training parameters
        self.loss_fn_name = hyp["loss"]
        if self.loss_fn_name not in ["MLE", "MAP"]:
            print("loss not defined")

        self.lr = hyp["learning_rate"]
        self.l2 = hyp["optimizer_weight_decay_l2"]
        self.k = hyp["k"]
        self.print_freq = hyp["train_print_freq"]
        self.total_epochs = hyp["total_epochs"]

    # def train(self, x_train, y_train):
    #     # optimize loss to learn network
    #     def mle_loss():
    #         y_pred = self.final_layer(self.basis(x_train))
    #         loss = torch.mean(torch.sum(torch.pow(self.final_layer(self.basis(x_train)) - y_train, 2.0), -1))
    #         return loss, (self.basis, self.final_layer)
    #
    #     (self.basis, self.final_layer), loss = self.train_objective(
    #         list(self.basis.parameters()) + list(self.final_layer.parameters()),
    #         mle_loss,
    #         lr=self.lr,
    #         print_freq=1000
    #     )
    #
    # def train_objective(self, params, loss_fn, lr=0.01, l2=0.0, epochs=4000, print_freq=100):
    #     '''
    #     Optimizes 'loss_fn' with respect to 'params'
    #     'loss_fn' must take no arguments, and must return a tuple of two:
    #     the value of the loss, and the model.
    #     '''
    #
    #     best_model = None
    #     min_loss = float('inf')
    #
    #     optimizer = optim.Adam(params, lr=lr, weight_decay=l2)
    #     try:
    #         for epoch in range(epochs):
    #             optimizer.zero_grad()
    #
    #             # save loss and model if loss is the smallest observed so far
    #             loss, model = loss_fn()
    #             if loss.item() < min_loss:
    #                 min_loss = loss.item()
    #                 best_model = copy.deepcopy(model)
    #
    #             loss.backward()
    #             optimizer.step()
    #
    #             if epoch % print_freq == 0:
    #                 print('Epoch {}: loss = {}'.format(epoch, loss.item()))
    #     except KeyboardInterrupt:
    #         print('Interrupted...')
    #
    #     print('Final Loss = {}'.format(min_loss))
    #     import sys; sys.exit()
    #     return best_model, min_loss


    def train(self, x_train, y_train, epochs):

        '''
        Optimizes 'loss_fn' with respect to 'params'
        'loss_fn' return a tuple of two:
        the value of the loss, and the model.

        k is the regularization term, default 0
        '''
        loss_fn_name = self.loss_fn_name
        params = list(self.basis.parameters()) + list(self.final_layer.parameters())

        def mle_loss():
            y_pred = self.final_layer(self.basis(x_train))
            loss = torch.mean(torch.sum(torch.pow(self.final_layer(self.basis(x_train)) - y_train, 2.0), -1))
            return loss, (self.basis, self.final_layer)

        def map_loss(k):
            y_pred = self.final_layer(self.basis(x_train))
            loss = torch.mean(torch.sum(torch.pow(self.final_layer(self.basis(x_train)) - y_train, 2.0), -1)) + k*(torch.linalg.norm(torch.cat((self.basis.get_weights(),self.final_layer.get_weights())), 2))**2
            return loss, (self.basis, self.final_layer)

        best_model = None
        min_loss = float('inf')

        optimizer = optim.Adam(params, lr=self.lr, weight_decay=self.l2)
        try:
            for epoch in range(epochs):
                optimizer.zero_grad()

                # save loss and model if loss is the smallest observed so far
                if self.loss_fn_name == "MLE":
                    loss, model = mle_loss()
                elif self.loss_fn_name == "MAP":
                    loss, model = map_loss(self.k)

                if loss.item() < min_loss:
                    min_loss = loss.item()
                    best_model = copy.deepcopy(model)

                loss.backward()
                optimizer.step()

                if epoch % self.print_freq == 0:
                    print('Epoch {}: loss = {}'.format(epoch, loss.item()))
        except KeyboardInterrupt:
            print('Interrupted...')

        print('Final Loss = {}'.format(min_loss))

        (self.basis, self.final_layer), self.min_loss = best_model, min_loss
        self.model.infer_posterior(self.basis(x_train), y_train)

    def visualize_posterior_predictive(self):
        # asdfasdfd
        pass

    def visualize_prior_predictive(self):
        pass
