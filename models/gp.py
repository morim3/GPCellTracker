import gpytorch
import torch


class TransitionLayer(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points=None, dims=2, beta=1.):

        self.beta = beta

        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            num_inducing_points=inducing_points.shape[0],
            batch_shape=torch.Size([dims]),
            mean_init_std=1e-3
        )
        variational_distribution.chol_variational_covar.data.copy_(
            torch.eye(inducing_points.shape[0], inducing_points.shape[0]).repeat(dims, 1, 1))

        variational_strategy = gpytorch.variational.IndependentMultitaskVariationalStrategy(
            gpytorch.variational.VariationalStrategy(
                self, inducing_points, variational_distribution, learn_inducing_locations=True
            ),
            num_tasks=dims
        )

        # Define models
        super().__init__(variational_strategy=variational_strategy)

        # lengthscale_prior = gpytorch.priors.GammaPrior(3.0, 0.01)
        # outputscale_prior = gpytorch.priors.GammaPrior(3.0, 0.01)

        # Define mean and kernel
        self.mean_module = gpytorch.means.ZeroMean(input_dims=dims, batch_shape=torch.Size([dims]))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(batch_shape=torch.Size([dims]),
                                       #lengthscale_prior=lengthscale_prior
                                       ),
            batch_shape=torch.Size([dims]),
            # outputscale_prior=outputscale_prior
        )
        self.covar_module.base_kernel.lengthscale = torch.Tensor([[1.]])
        self.covar_module.outputscale = torch.tensor(1.)

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)

    def prior_loss(self):

        kl_divergence = self.variational_strategy.kl_divergence()

        added_loss = torch.Tensor([0])
        for added_loss_term in self.added_loss_terms():
            added_loss += added_loss_term.loss()

        log_prior = torch.Tensor([0])
        for name, module, prior, closure, _ in self.named_priors():
            log_prior += prior.log_prob(closure(module)).sum()

        return self.beta * kl_divergence - log_prior + added_loss