import torch
import tqdm
from torch import nn
from torch.distributions import MultivariateNormal, kl

import tools.cauchy_dist
from tools import softplus
from models.gp import TransitionLayer


class FirstPos(torch.nn.Module):

    def __init__(self,
                 point_num, dim=2,
                 prior_mean=None,
                 prior_var=None,
                 init_loc=None,
                 init_covariance=None,
                 ):
        super().__init__()

        self.point_num = point_num
        self.dim = dim

        if prior_mean is None:
            prior_mean = torch.zeros(torch.Size([point_num, dim]))

        if prior_var is None:
            prior_var = torch.eye(dim).repeat(point_num, 1, 1)

        self.register_buffer("prior_mean", prior_mean)
        self.register_buffer("prior_var", prior_var)

        self.prior = [MultivariateNormal(self.prior_mean[i], scale_tril=self.prior_var[i]) for i in range(point_num)]

        if init_loc is None:
            init_loc = torch.randn(torch.Size([point_num, dim]))

        self.loc = torch.nn.Parameter(init_loc)

        if init_covariance is None:
            init_covariance = torch.eye(dim).repeat(point_num, 1, 1)

        self.covariance = torch.nn.Parameter(init_covariance)

    def forward(self):
        posterior = \
            [MultivariateNormal(
                self.loc[i],
                covariance_matrix=torch.tril(softplus.diagonal_softplus(self.covariance[i])) @ torch.tril(
                    softplus.diagonal_softplus(self.covariance[i])).T)
                for i in range(self.point_num)]

        return posterior

    def prior_loss(self, active_point):
        log_prior = torch.Tensor([0])

        for i in range(self.point_num):
            if active_point is None or active_point[i]:
                posterior_mu = MultivariateNormal(self.loc[i],
                                                  scale_tril=torch.tril(softplus.diagonal_softplus(self.covariance[i])))

                log_prior = log_prior + kl.kl_divergence(posterior_mu, self.prior[i])

        return log_prior


class DGPMotionEstimator(torch.nn.Module):

    def __init__(self,
                 time_len,
                 dim=2,
                 inducing_num=100,
                 sampling_num=10,
                 likelihood_noise=1.,
                 noise_prior=10,
                 beta=1.,
                 inducing_points=None,
                 learn_noise=True
                 ):
        super().__init__()

        self.time_len = time_len
        self.dim = dim

        # grid = torch.Tensor(np.linspace(0., 1., inducing_grid))
        if inducing_points is None:
            inducing_points = torch.randn(torch.Size([time_len - 1, inducing_num, 2]))

        self.transition = nn.ModuleList([TransitionLayer(inducing_points[t], beta=beta) for t in range(time_len - 1)])

        self.sampling_num = sampling_num

        if learn_noise:
            self.likelihood_noise = torch.nn.Parameter(torch.Tensor([likelihood_noise]))
        else:
            self.register_buffer("likelihood_noise", torch.Tensor([likelihood_noise]))

        self.noise_prior = noise_prior

    def expect_and_max(self, data, time, pi, pos_zero):
        with torch.no_grad():
            sample = self.sample(pos_zero)
            expected_likelihood = torch.exp(self.expected_likelihood(data, time, sample) + torch.log(pi))
            hidden = expected_likelihood.div(expected_likelihood.sum(dim=-1).unsqueeze(-1))
            return hidden, hidden.mean(dim=0)

    def expected_likelihood(self, data, time, sample):
        if data.shape[0] != time.shape[0]:
            print("data and time must be same shape 0")
        point_num = sample.shape[1]
        cov = torch.eye(self.dim).repeat(point_num, 1, 1) * softplus.softplus(self.likelihood_noise)
        likelihood_func = [
            MultivariateNormal(sample[t], scale_tril=torch.tril(cov.unsqueeze(1)))
            for t in range(self.time_len)]

        likelihood = torch.stack(
            [likelihood_func[t].log_prob(d).mean(axis=-1) for i, (d, t) in enumerate(zip(data, time))])

        return likelihood

    def sample(self, pos_zero):
        sample = torch.zeros([self.time_len, pos_zero.point_num, self.sampling_num, self.dim])
        first_sample = pos_zero.forward()
        sample[0] = torch.stack(
            [first_sample[i].rsample([self.sampling_num]) for i in range(pos_zero.point_num)])

        for i in range(1, self.time_len):
            trans_var = self.transition[i - 1](sample[i - 1].view((-1, self.dim)))
            if isinstance(trans_var, MultivariateNormal):
                scale = trans_var.variance.sqrt()
                trans_var = torch.distributions.Normal(loc=trans_var.mean, scale=scale)
            sample[i] = trans_var.rsample().view((pos_zero.point_num, self.sampling_num, self.dim)) + sample[i - 1]

        return sample

    def get_move(self, pos, time):
        variable = self.transition[time](pos.view((-1, self.dim)))
        return variable.mean, variable.variance

    def gp_loss(self, data, time, hidden, pos_zero, active_dim=None, batch_ratio=1):
        sample = self.sample(pos_zero)
        expected_likelihood = self.expected_likelihood(data, time, sample)
        weighted_likelihood = (hidden * expected_likelihood).sum()

        prior_loss = torch.Tensor([0.])

        prior_loss = prior_loss + pos_zero.prior_loss(active_dim)

        for trans in self.transition:
            prior_loss = prior_loss + trans.prior_loss()

        prior_loss = prior_loss - tools.cauchy_dist.cauchy_log_prob(softplus.softplus(self.likelihood_noise), self.noise_prior)

        return - weighted_likelihood * batch_ratio + prior_loss

