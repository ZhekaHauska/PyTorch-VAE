import torch
import numpy as np
from torchvae import BaseVAE
from torch import nn
from torch.nn import functional as F
from .types_ import *


class CategoricalVAE(BaseVAE):

    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 categorical_dim: int = 40,
                 hidden_dims: List = None,
                 temperature: float = 0.5,
                 anneal_rate: float = 3e-5,
                 anneal_interval: int = 100,
                 alpha: float = 30.,
                 beta: float = 1.,
                 trace_decay: float = 0.0,
                 gamma: float = 1.0,
                 **kwargs) -> None:
        super(CategoricalVAE, self).__init__()

        self.latent_dim = latent_dim
        self.categorical_dim = categorical_dim
        self.temp = 1.0
        self.min_temp = temperature
        self.anneal_rate = anneal_rate
        self.anneal_interval = anneal_interval
        self.alpha = alpha
        self.beta = beta
        self.trace_decay = trace_decay
        self.gamma = gamma

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_z = nn.Linear(hidden_dims[-1]*4, self.latent_dim * self.categorical_dim)

        # Build Decoder
        modules = []

        self.linear_decoder = nn.Linear(self.latent_dim * self.categorical_dim, 1)
        self.decoder_input = nn.Linear(self.latent_dim * self.categorical_dim, hidden_dims[-1] * 4)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels= 3,
                                      kernel_size= 3, padding= 1),
                            nn.Tanh())
        self.sampling_dist = torch.distributions.OneHotCategorical(
            1. / categorical_dim * torch.ones((self.categorical_dim, 1))
        )
        self.register_buffer('latent_trace', None)

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [B x C x H x W]
        :return: (Tensor) Latent code [B x D x Q]
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        z = self.fc_z(result)
        z = z.view(-1, self.latent_dim, self.categorical_dim)
        return [z]

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D x Q]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, z: Tensor, eps: float = 1e-7) -> Tensor:
        """
        Gumbel-softmax trick to sample from Categorical Distribution
        :param z: (Tensor) Latent Codes [B x D x Q]
        :return: (Tensor) [B x D]
        """
        # Sample from Gumbel
        u = torch.rand_like(z)
        g = - torch.log(- torch.log(u + eps) + eps)

        # Gumbel-Softmax sample
        s = F.softmax((z + g) / self.temp, dim=-1)
        s = s.view(-1, self.latent_dim * self.categorical_dim)
        return s

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        q = self.encode(input)[0]
        z = self.reparameterize(q)
        return [self.decode(z), input, q, self.linear_decoder(z),
                kwargs.get('labels'), z, kwargs.get('reset_flags')]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        recons = args[0]
        input = args[1]
        q = args[2]
        pred_reward = args[3]
        real_reward = args[4]
        z = args[5]
        reset_flags = args[6]

        q_p = F.softmax(q, dim=-1)

        kld_weight = kwargs['M_N']
        batch_idx = kwargs['batch_idx']

        if batch_idx % self.anneal_interval == 0 and self.training:
            self.temp = np.maximum(self.temp * np.exp(-self.anneal_rate * batch_idx),
                                   self.min_temp)

        recons_loss = F.mse_loss(recons, input, reduction='mean')

        eps = 1e-7
        h1 = q_p * torch.log(q_p + eps)
        h2 = q_p * np.log(1. / self.categorical_dim + eps)
        kld_loss = torch.mean(torch.sum(h1 - h2, dim=(1, 2)), dim=0)

        rew_loss = F.mse_loss(pred_reward, real_reward, reduction='mean')

        trace_loss = recons_loss.new_tensor(0.0)

        if self.trace_decay > 0 and self.training and z is not None:
            if reset_flags is None:
                reset_flags = torch.zeros(z.shape[0], device=z.device)

            if self.latent_trace is None or self.latent_trace.shape[0] != z.shape[0]:
                self.register_buffer('latent_trace',
                    torch.zeros(z.shape[0], self.latent_dim * self.categorical_dim,
                                device=z.device))

            trace_target = self.latent_trace.detach().clone()
            trace_loss = F.mse_loss(z, trace_target)

            with torch.no_grad():
                reset_mask = (reset_flags > 0).unsqueeze(1)
                z_detached = z.detach()
                new_trace = torch.where(
                    reset_mask,
                    z_detached,
                    self.trace_decay * self.latent_trace + (1 - self.trace_decay) * z_detached
                )
                self.latent_trace.copy_(new_trace)

        loss = self.alpha * recons_loss + kld_weight * kld_loss \
            + self.beta * rew_loss + self.gamma * trace_loss
        return {'loss': loss, 'Reconstruction_Loss': recons_loss,
                'KLD': -kld_loss, 'reward_loss': rew_loss, 'Trace_Loss': trace_loss}

    def sample(self,
               num_samples: int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        # [S x D x Q]

        M = num_samples * self.latent_dim
        np_y = np.zeros((M, self.categorical_dim), dtype=np.float32)
        np_y[range(M), np.random.choice(self.categorical_dim, M)] = 1
        np_y = np.reshape(np_y, [M // self.latent_dim, self.latent_dim, self.categorical_dim])
        z = torch.from_numpy(np_y)

        # z = self.sampling_dist.sample((num_samples * self.latent_dim, ))
        z = z.view(num_samples, self.latent_dim * self.categorical_dim).to(current_device)
        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]