import torch.nn.functional as F
from torch import nn
import torch


class ConditionalVAE(nn.Module):
    def __init__(self, args):
        super(ConditionalVAE, self).__init__()
        self.num_features = args.num_features
        self.num_hidden_units = args.num_hidden_units
        self.latent_dim = args.latent_dim
        # Encoder-Decoder
        self.encoder = self._encoder()
        self.mean_branch = self._encoder_block(self.num_hidden_units, self.latent_dim, activate=False)
        self.var_branch = self._encoder_block(self.num_hidden_units, self.latent_dim, activate=False)
        self.decoder = self._decoder()

    def _encoder(self):
        layers = []
        layers.append(self._encoder_block(2 * self.num_features, self.num_hidden_units))
        layers.append(self._encoder_block(self.num_hidden_units, self.num_hidden_units))
        return nn.Sequential(*layers)

    def _encoder_block(self, in_channel: int, out_channel: int, downsample: int = None, activate=True):
        layers = []
        #layers.append(nn.Conv1d(in_channel, out_channel, 3, stride=1, padding='same', bias=False))
        layers.append(nn.Conv1d(in_channel, out_channel, 3, stride=1, padding=1, bias=False))
        layers.append(nn.BatchNorm1d(out_channel))
        if activate:
            layers.append(nn.ReLU(inplace=False))
        if downsample is not None:
            layers.append(nn.MaxPool1d(downsample, stride=downsample))
        return nn.Sequential(*layers)

    def _decoder(self):
        layers = []
        layers.append(self._decoder_block(self.latent_dim + self.num_features, self.num_hidden_units))
        layers.append(self._decoder_block(self.num_hidden_units, self.num_features, activate=False))
        return nn.Sequential(*layers)

    def _decoder_block(self, in_channel: int, out_channel: int, upsample: int = None, activate=True):
        layers = []
        #layers.append(nn.Conv1d(in_channel, out_channel, 3, stride=1, padding='same', bias=False))
        layers.append(nn.Conv1d(in_channel, out_channel, 3, stride=1, padding=1 , bias=False))
        layers.append(nn.BatchNorm1d(out_channel))
        if activate:
            layers.append(nn.ReLU(inplace=False))
        if upsample is not None:
            layers.append(nn.Upsample(size=upsample, mode='linear', align_corners=False))
        return nn.Sequential(*layers)

    def encode(self, x, c):
        h = torch.cat([x, c], dim=1)
        h = self.encoder(h)
        mean = self.mean_branch(h)
        logvar = self.var_branch(h)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        z = mean + std * torch.randn_like(std)
        return z

    def decode(self, z, c):
        h = torch.cat([z, c], dim=1)
        x_ = self.decoder(h)
        return x_

    def forward(self, x, c):
        mean, logvar = self.encode(x, c)
        z = self.reparameterize(mean, logvar)
        x_ = self.decode(z, c)
        return x_, mean, logvar


def VAE_loss(x_, x, mean, logvar, KL_weight=1.0):
    reconstruction_loss = F.l1_loss(x_, x)
    KL_divergence = -0.5 * (1 + logvar - mean.pow(2) - logvar.exp()).sum()
    return reconstruction_loss + KL_weight * KL_divergence
