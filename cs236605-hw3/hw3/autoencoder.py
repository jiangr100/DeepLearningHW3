import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderCNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        modules = []

        # TODO: Implement a CNN. Save the layers in the modules list.
        # The input shape is an image batch: (N, in_channels, H_in, W_in).
        # The output shape should be (N, out_channels, H_out, W_out).
        # You can assume H_in, W_in >= 64.
        # Architecture is up to you, but you should use at least 3 Conv layers.
        # You can use any Conv layer parameters, use pooling or only strides,
        # use any activation functions, use BN or Dropout, etc.
        # ====== YOUR CODE: ======
        
        prev = in_channels
        stride = 1
        
        for i in range(3):
            stride = 1
            for j in range(2):
                conv = nn.Conv2d(prev, out_channels, kernel_size=3, padding=1, stride=stride)
                stride = 2
                batchnorm = nn.BatchNorm2d(out_channels)
                relu = nn.ReLU()
                dropout = nn.Dropout2d(0.5)
                
                prev = out_channels
                modules.extend([conv, batchnorm, relu, dropout])
        
        # ========================
        self.cnn = nn.Sequential(*modules)

    def forward(self, x):
        return self.cnn(x)


class DecoderCNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        modules = []

        # TODO: Implement the "mirror" CNN of the encoder.
        # For example, instead of Conv layers use transposed convolutions,
        # instead of pooling do unpooling (if relevant) and so on.
        # You should have the same number of layers as in the Encoder,
        # and they should produce the same volumes, just in reverse order.
        # Output should be a batch of images, with same dimensions as the
        # inputs to the Encoder were.
        # ====== YOUR CODE: ======
        ne = in_channels
        
        for i in range(3):
            stride = 2
            out_padding = 1
            for j in range(2):
                convTrans = nn.ConvTranspose2d(in_channels, ne, kernel_size=3, stride=stride, padding=1, output_padding=out_padding)
                stride = 1
                out_padding = 0
                
                if i == 2 and j == 0:
                    ne = out_channels
                modules.append(convTrans)
            
        # ========================
        self.cnn = nn.Sequential(*modules)

    def forward(self, h):
        # Tanh to scale to [-1, 1] (same dynamic range as original images).
        return torch.tanh(self.cnn(h))


class VAE(nn.Module):
    def __init__(self, features_encoder, features_decoder, in_size, z_dim):
        """
        :param features_encoder: Instance of an encoder the extracts features
        from an input.
        :param features_decoder: Instance of a decoder that reconstructs an
        input from it's features.
        :param in_size: The size of one input (without batch dimension).
        :param z_dim: The latent space dimension.
        """
        super().__init__()
        self.features_encoder = features_encoder
        self.features_decoder = features_decoder
        self.z_dim = z_dim

        self.features_shape, n_features = self._check_features(in_size)
        # TODO: Add parameters needed for encode() and decode().
        # ====== YOUR CODE: ======
        
        self.features_shape_flat = 1
        for dim in list(self.features_shape):
            self.features_shape_flat *= dim
        
        self.fc_mu = nn.Linear(self.features_shape_flat, z_dim, bias=True)
        self.fc_log_sigma2 = nn.Linear(self.features_shape_flat, z_dim, bias=True)
        
        self.fc_zh = nn.Linear(z_dim, self.features_shape_flat, bias=True)
        
        # ========================

    def _check_features(self, in_size):
        device = next(self.parameters()).device
        with torch.no_grad():
            # Make sure encoder and decoder are compatible
            x = torch.randn(1, *in_size, device=device)
            h = self.features_encoder(x)
            xr = self.features_decoder(h)
            assert xr.shape == x.shape
            # Return the shape and number of encoded features
            return h.shape[1:], torch.numel(h)//h.shape[0]

    def encode(self, x):
        # TODO: Sample a latent vector z given an input x.
        # 1. Use the features extracted from the input to obtain mu and
        # log_sigma2 (mean and log variance) of the posterior p(z|x).
        # 2. Apply the reparametrization trick.
        # ====== YOUR CODE: ======
        
        h = self.features_encoder(x)
        h_flat = h.reshape(-1, self.features_shape_flat)
        
        mu = self.fc_mu(h_flat).reshape(-1, self.z_dim)
        log_sigma2 = self.fc_log_sigma2(h_flat).reshape(-1, self.z_dim)
        normal_d = torch.distributions.Normal(0, 1)
        u = normal_d.sample((h_flat.shape[0], self.z_dim))
        
        z = mu + torch.mul(u, torch.exp(log_sigma2))
        # ========================

        return z, mu, log_sigma2

    def decode(self, z):
        # TODO: Convert a latent vector back into a reconstructed input.
        # 1. Convert latent to features.
        # 2. Apply features decoder.
        # ====== YOUR CODE: ======
        
        h_bar_flat = self.fc_zh(z)
        h_shape = [h_bar_flat.shape[0]]
        h_shape.extend(list(self.features_shape))
        h_bar = h_bar_flat.reshape(h_shape)
        
        x_rec = self.features_decoder(h_bar)
        
        # ========================

        # Scale to [-1, 1] (same dynamic range as original images).
        return torch.tanh(x_rec)

    def sample(self, n):
        samples = []
        device = next(self.parameters()).device
        with torch.no_grad():
            # TODO: Sample from the model.
            # Generate n latent space samples and return their reconstructions.
            # Remember that for the model, this is like inference.
            # ====== YOUR CODE: ======
            
            for i in range(n):
                normal_d = torch.distributions.Normal(0, 1)
                z = normal_d.sample((1, self.z_dim))
                x = self.decode(z)
                x = x.reshape(x.shape[1], x.shape[2], x.shape[3])
                samples.append(x)
            
            # ========================
        return samples

    def forward(self, x):
        z, mu, log_sigma2 = self.encode(x)
        return self.decode(z), mu, log_sigma2


def vae_loss(x, xr, z_mu, z_log_sigma2, x_sigma2):
    """
    Point-wise loss function of a VAE with latent space of dimension z_dim.
    :param x: Input image batch of shape (N,C,H,W).
    :param xr: Reconstructed (output) image batch.
    :param z_mu: Posterior mean (batch) of shape (N, z_dim).
    :param z_log_sigma2: Posterior log-variance (batch) of shape (N, z_dim).
    :param x_sigma2: Likelihood variance (scalar).
    :return:
        - The VAE loss
        - The data loss term
        - The KL divergence loss term
    all three are scalars, averaged over the batch dimension.
    """
    loss, data_loss, kldiv_loss = None, None, None
    # TODO: Implement the VAE pointwise loss calculation.
    # Remember:
    # 1. The covariance matrix of the posterior is diagonal.
    # 2. You need to average over the batch dimension.
    # ====== YOUR CODE: ======
    
    data_loss = (1/(x_sigma2)) * F.mse_loss(xr, x)
    
    N = x.shape[0]
    z_dim = z_mu.shape[1]
    kldiv_loss = 0
    
    z_sigma2 = torch.exp(z_log_sigma2)
    kldiv_loss = torch.sum(z_sigma2) + (torch.norm(z_mu)**2) - N*z_dim - torch.sum(z_log_sigma2)
    
    kldiv_loss /= N
    
    loss = data_loss + kldiv_loss
    
    # ========================

    return loss, data_loss, kldiv_loss
