# my_3d_text_editor/src/nerf_module/nerf_model.py
# Version: 1.1 - Added comments for clarity, ensured correct parameters in __init__

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoder(nn.Module):
    """
    Implements positional encoding for input coordinates.
    Maps input coordinates to a higher-dimensional space using sine and cosine functions.
    This helps the MLP learn high-frequency details.
    """

    def __init__(self, N_freqs: int, log_sampling: bool = True):
        super(PositionalEncoder, self).__init__()
        self.N_freqs = N_freqs
        self.log_sampling = log_sampling
        self.funcs = [torch.sin, torch.cos]

        if self.log_sampling:
            self.freq_bands = 2. ** (torch.linspace(0, N_freqs - 1, N_freqs)) * math.pi
        else:
            self.freq_bands = torch.linspace(1, N_freqs, N_freqs) * math.pi

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies positional encoding to the input tensor.
        """
        x_encoded_list = [x]
        for freq in self.freq_bands:
            x_encoded_list.extend([func(x * freq) for func in self.funcs])

        return torch.cat(x_encoded_list, dim=-1)


class NeRF(nn.Module):
    """
    NeRF (Neural Radiance Fields) model architecture.
    A multi-layer perceptron (MLP) that maps 5D coordinates (x,y,z,theta,phi)
    to color (RGB) and density (sigma).
    """

    def __init__(self,
                 D: int = 8,  # Number of layers for coordinate processing
                 W: int = 256,  # Width of each layer (number of neurons)
                 input_ch: int = 3,  # Number of input channels for coordinates (x,y,z) AFTER positional encoding
                 input_ch_views: int = 3,
                 # Number of input channels for view direction (theta,phi) AFTER positional encoding
                 output_ch: int = 4,  # Number of output channels (RGB + sigma = 3 + 1 = 4)
                 skips: list = [4],  # Layers to add skip connections (e.g., [4] means skip connection after 4th layer)
                 use_viewdirs: bool = True  # Whether to use view direction as input
                 ):
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs

        # Layers for processing spatial coordinates (x,y,z)
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] +
            [nn.Linear(W, W) if i not in skips else nn.Linear(W + input_ch, W) for i in range(D - 1)]
        )

        # Output density (sigma)
        # This layer acts as a bottleneck for sigma, then branches for RGB if viewdirs are used
        self.output_linear = nn.Linear(W, output_ch)  # This will initially output (W, 4)

        # Layers for processing view direction (theta,phi) if use_viewdirs is True
        if use_viewdirs:
            # The input to views_linear is the *last layer output from pts_linears (h)*
            # concatenated with the encoded view directions (input_ch_views).
            self.views_linear = nn.Linear(W + input_ch_views, W // 2)
            self.rgb_linear = nn.Linear(W // 2, 3)  # Final RGB output
        else:
            # If no view directions, RGB comes directly from the main branch's features.
            # In this simplified case, output_ch for output_linear should be 3 (for RGB only)
            # or the rgb_linear should be replaced. The provided structure assumes branching for RGB.
            self.rgb_linear = nn.Linear(W, 3)  # Final RGB output if no viewdirs (not common)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the NeRF MLP.

        Args:
            x (torch.Tensor): Concatenated encoded coordinates and view directions.
                              Shape: (batch_size, encoded_input_ch + encoded_input_ch_views)

        Returns:
            torch.Tensor: Raw output from NeRF. Shape: (batch_size, 4) (RGB + sigma)
        """
        # Split input into encoded coordinates and encoded view directions
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts

        # Process spatial coordinates through MLP layers
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], dim=-1)

        # Get raw density (sigma)
        # Sigma is typically branched off from the main spatial MLP (h)
        sigma = self.output_linear(h)[:, 3]  # Extract sigma (4th element)

        # Process RGB
        if self.use_viewdirs:
            # The input to views_linear is 'h' (output of spatial MLP) concatenated with view directions
            feature_for_views_branch = torch.cat([h, input_views], dim=-1)
            feature = self.views_linear(feature_for_views_branch)
            feature = F.relu(feature)
            rgb = self.rgb_linear(feature)
        else:
            # If not using view directions, RGB comes directly from the main spatial branch (h)
            rgb = self.output_linear(h)[:, :3]  # Extract RGB (first 3 elements)

        # Raw outputs
        return torch.cat([rgb, sigma.unsqueeze(-1)], dim=-1)


# Example usage (for testing purposes) - This part will not be called in main.py
if __name__ == '__main__':
    print("Testing NeRF model architecture...")

    # Define positional encoders
    pos_encoder_xyz = PositionalEncoder(N_freqs=10)
    pos_encoder_views = PositionalEncoder(N_freqs=4)

    # Calculate input dimensions after positional encoding
    encoded_xyz_dim = 3 + 3 * pos_encoder_xyz.N_freqs * 2
    encoded_views_dim = 3 + 3 * pos_encoder_views.N_freqs * 2

    # Instantiate NeRF model
    nerf_model = NeRF(D=8, W=256,
                      input_ch=encoded_xyz_dim,
                      input_ch_views=encoded_views_dim,
                      use_viewdirs=True).cpu()  # Use CPU for __main__ test if GPU isn't primary

    print(f"NeRF model instantiated with input_ch={encoded_xyz_dim}, input_ch_views={encoded_views_dim}")
    print(nerf_model)

    # Create dummy inputs (batch size 1,000)
    num_samples = 1000
    dummy_xyz = torch.randn(num_samples, 3).cpu()
    dummy_views = torch.randn(num_samples, 3).cpu()
    dummy_views = dummy_views / dummy_views.norm(dim=-1, keepdim=True)

    # Encode dummy inputs
    encoded_xyz = pos_encoder_xyz(dummy_xyz)
    encoded_views = pos_encoder_views(dummy_views)

    # Concatenate encoded inputs for NeRF's forward pass
    nerf_input = torch.cat([encoded_xyz, encoded_views], dim=-1)

    # Forward pass through NeRF model
    with torch.no_grad():
        raw_outputs = nerf_model(nerf_input)

    print(f"Shape of raw NeRF outputs: {raw_outputs.shape}")
    print("NeRF model architecture test successful.")