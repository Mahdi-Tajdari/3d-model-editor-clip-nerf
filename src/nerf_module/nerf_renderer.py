# my_3d_text_editor/src/nerf_module/nerf_renderer.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Import NeRF model components
from src.nerf_module.nerf_model import NeRF, PositionalEncoder


def raw2outputs(raw: torch.Tensor, z_vals: torch.Tensor, rays_d: torch.Tensor, raw_noise_std: float = 0.0) -> dict:
    """
    Transforms raw NeRF outputs (RGB and sigma) into renderable outputs (colors and depths).
    Implements the volume rendering equation.
    """
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[..., :1].shape).to(dists.device)], dim=-1)

    alpha = F.relu(raw[..., 3] + raw_noise_std * torch.randn(raw[..., 3].shape, device=raw.device))
    alpha = 1. - torch.exp(-alpha * dists)

    weights = alpha * torch.cumprod(
        torch.cat([torch.ones((alpha.shape[0], 1), device=alpha.device), 1. - alpha + 1e-10], dim=-1), dim=-1)[:, :-1]

    rgb_map = torch.sum(weights[..., None] * F.sigmoid(raw[..., :3]), dim=-2)
    depth_map = torch.sum(weights * z_vals, dim=-1)
    acc_map = torch.sum(weights, dim=-1)

    return {'rgb_map': rgb_map, 'depth_map': depth_map, 'acc_map': acc_map, 'weights': weights}


# --- CRITICAL FIX FOR GET_RAYS ---
def get_rays(H: int, W: int, focal: float, c2w: torch.Tensor, device: str = "cuda") -> tuple:
    """
    Gets ray origin and direction for all pixels in an image given camera intrinsics and extrinsics.

    Args:
        H (int): Image height.
        W (int): Image width.
        focal (float): Focal length of the camera.
        c2w (torch.Tensor): Camera-to-world transformation matrix (extrinsics). Shape: (4, 4).
        device (str): Device to place the generated ray tensors.

    Returns:
        tuple: (rays_o, rays_d) - Ray origins and directions.
               rays_o shape: (H*W, 3)
               rays_d shape: (H*W, 3)
    """
    # Create coordinate grid for pixels (normalized from -1 to 1)
    # i and j represent pixel coordinates in the image plane
    i, j = torch.meshgrid(torch.linspace(0, W - 1, W), torch.linspace(0, H - 1, H), indexing='ij')
    i = i.T.to(device)  # Transpose and move to device
    j = j.T.to(device)  # Transpose and move to device

    # Convert pixel coordinates (i,j) into ray directions in camera space
    # These directions are relative to the camera's optical axis.
    # The -1 for Z-axis means looking into the screen.
    dirs = torch.stack([(i - W * .5) / focal,
                        -(j - H * .5) / focal,  # Y-axis inverted for image coords
                        -torch.ones_like(i)], dim=-1)  # Z-axis pointing away from camera

    # Transform ray directions from camera space to world space
    # rays_d_camera_space: (H, W, 3)
    # c2w[:3,:3]: rotation matrix R (3, 3) from camera-to-world
    # We want to apply R to each (3,) vector in rays_d_camera_space

    # Reshape dirs from (H, W, 3) to (H*W, 3) for efficient matrix multiplication
    dirs_flat = dirs.reshape(-1, 3)  # Shape: (H*W, 3)

    # Apply camera-to-world rotation to ray directions
    # (H*W, 3) @ (3, 3) -> (H*W, 3)
    # The rotation matrix c2w[:3,:3] is applied to transform vectors.
    rays_d = torch.matmul(dirs_flat, c2w[:3, :3].T)  # Applying R to (x,y,z) directions

    # Ray origin is the camera position in world space
    # c2w[:3,-1] is the translation vector (camera position) (3,)
    # We need to expand it to match the number of rays.
    rays_o = c2w[:3, -1].expand(rays_d.shape)  # Expand camera origin (3,) to (H*W, 3)

    return rays_o, rays_d


# --- END CRITICAL FIX FOR GET_RAYS ---


def render_rays(nerf_model: nn.Module,
                pos_encoder_xyz: PositionalEncoder,
                pos_encoder_views: PositionalEncoder,
                rays_o: torch.Tensor,
                rays_d: torch.Tensor,  # Shape should now be (num_rays, 3)
                near: float,
                far: float,
                N_samples: int,
                rand: bool = False,
                raw_noise_std: float = 0.0,
                device: str = "cuda") -> dict:
    """
    Renders an image by tracing rays through the NeRF model.
    """
    t_vals = torch.linspace(0., 1., N_samples).to(device)
    z_vals = near * (1. - t_vals) + far * t_vals

    if rand:
        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], dim=-1)
        lower = torch.cat([z_vals[..., :1], mids], dim=-1)
        rand_dist = torch.rand(z_vals.shape).to(device) * (upper - lower)
        z_vals = lower + rand_dist

    # Get sample points in 3D space: o + t*d
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # Shape: (num_rays, N_samples, 3)

    # Encode coordinates and view directions using positional encoders
    input_pts_encoded = pos_encoder_xyz(pts.reshape(-1, 3))  # Shape: (num_rays * N_samples, encoded_xyz_dim)

    # --- THIS LINE WAS THE PROBLEM, NOW RAYS_D IS CORRECTLY SHAPED ---
    # rays_d is (num_rays, 3). We need it repeated N_samples times for each ray.
    # Unsqueeze to (num_rays, 1, 3), then expand to (num_rays, N_samples, 3), then reshape to (-1, 3).
    input_views_encoded = pos_encoder_views(rays_d.unsqueeze(1).expand(-1, N_samples, -1).reshape(-1, 3))
    # --- END CRITICAL FIX ---

    # Pass encoded inputs through NeRF model
    raw_outputs = nerf_model(torch.cat([input_pts_encoded, input_views_encoded], dim=-1))

    # Reshape raw outputs to (num_rays, N_samples, 4)
    raw_outputs = raw_outputs.reshape(rays_o.shape[0], N_samples, raw_outputs.shape[-1])

    # Transform raw outputs to renderable outputs using volume rendering
    outputs = raw2outputs(raw_outputs, z_vals, rays_d, raw_noise_std)
    return outputs


if __name__ == '__main__':
    print("Testing NeRF renderer...")

    # Basic setup
    H, W, focal = 100, 100, 80  # Example image dimensions and focal length
    test_device = "cuda" if torch.cuda.is_available() else "cpu"
    c2w = torch.eye(4, device=test_device)

    # Initialize positional encoders and NeRF model
    pos_encoder_xyz = PositionalEncoder(N_freqs=10).to(test_device)
    pos_encoder_views = PositionalEncoder(N_freqs=4).to(test_device)
    encoded_xyz_dim = 3 + 3 * 10 * 2
    encoded_views_dim = 3 + 3 * 4 * 2
    nerf_model = NeRF(input_ch=encoded_xyz_dim, input_ch_views=encoded_views_dim, use_viewdirs=True).to(test_device)

    # Get rays for an entire image
    rays_o, rays_d = get_rays(H, W, focal, c2w, device=test_device)  # Pass device here

    # Define rendering parameters
    near, far = 0., 1.
    N_samples = 64
    raw_noise_std = 0.0

    with torch.no_grad():
        outputs = render_rays(nerf_model, pos_encoder_xyz, pos_encoder_views,
                              rays_o, rays_d, near, far, N_samples,
                              rand=False, raw_noise_std=raw_noise_std, device=test_device)

    rgb_map = outputs['rgb_map'].reshape(H, W, 3).cpu().numpy()
    plt.imshow(rgb_map)
    plt.title("Rendered Image (Untrained NeRF)")
    plt.axis('off')
    plt.show()

    print(f"Shape of rendered RGB map: {rgb_map.shape}")
    print("NeRF renderer test successful (output will be random for untrained model).")