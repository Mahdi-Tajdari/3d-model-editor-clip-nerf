# my_3d_text_editor/src/nerf_module/optimization_loop.py

import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm  # Using standard tqdm

# Import core NeRF and CLIP components
from src.nerf_module.nerf_model import NeRF, PositionalEncoder
from src.nerf_module.nerf_renderer import get_rays, render_rays
from src.clip_module.clip_loader import load_clip_model
import clip  # Make sure clip is imported here for tokenize (for text_features)
from src.clip_module.clip_loss import calculate_clip_loss


class NeRFOptimizer:
    """
    Manages the optimization loop for a NeRF model guided by CLIP.
    """

    def __init__(self,
                 text_prompt: str,
                 nerf_config: dict,
                 optimization_config: dict,
                 project_root_path: str,
                 device: str = "cuda"):

        self.text_prompt = text_prompt
        self.nerf_config = nerf_config  # Stored here
        self.opt_config = optimization_config  # Stored here ## FIX: Stored as self.opt_config
        self.project_root_path = project_root_path
        self.device = device

        # --- 1. Load CLIP model (fixed, not trainable) ---
        self.clip_model, self.clip_preprocess = load_clip_model(device=device)
        if self.clip_model is None:
            raise RuntimeError("Failed to load CLIP model. Cannot proceed with optimization.")

        # Encode text prompt once (fixed target for optimization)
        with torch.no_grad():
            self.text_features = self.clip_model.encode_text(clip.tokenize([text_prompt]).to(device))
            self.text_features = F.normalize(self.text_features, p=2, dim=-1)  # Normalize

        # --- 2. Initialize NeRF model and positional encoders ---
        # ## FIX: Changed to self.nerf_config
        self.pos_encoder_xyz = PositionalEncoder(N_freqs=self.nerf_config['pos_freqs_xyz']).to(device)
        self.pos_encoder_views = PositionalEncoder(N_freqs=self.nerf_config['pos_freqs_views']).to(device)

        # Calculate encoded dimensions
        encoded_xyz_dim = 3 + 3 * self.pos_encoder_xyz.N_freqs * 2
        encoded_views_dim = 3 + 3 * self.pos_encoder_views.N_freqs * 2

        # ## FIX: Changed to self.nerf_config
        self.nerf_model = NeRF(D=self.nerf_config['D'], W=self.nerf_config['W'],
                               input_ch=encoded_xyz_dim,
                               input_ch_views=encoded_views_dim,
                               use_viewdirs=self.nerf_config['use_viewdirs']).to(device)
        print("NeRF model initialized for optimization.")

        # --- 3. Initialize optimizer for NeRF model parameters ---
        # ## FIX: Changed to self.opt_config
        self.optimizer = optim.Adam(self.nerf_model.parameters(), lr=self.opt_config['learning_rate'])
        # ## FIX: Changed to self.opt_config
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                        milestones=self.opt_config['lr_decay_steps'],
                                                        gamma=self.opt_config['lr_decay_gamma'])

        # --- 4. Setup rendering parameters ---
        # ## FIX: Changed to self.opt_config
        self.H, self.W, self.focal = self.opt_config['render_H'], self.opt_config['render_W'], self.opt_config[
            'render_focal']
        self.near, self.far = self.opt_config['near'], self.opt_config['far']
        self.N_samples_per_ray = self.opt_config['N_samples_per_ray']
        self.raw_noise_std = self.opt_config['raw_noise_std']
        self.rays_batch_size = self.opt_config['rays_batch_size']
        self.clip_eval_size = self.opt_config['clip_eval_size']

        # Setup output directory
        self.results_dir = os.path.join(self.project_root_path, 'results', 'generated_renders',
                                        self.text_prompt.replace(" ", "_").replace("'", ""))
        os.makedirs(self.results_dir, exist_ok=True)
        print(f"Results will be saved in: {self.results_dir}")

        # Prepare fixed camera pose for CLIP evaluation renders
        self.clip_eval_c2w = torch.eye(4, device=self.device)
        self.eval_rays_o, self.eval_rays_d = get_rays(self.clip_eval_size, self.clip_eval_size,
                                                      self.focal, self.clip_eval_c2w, self.device)

    def optimize_nerf(self):
        """
        Runs the main optimization loop for NeRF guided by CLIP.
        """
        print(f"\nStarting optimization for text: '{self.text_prompt}'")
        print(
            f"Total iterations: {self.opt_config['num_iterations']}, Learning Rate: {self.opt_config['learning_rate']}")

        # Get ALL rays for the main render size (H, W) once at the beginning
        all_rays_o, all_rays_d = get_rays(self.H, self.W, self.focal, self.clip_eval_c2w, self.device)
        num_rays_total = all_rays_o.shape[0]

        for i in tqdm(range(self.opt_config['num_iterations']),
                      desc="Optimizing NeRF"):  # ## FIX: Changed to self.opt_config
            self.optimizer.zero_grad()

            # --- 1. Sample a batch of rays for optimization ---
            rand_idx = torch.randperm(num_rays_total, device=self.device)[
                       :self.rays_batch_size]  # ## FIX: Changed to self.rays_batch_size
            batch_rays_o = all_rays_o[rand_idx]
            batch_rays_d = all_rays_d[rand_idx]

            # --- 2. Render raw outputs from NeRF for this batch of rays ---
            outputs = render_rays(self.nerf_model, self.pos_encoder_xyz, self.pos_encoder_views,
                                  batch_rays_o, batch_rays_d,
                                  self.near, self.far, self.N_samples_per_ray,
                                  # ## FIX: Changed to self.near, self.far, self.N_samples_per_ray
                                  rand=True,
                                  raw_noise_std=self.raw_noise_std,  # ## FIX: Changed to self.raw_noise_std
                                  device=self.device)

            rendered_rgb = outputs['rgb_map']

            # --- 3. Compute CLIP Loss (main guiding loss) ---
            with torch.no_grad():
                eval_outputs = render_rays(self.nerf_model, self.pos_encoder_xyz, self.pos_encoder_views,
                                           self.eval_rays_o, self.eval_rays_d, self.near, self.far,
                                           # ## FIX: Changed to self.near, self.far
                                           self.N_samples_per_ray, rand=False, raw_noise_std=0.0,
                                           # ## FIX: Changed to self.N_samples_per_ray
                                           device=self.device)
            eval_rgb_map = eval_outputs['rgb_map'].reshape(self.clip_eval_size, self.clip_eval_size,
                                                           3)  # ## FIX: Changed to self.clip_eval_size

            pil_image = Image.fromarray((eval_rgb_map.cpu().numpy() * 255).astype(np.uint8))
            processed_for_clip = self.clip_preprocess(pil_image).unsqueeze(0).to(self.device)

            image_features = self.clip_model.encode_image(processed_for_clip)
            image_features = F.normalize(image_features, p=2, dim=-1)

            clip_loss = calculate_clip_loss(image_features, self.text_features)

            total_loss = clip_loss

            # --- 4. Backpropagation and Optimization ---
            total_loss.backward()
            self.optimizer.step()
            self.scheduler.step()  # Update learning rate

            # --- 5. Logging and Visualization ---
            if (i + 1) % self.opt_config['log_interval'] == 0:  # ## FIX: Changed to self.opt_config
                print(
                    f"Iter {i + 1}/{self.opt_config['num_iterations']}, Loss: {total_loss.item():.4f}, LR: {self.optimizer.param_groups[0]['lr']:.6f}")  # ## FIX: Changed to self.opt_config

                save_path = os.path.join(self.results_dir, f"iter_{i + 1:06d}.png")
                plt.imsave(save_path, np.clip(eval_rgb_map.cpu().numpy(), 0, 1))

        print("\nOptimization completed.")
        final_model_path = os.path.join(self.results_dir, "final_nerf_model.pth")
        torch.save(self.nerf_model.state_dict(), final_model_path)
        print(f"Final NeRF model saved to {final_model_path}")

        plt.figure(figsize=(5, 5))
        plt.imshow(np.clip(eval_rgb_map.cpu().numpy(), 0, 1))
        plt.title(f"Final Rendered Image for '{self.text_prompt}'")
        plt.axis('off')
        plt.show()


# Example usage (for standalone testing of optimization_loop.py)
if __name__ == '__main__':
    print("Testing NeRFOptimizer as a standalone module...")

    # Define dummy configs (these would come from config files in a real project)
    dummy_nerf_config = {
        'D': 8, 'W': 256, 'pos_freqs_xyz': 10, 'pos_freqs_views': 4, 'use_viewdirs': True
    }
    dummy_optimization_config = {
        'learning_rate': 5e-4,
        'lr_decay_steps': [100, 200],
        'lr_decay_gamma': 0.1,
        'num_iterations': 500,
        'render_H': 100, 'render_W': 100, 'render_focal': 80.0,
        'near': 0.0, 'far': 2.0, 'N_samples_per_ray': 64,
        'raw_noise_std': 0.0,
        'log_interval': 50,
        'rays_batch_size': 256,
        'clip_eval_size': 128
    }

    test_text_prompt = "A red ball"
    test_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

    optimizer_instance = NeRFOptimizer(test_text_prompt, dummy_nerf_config, dummy_optimization_config,
                                       test_project_root)
    optimizer_instance.optimize_nerf()
    print("NeRFOptimizer standalone test completed.")