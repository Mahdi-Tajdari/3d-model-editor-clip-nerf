# my_3d_text_editor/src/main.py

import torch
from PIL import Image, UnidentifiedImageError
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import clip


# Import modules from your project structure
from src.clip_module.clip_loader import load_clip_model
from src.clip_module.clip_loss import calculate_clip_loss

# --- NEW IMPORTS FOR NERF ---
from src.nerf_module.nerf_model import NeRF, PositionalEncoder
from src.nerf_module.nerf_renderer import get_rays, render_rays
from src.nerf_module.optimization_loop import NeRFOptimizer  # Make sure this is imported


# --- END NEW IMPORTS FOR NERF ---


# --- run_clip_basic_test function definition ---
def run_clip_basic_test(project_root_path: str = None):
    """
    Performs a basic test of CLIP's functionality:
    Loading the model, preparing inputs, computing embeddings, and calculating similarity/loss.
    """
    if project_root_path is None:
        print("Warning: project_root_path not provided. Attempting to infer.")
        project_root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Starting basic CLIP test on device: {device}")

    # 1. Load the CLIP model
    clip_model, clip_preprocess = load_clip_model(device=device)
    if clip_model is None:
        print("Failed to load CLIP model. Exiting CLIP test.")
        return

    # 2. Prepare inputs (using local image paths from assets folder)
    base_image_dir = os.path.join(project_root_path, 'assets', 'sample_images')

    image_filenames = [
        "red_car.jpg",
        "cute_cat.jpg",
        "dog_on_grass.jpg"
    ]

    image_paths = [os.path.join(base_image_dir, filename) for filename in image_filenames]

    texts = [
        "A photo of a dog.",
        "A picture of a cat.",
        "A fast sports car."
    ]

    images_processed = []
    for path in image_paths:
        try:
            img = Image.open(path).convert("RGB")
            images_processed.append(clip_preprocess(img).unsqueeze(0))
        except FileNotFoundError:
            print(
                f"Error: Image file not found at {path}. Please ensure the file exists and is in the correct directory.")
            continue
        except UnidentifiedImageError as e:
            print(f"Error identifying image from {path}: {e}")
            continue

    if not images_processed:
        print("No images were successfully loaded for CLIP test. Exiting test.")
        return

    image_input_batch = torch.cat(images_processed).to(device)
    text_input_batch = clip.tokenize(texts).to(device)

    # 3. Compute Embeddings
    with torch.no_grad():
        image_features = clip_model.encode_image(image_input_batch)
        text_features = clip_model.encode_text(text_input_batch)

    # Normalize embeddings
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # 4. Calculate Similarity (for display purposes)
    similarity_matrix = (100.0 * image_features @ text_features.T)

    # 5. Display Results
    print("\n--- CLIP Similarity Results ---")
    for i, img_path in enumerate(image_paths):
        if i < len(images_processed):
            img_name = os.path.basename(img_path)
            print(f"\nImage: {img_name}")

            try:
                img_display = Image.open(img_path).convert("RGB")
                plt.imshow(img_display)
                plt.axis('off')
                plt.title(f"Image: {img_name}")
                plt.show()
            except Exception as e:
                print(f"  Could not display image {img_name}. Error: {e}")

            probs_for_display = similarity_matrix[i].softmax(dim=-1)
            top_probs, top_labels = probs_for_display.topk(len(texts))

            for prob, label in zip(top_probs, top_labels):
                print(f"  '{texts[label]}': {prob.item():.2f}%")

    if len(image_features) > 0 and len(text_features) > 0:
        sample_loss = calculate_clip_loss(image_features[0].unsqueeze(0), text_features[0].unsqueeze(0))
        print(f"\nExample CLIP Loss for first image and first text: {sample_loss.item():.4f}")
    else:
        print("\nNot enough features to calculate sample CLIP Loss. Ensure images loaded successfully.")

    print("\n--- CLIP basic test completed ---")


# --- run_nerf_basic_rendering_test function definition ---
def run_nerf_basic_rendering_test():
    """
    Performs a basic rendering test using a randomly initialized NeRF model.
    Implemented with ray batching to solve GPU memory issues.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nStarting basic NeRF rendering test on device: {device}")

    pos_encoder_xyz = PositionalEncoder(N_freqs=10).to(device)
    pos_encoder_views = PositionalEncoder(N_freqs=4).to(device)

    encoded_xyz_dim = 3 + 3 * pos_encoder_xyz.N_freqs * 2
    encoded_views_dim = 3 + 3 * pos_encoder_views.N_freqs * 2

    nerf_model = NeRF(D=8, W=256,
                      input_ch=encoded_xyz_dim,
                      input_ch_views=encoded_views_dim,
                      use_viewdirs=True).to(device)
    print("NeRF model instantiated (randomly initialized).")
    print(nerf_model)

    H, W, focal = 100, 100, 80.0
    c2w = torch.eye(4, device=device)

    rays_o, rays_d = get_rays(H, W, focal, c2w, device=device)

    near, far = 0.0, 1.0
    N_samples = 64
    raw_noise_std = 0.0

    chunk_size = 256  # Process 256 rays at a time. This is key for OOM.

    all_rgb_maps = []
    all_depth_maps = []

    for i in range(0, rays_o.shape[0], chunk_size):
        rays_o_chunk = rays_o[i:i + chunk_size]
        rays_d_chunk = rays_d[i:i + chunk_size]

        with torch.no_grad():
            outputs = render_rays(nerf_model, pos_encoder_xyz, pos_encoder_views,
                                  rays_o_chunk, rays_d_chunk, near, far, N_samples,
                                  rand=False, raw_noise_std=raw_noise_std, device=device)
            all_rgb_maps.append(outputs['rgb_map'])
            # all_depth_maps.append(outputs['depth_map'])

    rgb_map = torch.cat(all_rgb_maps, dim=0).reshape(H, W, 3).cpu().numpy()
    # depth_map = torch.cat(all_depth_maps, dim=0).reshape(H, W).cpu().numpy()

    rgb_map = np.clip(rgb_map, 0, 1)

    plt.figure(figsize=(5, 5))
    plt.imshow(rgb_map)
    plt.title("Rendered Image (Untrained NeRF - Expect Noise)")
    plt.axis('off')
    plt.show()

    print(f"Shape of rendered RGB map: {rgb_map.shape}")
    print("Basic NeRF rendering test completed.")


# --- NEW FUNCTION TO RUN 3D GENERATION/EDITING (DEFINITION IS HERE) ---
def run_3d_generation_or_editing(text_prompt: str, is_generation: bool, project_root_path: str):
    """
    Sets up and runs the CLIP-guided NeRF optimization process.

    Args:
        text_prompt (str): The target text description.
        is_generation (bool): True for generating a new 3D model, False for editing an existing one.
        project_root_path (str): Path to the project root directory.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n--- Starting 3D Generation/Editing for '{text_prompt}' on {device} ---")

    # Define NeRF and Optimization configurations
    # These would typically come from config files in the 'config/' directory
    nerf_config = {
        'D': 8, 'W': 256, 'pos_freqs_xyz': 10, 'pos_freqs_views': 4, 'use_viewdirs': True
    }
    optimization_config = {
        'learning_rate': 5e-4,
        'lr_decay_steps': [2500, 5000, 7500],
        'lr_decay_gamma': 0.1,
        'num_iterations': 10000,
        'render_H': 100, 'render_W': 100, 'render_focal': 80.0,
        'near': 0.0, 'far': 2.0, 'N_samples_per_ray': 64,
        'raw_noise_std': 0.0,
        'log_interval': 500,
        'rays_batch_size': 2048,
        'clip_eval_size': 224
    }

    # Initialize the optimizer instance
    optimizer_instance = NeRFOptimizer(
        text_prompt=text_prompt,
        nerf_config=nerf_config,
        optimization_config=optimization_config,
        project_root_path=project_root_path,
        device=device
    )

    # If editing, load pre-trained NeRF weights here
    if not is_generation:
        print("Note: Editing feature not fully implemented in this basic example.")
        pass  # Placeholder for loading existing NeRF weights

    # Run the optimization
    optimizer_instance.optimize_nerf()

    print(f"\n--- 3D Generation/Editing for '{text_prompt}' completed ---")


# --- Main execution block (CALLS THE run_3d_generation_or_editing function) ---
if __name__ == "__main__":
    # Determine project root path for file loading
    project_root_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

    # --- Run the 3D Generation/Editing process (CORE PROJECT FUNCTIONALITY) ---
    target_text = "A small red car"
    is_generation_mode = True

    run_3d_generation_or_editing(target_text, is_generation_mode, project_root_path)