# --- STEP 3.3: Write src/main.py ---
%%writefile
src / main.py
import torch
from PIL import Image, UnidentifiedImageError
import matplotlib.pyplot as plt
import numpy as np
import os
import clip  # Make sure clip is imported here for tokenize

# Import modules from your project structure
from src.clip_module.clip_loader import load_clip_model
from src.clip_module.clip_loss import calculate_clip_loss


def run_clip_basic_test():
    """
    Performs a basic test of CLIP's functionality:
    Loading the model, preparing inputs, computing embeddings, and calculating similarity/loss.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Starting basic CLIP test on device: {device}")

    # 1. Load the CLIP model
    clip_model, clip_preprocess = load_clip_model(device=device)
    if clip_model is None:
        print("Failed to load CLIP model. Exiting test.")
        return

    # 2. Prepare inputs (using local image paths from assets folder)
    # Construct the base path to your sample images relative to the project root
    # os.path.abspath(__file__) gets the full path of the current script (e.g., /content/my_3d_text_editor/src/main.py)
    # os.path.dirname(...) gets its directory (e.g., /content/my_3d_text_editor/src/)
    # os.path.join(..., '..', 'assets', 'sample_images') navigates up to project root and then to assets/sample_images
    base_image_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'assets', 'sample_images')

    # List of image filenames. Ensure these files exist in assets/sample_images/
    image_filenames = [
        "red_car.jpg",
        "cute_cat.jpg",
        "dog_on_grass.jpg"
    ]

    # Construct full paths
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
            images_processed.append(clip_preprocess(img).unsqueeze(0))  # Add batch dimension
        except FileNotFoundError:
            print(
                f"Error: Image file not found at {path}. Please ensure the file exists and is in the correct directory.")
            continue
        except UnidentifiedImageError as e:
            print(f"Error identifying image from {path}: {e}")
            continue

    if not images_processed:
        print("No images were successfully loaded. Exiting test.")
        return

    image_input_batch = torch.cat(images_processed).to(device)
    text_input_batch = clip.tokenize(texts).to(device)

    # 3. Compute Embeddings
    with torch.no_grad():
        image_features = clip_model.encode_image(image_input_batch)
        text_features = clip_model.encode_text(text_input_batch)

    # Normalize embeddings (crucial for cosine similarity)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # 4. Calculate Similarity (for display purposes, not for model update)
    similarity_matrix = (100.0 * image_features @ text_features.T)

    # 5. Display Results
    print("\n--- Similarity Results ---")
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

            # Print similarity scores
            probs_for_display = similarity_matrix[i].softmax(dim=-1)
            top_probs, top_labels = probs_for_display.topk(len(texts))

            for prob, label in zip(top_probs, top_labels):
                print(f"  '{texts[label]}': {prob.item():.2f}%")

    # Example of calculating CLIP Loss for a specific pair
    if len(image_features) > 0 and len(text_features) > 0:
        sample_loss = calculate_clip_loss(image_features[0].unsqueeze(0), text_features[0].unsqueeze(0))
        print(f"\nExample CLIP Loss for first image and first text: {sample_loss.item():.4f}")
    else:
        print("\nNot enough features to calculate sample CLIP Loss. Ensure images loaded successfully.")


if __name__ == "__main__":
    run_clip_basic_test()