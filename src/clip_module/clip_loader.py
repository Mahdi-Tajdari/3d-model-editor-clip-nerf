# --- STEP 3.1: Write src/clip_module/clip_loader.py ---
%%writefile src/clip_module/clip_loader.py
import torch
import clip

def load_clip_model(model_name="ViT-B/32", device="cuda"):
    """
    Loads a pre-trained CLIP model and its preprocessing function.

    Args:
        model_name (str): The name of the CLIP model to load (e.g., "ViT-B/32").
        device (str): The device to load the model onto ('cuda' or 'cpu').

    Returns:
        tuple: A tuple containing the CLIP model and its preprocessing function.
    """
    try:
        model, preprocess = clip.load(model_name, device=device)
        print(f"CLIP model '{model_name}' loaded successfully on {device}.")
        return model, preprocess
    except Exception as e:
        print(f"Error loading CLIP model: {e}")
        return None, None

if __name__ == '__main__':
    # Example usage:
    clip_model, clip_preprocess = load_clip_model()
    if clip_model and clip_preprocess:
        print("CLIP model and preprocess function loaded for testing.")