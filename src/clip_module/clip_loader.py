# my_3d_text_editor/src/clip_module/clip_loader.py

import torch
import clip


# --- CRITICAL FIX: Changed default model_name to "RN50" for lower GPU memory usage ---
def load_clip_model(model_name: str = "RN50", device: str = "cuda"):
    """
    Loads a pre-trained CLIP model and its preprocessing function.
    """
    if not torch.cuda.is_available() and device == "cuda":
        print("Warning: CUDA is not available. Model will be loaded on CPU.")
        device = "cpu"

    try:
        model, preprocess = clip.load(model_name, device=device)
        print(f"CLIP model '{model_name}' loaded successfully on {device}.")
        return model, preprocess
    except Exception as e:
        print(f"Error loading CLIP model: {e}. Check model_name or connectivity. Trying CPU...")
        try:  # Fallback to CPU if GPU loading fails
            model, preprocess = clip.load(model_name, device="cpu")
            print(f"CLIP model '{model_name}' loaded successfully on CPU as fallback.")
            return model, preprocess
        except Exception as e_cpu:
            print(f"Error loading CLIP model even on CPU: {e_cpu}")
            return None, None