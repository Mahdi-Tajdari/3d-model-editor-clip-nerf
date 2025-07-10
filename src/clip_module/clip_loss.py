# --- STEP 3.2: Write src/clip_module/clip_loss.py ---
%%writefile src/clip_module/clip_loss.py
import torch
import torch.nn.functional as F

def calculate_clip_loss(image_features: torch.Tensor, text_features: torch.Tensor, logit_scale: float = 100.0) -> torch.Tensor:
    """
    Calculates the symmetric CLIP loss (contrastive loss).

    Args:
        image_features (torch.Tensor): Normalized image embeddings.
                                       Shape: (batch_size, embedding_dim)
        text_features (torch.Tensor): Normalized text embeddings.
                                      Shape: (batch_size, embedding_dim)
        logit_scale (float): A learnable temperature parameter, often initialized to 100.0.

    Returns:
        torch.Tensor: The scalar CLIP loss.
    """
    # Compute similarity logits
    # (batch_size, batch_size) matrix where element (i, j) is similarity between image_i and text_j
    logits = (image_features @ text_features.T) * logit_scale

    # Create ground truth labels for contrastive loss
    # The diagonal elements are the positive pairs (image_i, text_i)
    labels = torch.arange(len(logits)).to(logits.device)

    # Calculate cross-entropy loss for image-to-text and text-to-image
    loss_i = F.cross_entropy(logits, labels)
    loss_t = F.cross_entropy(logits.T, labels) # Transpose logits for text-to-image

    # CLIP loss is the average of these two losses
    total_loss = (loss_i + loss_t) / 2
    return total_loss

if __name__ == '__main__':
    # Example usage:
    # Create dummy normalized features
    dummy_image_features = torch.randn(4, 512)
    dummy_text_features = torch.randn(4, 512)
    dummy_image_features = dummy_image_features / dummy_image_features.norm(dim=-1, keepdim=True)
    dummy_text_features = dummy_text_features / dummy_text_features.norm(dim=-1, keepdim=True)

    loss = calculate_clip_loss(dummy_image_features, dummy_text_features)
    print(f"Example CLIP Loss: {loss.item():.4f}")