import torch
from transformers import CLIPModel, CLIPProcessor, CLIPVisionModel


def save_model(clip_model, save_dir, epoch=None):
    """
    Save the fine-tuned model components.

    Args:
    clip_model: The fine-tuned CLIP model.
    save_dir: Directory to save the model.
    epoch: Current epoch (optional). If provided, saves checkpoint with epoch number.
    """
    import os
    os.makedirs(save_dir, exist_ok=True)

    try:
        if epoch is not None:
            # Save epoch checkpoint
            checkpoint_path = f"{save_dir}/epoch_{epoch}_state_dict.pth"
            torch.save(clip_model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}")
        else:
            # Save the entire model
            final_model_path = f"{save_dir}/fine_tuned_clip_model.pth"
            torch.save(clip_model.state_dict(), final_model_path)
            print(f"Final model state dict saved at {final_model_path}")

            # Save vision model separately
            vision_model_path = f"{save_dir}/vision_model_state.pth"
            torch.save(clip_model.vision_model.state_dict(), vision_model_path)
            print(f"Vision model saved at {vision_model_path}")

            # Save classification head separately
            classification_head_path = f"{save_dir}/classification_head_state.pth"
            torch.save(clip_model.classification_head.state_dict(), classification_head_path)
            print(f"Classification head saved at {classification_head_path}")

    except RuntimeError as e:
        print(f"Error saving model: {e}")


# Example usage
if __name__ == "__main__":
    # Assuming `clip_model` is your fine-tuned CLIP model
    save_dir = "./models/wikiart"

    # Save the model after each epoch
    for epoch in range(1, 6):  # Replace with the actual number of epochs
        save_model(clip_model, save_dir, epoch=epoch)

    # Save the final model components
    save_model(clip_model, save_dir)
