import torch
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
from PIL import Image
from .config import MODEL_NAME

# Render free instance is CPU-only.
device = torch.device("cpu")

# Reduce unnecessary CPU/thread overhead.
torch.set_num_threads(1)

print("Loading CLIP vision model...")

# Load ONLY the vision side of CLIP.
# We don't need CLIP's text encoder for image-to-image search.
model = CLIPVisionModelWithProjection.from_pretrained(
    MODEL_NAME,
    low_cpu_mem_usage=True
)

model.eval()
model.to(device)

processor = CLIPImageProcessor.from_pretrained(MODEL_NAME)

print("CLIP vision model loaded.")


def get_image_embedding(image: Image.Image):

    image = image.convert("RGB")

    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    pixel_values = inputs["pixel_values"].to(device)

    # inference_mode uses less overhead than normal gradient execution
    with torch.inference_mode():

        outputs = model(
            pixel_values=pixel_values
        )

        embedding = outputs.image_embeds

    # Normalize embedding for similarity search
    embedding = embedding / embedding.norm(
        dim=-1,
        keepdim=True
    )

    return embedding.cpu().numpy().flatten()