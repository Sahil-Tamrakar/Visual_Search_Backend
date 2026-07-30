import torch
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
from .config import MODEL_NAME

device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)
model.eval()

def get_image_embedding(image: Image.Image):
    image = image.convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.get_image_features(**inputs)

    if isinstance(output, torch.Tensor):
        embedding = output
    elif hasattr(output, "image_embeds"):
        embedding = output.image_embeds
    elif hasattr(output, "pooler_output"):
        embedding = output.pooler_output
    else:
        raise ValueError(f"Unexpected output type: {type(output)}")

    embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    return embedding.cpu().numpy().flatten()