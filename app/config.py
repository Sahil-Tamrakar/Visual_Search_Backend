import os
from dotenv import load_dotenv

load_dotenv()

INDEX_PATH = "data/catalog_index.faiss"
PATHS_PICKLE = "data/valid_paths.pkl"
MODEL_NAME = "openai/clip-vit-base-patch32"
METADATA_PATH = "data/catalog_data/myntradataset/styles.csv"

IMGBB_API_KEY = os.getenv("IMGBB_API_KEY")
SERPAPI_KEY = os.getenv("SERPAPI_KEY")