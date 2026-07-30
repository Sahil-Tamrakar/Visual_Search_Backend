from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
import io

from .model import get_image_embedding
from .search import search_similar, find_complementary, DATA_PREFIX
from .web_search import upload_to_imgbb, google_lens_search

app = FastAPI(title="Visual Product Search API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/images", StaticFiles(directory="data"), name="images")

@app.get("/")
def health_check():
    return {"status": "ok"}

@app.post("/search")
async def search(file: UploadFile = File(...), k: int = 5):
    """Catalog search — your own CLIP + FAISS pipeline."""
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))
    embedding = get_image_embedding(image)
    results = search_similar(embedding, k=k)
    return {"query_filename": file.filename, "results": results}

@app.post("/web-search")
async def web_search(file: UploadFile = File(...), k: int = 10):
    """Web search — real Google Lens results via SerpApi, localized to India."""
    image_bytes = await file.read()
    public_url = upload_to_imgbb(image_bytes)
    results = google_lens_search(public_url, num_results=k, country="in", language="en")
    return {"query_filename": file.filename, "results": results}

@app.post("/complementary")
async def complementary(file: UploadFile = File(...), k: int = 5):
    """Suggests items from DIFFERENT but complementary categories (e.g. shirt -> pants/shoes)."""
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))
    embedding = get_image_embedding(image)

    # find_complementary needs a catalog path to look up the query's own category —
    # so we reuse the top catalog match's path as a proxy for "what category is this?"
    top_match = search_similar(embedding, k=1)
    if not top_match:
        return {"query_filename": file.filename, "results": []}

    query_path = top_match[0]["path"].replace(DATA_PREFIX, "", 1)
    results = find_complementary(embedding, query_path, k=k)
    return {"query_filename": file.filename, "results": results}