import faiss
import pickle
import numpy as np
import pandas as pd
import os
from .config import INDEX_PATH, PATHS_PICKLE, METADATA_PATH

index = faiss.read_index(INDEX_PATH)
with open(PATHS_PICKLE, "rb") as f:
    valid_paths = pickle.load(f)

DATA_PREFIX = "data/"

# load metadata once at startup
metadata_df = pd.read_csv(METADATA_PATH, on_bad_lines="skip")
id_to_name = dict(zip(metadata_df["id"], metadata_df["productDisplayName"]))

def get_product_id(path):
    filename = os.path.basename(path)
    return int(os.path.splitext(filename)[0])

def search_similar(query_embedding, k=5):
    query_embedding = query_embedding.astype("float32").reshape(1, -1)
    scores, indices = index.search(query_embedding, k)
    results = []
    for j, i in enumerate(indices[0]):
        raw_path = valid_paths[i]
        pid = get_product_id(raw_path)
        product_name = id_to_name.get(pid, "Product")
        results.append({
            "path": DATA_PREFIX + raw_path,
            "score": float(scores[0][j]),
            "product_name": product_name
        })
    return results

# Metadata needed for complementary lookups (subCategory column)
id_to_subcategory = dict(zip(metadata_df["id"], metadata_df["subCategory"]))

def get_subcategory(path):
    pid = get_product_id(path)
    return id_to_subcategory.get(pid)

# Which categories pair well with which — built from this dataset's real subCategory values
COMPLEMENTARY_MAP = {
    "Topwear": ["Bottomwear", "Shoes", "Sandal", "Flip Flops", "Watches", "Belts", "Bags", "Ties", "Scarves"],
    "Bottomwear": ["Topwear", "Shoes", "Sandal", "Flip Flops", "Belts", "Watches", "Bags"],
    "Shoes": ["Topwear", "Bottomwear", "Socks"],
    "Sandal": ["Topwear", "Bottomwear"],
    "Flip Flops": ["Topwear", "Bottomwear"],
    "Watches": ["Topwear", "Bottomwear"],
    "Bags": ["Topwear", "Bottomwear", "Shoes"],
    "Belts": ["Topwear", "Bottomwear"],
    "Socks": ["Shoes", "Bottomwear"],
    "Dress": ["Shoes", "Sandal", "Bags", "Watches", "Belts", "Jewellery"],
    "Saree": ["Bags", "Jewellery", "Watches"],
    "Jewellery": ["Dress", "Saree", "Topwear"],
    "Eyewear": ["Topwear", "Bottomwear"],
    "Headwear": ["Topwear", "Bottomwear"],
    "Wallets": ["Bags", "Belts"],
    "Innerwear": ["Topwear", "Bottomwear"],
    "Loungewear and Nightwear": ["Flip Flops", "Sandal"],
    "Apparel Set": ["Shoes", "Sandal", "Bags", "Watches"],
    "Stoles": ["Topwear", "Dress"],
    "Scarves": ["Topwear", "Dress"],
    "Mufflers": ["Topwear"],
    "Gloves": ["Topwear"],
    "Cufflinks": ["Topwear"],
}

# Pre-build one FAISS sub-index per complementary category ONCE at startup,
# instead of rebuilding a filtered index on every request (fast lookups, no per-query cost)
_category_to_indices = {}
for idx, path in enumerate(valid_paths):
    subcat = get_subcategory(path)
    if subcat:
        _category_to_indices.setdefault(subcat, []).append(idx)

def find_complementary(query_embedding, query_path, k=5):
    query_subcat = get_subcategory(query_path)
    if query_subcat not in COMPLEMENTARY_MAP:
        return []

    allowed_subcats = COMPLEMENTARY_MAP[query_subcat]

    # gather all catalog indices belonging to allowed complementary categories
    candidate_indices = []
    for subcat in allowed_subcats:
        candidate_indices.extend(_category_to_indices.get(subcat, []))

    if not candidate_indices:
        return []

    # reconstruct embeddings for just these candidates from the existing FAISS index
    # (avoids re-embedding images — index.reconstruct pulls stored vectors back out)
    candidate_embeddings = np.array(
        [index.reconstruct(i) for i in candidate_indices]
    ).astype("float32")

    temp_index = faiss.IndexFlatIP(candidate_embeddings.shape[1])
    temp_index.add(candidate_embeddings)

    query_embedding = query_embedding.astype("float32").reshape(1, -1)
    scores, local_indices = temp_index.search(query_embedding, min(k, len(candidate_indices)))

    results = []
    for j, local_i in enumerate(local_indices[0]):
        global_i = candidate_indices[local_i]
        raw_path = valid_paths[global_i]
        pid = get_product_id(raw_path)
        product_name = id_to_name.get(pid, "Product")
        results.append({
            "path": DATA_PREFIX + raw_path,
            "score": float(scores[0][j]),
            "product_name": product_name
        })
    return results