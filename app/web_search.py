import requests
from .config import IMGBB_API_KEY, SERPAPI_KEY

def upload_to_imgbb(image_bytes: bytes) -> str:
    """Uploads image bytes to imgbb and returns a public URL Google can fetch."""
    response = requests.post(
        "https://api.imgbb.com/1/upload",
        params={"key": IMGBB_API_KEY},
        files={"image": image_bytes},
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()
    return data["data"]["url"]

def google_lens_search(image_url: str, num_results: int = 10, country: str = "in", language: str = "en"):
    """Calls SerpApi's Google Lens engine, localized to a specific country, and returns parsed visual matches."""
    response = requests.get(
        "https://serpapi.com/search",
        params={
            "engine": "google_lens",
            "url": image_url,
            "api_key": SERPAPI_KEY,
            "country": country,
            "hl": language,
        },
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()

    matches = data.get("visual_matches", [])
    results = []
    for m in matches[:num_results]:
        price_info = m.get("price")
        results.append({
            "product_name": m.get("title", "Product"),
            "path": m.get("thumbnail", ""),
            "score": 1.0,
            "source": m.get("source", "Unknown"),
            "link": m.get("link", "#"),
            "price": price_info.get("value") if price_info else None,
        })
    return results