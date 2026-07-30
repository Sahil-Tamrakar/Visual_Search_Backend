import streamlit as st
import requests
from PIL import Image
import os
import urllib.parse

st.set_page_config(page_title="Visual Product Search", layout="wide", page_icon="🔍")

API_URL = "http://localhost:8000/search"

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }

    #MainMenu, footer, header {visibility: hidden;}
    .block-container {padding-top: 2.5rem; max-width: 1100px;}

    :root {
        --primary: #059669;
        --accent: #EA580C;
        --bg: #ECFDF5;
        --card: #FFFFFF;
        --fg: #064E3B;
        --muted-fg: #64748B;
        --border: #A7F3D0;
    }

    .stApp { background-color: var(--bg); }

    h2 { color: var(--fg); font-weight: 700; }

    .stButton>button, .stLinkButton>a {
        border-radius: 8px;
        background-color: var(--primary) !important;
        color: white !important;
        border: none;
        font-weight: 500;
        padding: 0.5rem 1.2rem;
    }
    .stButton>button:hover, .stLinkButton>a:hover {
        background-color: #047857 !important;
    }

    div[data-testid="stImage"] img {
        border-radius: 10px;
        border: 1px solid var(--border);
    }

    div[data-testid="stFileUploader"] {
        border: 1px dashed var(--border);
        border-radius: 10px;
        padding: 0.5rem;
        background-color: var(--card);
    }
</style>
""", unsafe_allow_html=True)

st.markdown("## Visual Product Search")
st.caption("Upload a product photo to find visually similar items.")

SITE_TEMPLATES = {
    "Amazon India": "https://www.amazon.in/s?k={query}",
    "Amazon US": "https://www.amazon.com/s?k={query}",
    "Flipkart": "https://www.flipkart.com/search?q={query}",
    "Myntra": "https://www.myntra.com/{query_dash}",
    "eBay": "https://www.ebay.com/sch/i.html?_nkw={query}",
}

selected_site = st.selectbox("Search on", list(SITE_TEMPLATES.keys()), index=0)
uploaded_file = st.file_uploader(" ", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

if "num_results" not in st.session_state:
    st.session_state.num_results = 5
if "last_file" not in st.session_state:
    st.session_state.last_file = None

def build_link(site, product_name):
    template = SITE_TEMPLATES[site]
    query_encoded = urllib.parse.quote_plus(product_name)
    query_dash = product_name.replace(" ", "-").lower()
    return template.format(query=query_encoded, query_dash=urllib.parse.quote(query_dash))

if uploaded_file is not None:
    if uploaded_file.name != st.session_state.last_file:
        st.session_state.num_results = 5
        st.session_state.last_file = uploaded_file.name

    left, right = st.columns([1, 4])
    with left:
        st.image(uploaded_file, use_container_width=True)

    with st.spinner("Finding similar products..."):
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
        response = requests.post(API_URL, files=files, params={"k": st.session_state.num_results})

    if response.status_code == 200:
        results = response.json()["results"]
        st.markdown(f"**{len(results)} matches**")

        cols_per_row = 5
        for row_start in range(0, len(results), cols_per_row):
            row = results[row_start:row_start + cols_per_row]
            cols = st.columns(cols_per_row)
            for col, result in zip(cols, row):
                with col:
                    path = result["path"]
                    name = result.get("product_name", "Product")
                    if os.path.exists(path):
                        st.image(Image.open(path), use_container_width=True)
                    st.caption(f"{result['score']:.2f} match")
                    st.markdown(f"<small>{name}</small>", unsafe_allow_html=True)
                    link = build_link(selected_site, name)
                    st.link_button(f"Buy on {selected_site}", link, use_container_width=True)

        if st.session_state.num_results < 10:
            if st.button("Show more results"):
                st.session_state.num_results = 10
                st.rerun()
    else:
        st.error("Something went wrong fetching results.")