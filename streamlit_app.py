import streamlit as st
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import requests
from io import BytesIO

# ==============================
# CONFIG & STYLING
# ==============================
st.set_page_config(
    page_title="Landmark Classification App",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom dark CSS
st.markdown("""
    <style>
        body {
            background-color: #0E1117;
            color: #FAFAFA;
        }
        .stApp {
            background-color: #0E1117;
        }
        h1, h2, h3, h4 {
            color: #FAFAFA !important;
        }
        .css-1d391kg {
            color: #FAFAFA !important;
        }

        /* Class name grid cards */
        .class-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
            gap: 10px;
            margin-top: 12px;
        }
        .class-card {
            background: #1C2333;
            border: 1px solid #2E3A50;
            border-radius: 8px;
            padding: 10px 14px;
            display: flex;
            align-items: center;
            gap: 10px;
            font-size: 0.85rem;
            color: #C9D1D9;
            transition: border-color 0.2s;
        }
        .class-card:hover {
            border-color: #58A6FF;
        }
        .class-index {
            background: #2E3A50;
            color: #58A6FF;
            border-radius: 4px;
            padding: 2px 7px;
            font-size: 0.75rem;
            font-weight: 700;
            font-family: monospace;
            min-width: 28px;
            text-align: center;
        }
        .class-name {
            font-weight: 500;
        }

        /* Example image cards */
        .example-label {
            font-size: 0.78rem;
            color: #8B949E;
            text-align: center;
            margin-top: 6px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }

        /* Dataset stats bar */
        .stats-row {
            display: flex;
            gap: 24px;
            margin: 16px 0 8px 0;
            flex-wrap: wrap;
        }
        .stat-pill {
            background: #1C2333;
            border: 1px solid #2E3A50;
            border-radius: 20px;
            padding: 6px 18px;
            font-size: 0.85rem;
            color: #C9D1D9;
        }
        .stat-pill span {
            color: #58A6FF;
            font-weight: 700;
        }

        /* Section divider */
        .section-divider {
            border: none;
            border-top: 1px solid #2E3A50;
            margin: 32px 0 24px 0;
        }
    </style>
""", unsafe_allow_html=True)

# ==============================
# LOAD MODELS (TorchScript)
# ==============================
@st.cache_resource
def load_model(path):
    model = torch.jit.load(path, map_location="cpu")
    model.eval()
    return model

cnn_model    = load_model("models/cnn_scratch.pt")
rescnn_model = load_model("models/cnn_residual.pt")
resnet_model = load_model("models/transfer_exported.pt")

# ==============================
# PREDICTION HELPER
# ==============================
def predict(image, model):
    img_t = T.ToTensor()(image).unsqueeze(0)
    outputs = model(img_t)
    probs = outputs.data.cpu().numpy().squeeze()
    idxs = np.argsort(probs)[::-1]
    labels = []
    for i in range(5):
        p = probs[idxs[i]]
        landmark_name = model.class_names[idxs[i]]
        labels.append((landmark_name, float(p)))
    return labels

# ==============================
# CLASS NAMES & EXAMPLE IMAGES
# ==============================
CLASS_NAMES = [
    "Haleakala National Park", "Mount Rainier National Park", "Ljubljana Castle",
    "Dead Sea", "Wroclaw's Dwarves", "London Olympic Stadium", "Niagara Falls",
    "Stonehenge", "Grand Canyon", "Golden Gate Bridge", "Edinburgh Castle",
    "Mount Rushmore National Memorial", "Kantanagar Temple", "Yellowstone National Park",
    "Terminal Tower", "Central Park", "Eiffel Tower", "Changdeokgung", "Delicate Arch",
    "Vienna City Hall", "Matterhorn", "Taj Mahal", "Moscow Raceway", "Externsteine",
    "Soreq Cave", "Banff National Park", "Pont du Gard", "Seattle Japanese Garden",
    "Sydney Harbour Bridge", "Petronas Towers", "Brooklyn Bridge",
    "Washington Monument", "Hanging Temple", "Sydney Opera House", "Great Barrier Reef",
    "Monumento a la Revolución", "Badlands National Park", "Atomium", "Forth Bridge",
    "Gateway of India", "Stockholm City Hall", "Machu Picchu",
    "Death Valley National Park", "Gullfoss Falls", "Trevi Fountain",
    "Temple of Heaven", "Great Wall of China", "Prague Astronomical Clock",
    "Whitby Abbey", "Temple of Olympian Zeus",
]

# 5 example images: (display label, public image URL)
# EXAMPLE_IMAGES = [
#     {
#         "label": "Eiffel Tower",
#         "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/8/85/Smiley.svg/200px-Smiley.svg.png",  # placeholder replaced below
#         "real_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/Camponotus_flavomarginatus_ant.jpg/320px-Camponotus_flavomarginatus_ant.jpg",
#     },
# ]

# Reliable public-domain landmark image URLs (Wikimedia Commons)
EXAMPLE_IMAGES = [
    {
        "label": "Eiffel Tower",
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a8/Tour_Eiffel_Wikimedia_Commons.jpg/500px-Tour_Eiffel_Wikimedia_Commons.jpg",
    },
    # {
    #     "label": "Stockholm City Hall",
    #     "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/f/fc/Stockholms_stadshus_February_2026_08.jpg/640px-Stockholms_stadshus_February_2026_08.jpg",
    # },
    # {
    #     "label": "Machu Picchu",
    #     "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/7/71/Machu_Picchu%2C_Per%C3%BA%2C_2015-07-30%2C_DD_47.JPG/640px-Machu_Picchu%2C_Per%C3%BA%2C_2015-07-30%2C_DD_47.JPG",
    # },
    # {
    #     "label": "Stonehenge",
    #     "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/a/aa/Stonehenge_from_the_north.jpg/640px-Stonehenge_from_the_north.jpg",
    # },
    {
        "label": "Golden Gate Bridge",
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c7/Golden_Gate_Bridge_as_seen_from_Marshall%E2%80%99s_Beach%2C_March_2018.jpg/1920px-Golden_Gate_Bridge_as_seen_from_Marshall%E2%80%99s_Beach%2C_March_2018.jpg",
    },
]

@st.cache_data(show_spinner=False)
def fetch_image_from_url(url: str) -> Image.Image | None:
    """Download and return a PIL Image from a URL, or None on failure."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        )
    }
    try:
        resp = requests.get(url, headers=headers, timeout=8)
        resp.raise_for_status()
        return Image.open(BytesIO(resp.content)).convert("RGB")
    except Exception:
        return None

# ==============================
# UI LAYOUT
# ==============================
st.title("🗺️ Landmark Classification App")
st.markdown("Upload a landmark image to compare predictions from three different models.")

# Sidebar info
st.sidebar.header("About this project")
st.sidebar.info("""
This project was built as part of the **AWS ML Engineer Nanodegree (Udacity + AWS)**.  
It compares three models:  
- CNN (Scratch)  
- CNN + Residual Connections  
- Transfer Learning (ResNet34)  

Author: [Ahmed Abdelgelel](https://linkedin.com/in/Azahran0)  
Code: [GitHub Repo](https://github.com/Azahran0/Landmark-classification-tagging-for-social-media)
""")

# ============================================================
# SECTION 1 — DATASET OVERVIEW
# ============================================================
st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
st.subheader("📚 Dataset Overview")

st.markdown("""
<div class="stats-row">
    <div class="stat-pill">Classes: <span>50</span></div>
    <div class="stat-pill">Images per class: <span>~100</span></div>
    <div class="stat-pill">Total images: <span>~5,000</span></div>
    <div class="stat-pill">Task: <span>Multi-class Classification</span></div>
</div>
""", unsafe_allow_html=True)

st.markdown(
    "The model was trained on **50 world-famous landmarks** spanning 6 continents. "
    "Each class contains approximately **100 images** collected from diverse angles, "
    "lighting conditions, and seasons to improve generalization."
)

with st.expander("🗂️ View all 50 class names", expanded=False):
    cards_html = '<div class="class-grid">'
    for i, name in enumerate(CLASS_NAMES):
        cards_html += (
            f'<div class="class-card">'
            f'<span class="class-index">{i:02d}</span>'
            f'<span class="class-name">{name}</span>'
            f'</div>'
        )
    cards_html += "</div>"
    st.markdown(cards_html, unsafe_allow_html=True)

# ============================================================
# SECTION 2 — Model Comaprison Summary
# ============================================================
st.subheader("📈 Model Comaprison Summary")
st.markdown("""
                
    | Model                       | F1-score | Notes                          |
    |-----------------------------|----------|--------------------------------|
    | CNN from Scratch            | 70.88 %  | Baseline model                 |
    | CNN + Residual Connections  | 74.8 %   | Residuals improved accuracy    |
    | Transfer Learning (ResNet34)| 72  %    | Trained only for 50 epochs     |

""")


# ============================================================
# SECTION 3 — TRY AN EXAMPLE IMAGE
# ============================================================
st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
st.subheader("🖼️ Try an Example Image")
st.markdown("Click any thumbnail below to load it into the classifier — or upload your own image further down.")

# Render example image grid
example_cols = st.columns(len(EXAMPLE_IMAGES))
selected_example_image: Image.Image | None = None

for col, example in zip(example_cols, EXAMPLE_IMAGES):
    with col:
        img = fetch_image_from_url(example["url"])
        if img:
            st.image(img, use_container_width=True)
            st.markdown(
                f'<p class="example-label">{example["label"]}</p>',
                unsafe_allow_html=True,
            )
            if st.button(f"Use this →", key=f"btn_{example['label']}"):
                selected_example_image = img
                st.session_state["example_image"] = img
                st.session_state["example_label"] = example["label"]
        else:
            st.warning(f"Could not load {example['label']}")

# ============================================================
# SECTION 4 — UPLOAD OR USE EXAMPLE
# ============================================================
st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
st.subheader("📤 Upload Your Own Image")

uploaded_file = st.file_uploader(
    "Browse an image from your device (JPG / JPEG / PNG)",
    type=["jpg", "jpeg", "png"],
)

# Decide which image to classify
image_to_classify: Image.Image | None = None
image_caption = ""

if uploaded_file:
    image_to_classify = Image.open(uploaded_file).convert("RGB")
    image_caption = "Uploaded Image"
    # Clear any cached example to avoid confusion
    st.session_state.pop("example_image", None)
    st.session_state.pop("example_label", None)
elif "example_image" in st.session_state:
    image_to_classify = st.session_state["example_image"]
    image_caption = f"Example: {st.session_state.get('example_label', '')}"

# ============================================================
# SECTION 5 — MODEL PREDICTIONS
# ============================================================
if image_to_classify:
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.subheader("🤖 Model Predictions")

    st.image(image_to_classify, caption=image_caption, width=380)

    col1, col2, col3 = st.columns(3)

    def render_prediction_col(col, title, model, bar_color):
        with col:
            st.subheader(title)
            preds = predict(image_to_classify, model)
            st.write(f"**Top-1:** {preds[0][0]} ({preds[0][1]*100:.1f}%)")
            fig, ax = plt.subplots(figsize=(4, 2.8))
            fig.patch.set_facecolor("#0E1117")
            ax.set_facecolor("#1C2333")
            ax.barh([p[0] for p in preds], [p[1] * 100 for p in preds], color=bar_color)
            ax.invert_yaxis()
            ax.set_xlabel("Probability (%)", color="#8B949E", fontsize=8)
            ax.tick_params(colors="#C9D1D9", labelsize=7)
            for spine in ax.spines.values():
                spine.set_edgecolor("#2E3A50")
            st.pyplot(fig)
            plt.close(fig)

    render_prediction_col(col1, "CNN (Scratch)",              cnn_model,    "#1f77b4")
    render_prediction_col(col2, "CNN + Residuals",            rescnn_model, "#ff7f0e")
    render_prediction_col(col3, "Transfer Learning (ResNet34)", resnet_model, "#2ca02c")
