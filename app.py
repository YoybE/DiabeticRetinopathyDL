import streamlit as st
from st_files_connection import FilesConnection
import torch
import torchvision.transforms as transforms
from PIL import Image
from utils.loader import import_dataset
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(
    page_title="Diabetic Retinopathy Classifier",
    page_icon="👁",
    layout="wide",
)

CLASSIFIERS_DIR = "./classifiers"
DATASET_DIR = "./dataset"
CLASS_NAMES = ["Healthy", "Severe DR"]
CLASS_FOLDERS = ["Healthy", "Severe DR"]
N_SAMPLES = 8

MODEL_INFO = {
    "UNetClassifier": (
        "**Baseline U-Net** — Standard encoder-decoder with max-pooling downsampling, "
        "transposed convolution upsampling, and skip connections that concatenate encoder "
        "feature maps into the decoder."
    ),
    "AUNetClassifier": (
        "**Attention U-Net** — Extends the baseline U-Net by placing Attention Gates (AG) "
        "on skip connections. Each gate uses additive attention to suppress irrelevant "
        "spatial features before they reach the decoder."
    ),
    "ResUNetClassifier": (
        "**Residual U-Net** — Replaces standard double-conv blocks with residual blocks "
        "(identity shortcut connections). Helps stabilise gradient flow in deeper networks."
    ),
    "AResUNetClassifier": (
        "**Attention + Residual U-Net** — Combines Attention Gates on skip connections with "
        "residual convolutional blocks. Best of both attention-filtering and residual learning."
    ),
    "EfficientNetB0Classifier": (
        "**EfficientNet-B0** — Pretrained on ImageNet (transfer learning). Classifier head "
        "replaced with a linear layer for binary classification. No UNet decoder; "
        "segmentation map is not available."
    ),
    "NoSkipUNetClassifier": (
        "**No-Skip U-Net** — Ablation variant that removes all skip connections. The decoder "
        "upsamples purely from the bottleneck, showing the importance of skip connections."
    ),
}

# upload models from gcs
@st.cache_resource
def get_models_gcs():
    if not os.path.exists(CLASSIFIERS_DIR):
        conn = st.connection('gcs', type=FilesConnection)
        conn.fs.get("50039-dbrtpydl/classifiers", CLASSIFIERS_DIR, recursive=True)


#import dataset locally w/o uploading to github
@st.cache_resource
def get_dataset_kaggle():
    import_dataset()


def get_available_models() -> dict:
    models = {}
    get_models_gcs()
    if os.path.exists(CLASSIFIERS_DIR):
        for fname in sorted(os.listdir(CLASSIFIERS_DIR)):
            if fname.endswith(".pt"):
                name = fname.split("_")[0]
                models[name] = os.path.join(CLASSIFIERS_DIR, fname)
    return models


@st.cache_resource
def load_model(model_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.eval()
    return model, device


@st.cache_resource
def load_all_models():
    available = get_available_models()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded = {}
    for name, path in available.items():
        m = torch.load(path, map_location=device, weights_only=False)
        m.eval()
        loaded[name] = m
    return loaded, device


def get_class_samples(class_folder: str, n: int = None) -> list:
    cls_dir = os.path.join(DATASET_DIR, class_folder)
    if not os.path.exists(cls_dir):
        return []
    files = sorted(f for f in os.listdir(cls_dir) if f.lower().endswith(".png"))
    return files[:n] if n is not None else files


@st.cache_resource
def get_test_file_paths() -> dict:
    """Replays the deterministic seed-42 split and returns test-set file paths per class."""
    if not os.path.exists(DATASET_DIR):
        return {}
    import torchvision
    full_dataset = torchvision.datasets.DatasetFolder(
        root=DATASET_DIR,
        loader=torchvision.datasets.folder.default_loader,
        transform=transforms.ToTensor(),
        extensions=[".png"],
    )
    targets = full_dataset.targets
    result = {cls: [] for cls in full_dataset.classes}
    for c, cls_name in enumerate(full_dataset.classes):
        class_indices = [i for i, t in enumerate(targets) if t == c]
        torch.manual_seed(42)
        perm = torch.randperm(len(class_indices)).tolist()
        r = len(class_indices)
        a = int(r * 0.8)
        b = int(r * 0.9)
        for p in perm[a:b]:
            path, _ = full_dataset.samples[class_indices[p]]
            result[cls_name].append(path)
    return result


def run_inference(model, image: Image.Image, device):
    w, h = image.size
    def _bad(n): return n % 8 != 0 #incase the uploaded image dimension differ
    if _bad(w) or _bad(h):
        new_w = max(8, round(w / 8) * 8) if _bad(w) else w
        new_h = max(8, round(h / 8) * 8) if _bad(h) else h
        image = image.resize((new_w, new_h), Image.BILINEAR)
    tensor = transforms.ToTensor()(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)[0].cpu().numpy()
    pred = int(np.argmax(probs))
    unet_out = None
    if hasattr(model, "unet_output") and model.unet_output is not None:
        raw = model.unet_output.detach().cpu()[0]   # (C, H, W)
        arr = (raw[1] - raw[0]).numpy()             # DR score minus Healthy score
        lo, hi = arr.min(), arr.max()
        unet_out = (arr - lo) / (hi - lo + 1e-8)   # float [0, 1]
        model.unet_output = None
    return pred, probs, unet_out


def image_source_selector(key_prefix: str):
    get_dataset_kaggle()
    source = st.radio(
        "Image source",
        ["Upload", "Dataset sample", "Test set sample"],
        horizontal=True,
        key=f"{key_prefix}_source",
    )

    img, label = None, None

    if source == "Upload":
        uploaded = st.file_uploader(
            "Choose a retinal fundus image",
            type=["png", "jpg", "jpeg"],
            key=f"{key_prefix}_upload",
        )
        if uploaded:
            img = Image.open(uploaded).convert("RGB")
            label = uploaded.name

    elif source == "Dataset sample":
        cls = st.radio(
            "Class",
            ["Healthy", "Severe DR"],
            horizontal=True,
            key=f"{key_prefix}_ds_cls",
        )
        n_ds = st.session_state.get("samples_n", N_SAMPLES)
        samples = get_class_samples(cls, n_ds)
        if samples:
            chosen = st.selectbox(f"Image", samples, key=f"{key_prefix}_ds_sel")
            img = Image.open(os.path.join(DATASET_DIR, cls, chosen)).convert("RGB")
            label = chosen
        else:
            st.warning(f"No images found in `./dataset/{cls}/`.")

    else:  # Test set sample
        cls = st.radio(
            "Class",
            ["Healthy", "Severe DR"],
            horizontal=True,
            key=f"{key_prefix}_ts_cls",
        )
        test_paths = get_test_file_paths()
        paths = test_paths.get(cls, [])
        if paths:
            fnames = [os.path.basename(p) for p in paths]
            chosen_idx = st.selectbox(
                "Image",
                range(len(fnames)),
                format_func=lambda i: fnames[i],
                key=f"{key_prefix}_ts_sel",
            )
            img = Image.open(paths[chosen_idx]).convert("RGB")
            label = fnames[chosen_idx]
        else:
            st.warning(f"Test set not available. Make sure `./dataset/` exists.")

    return img, label


def _normalize(arr: np.ndarray) -> np.ndarray:
    lo, hi = arr.min(), arr.max()
    return (arr - lo) / (hi - lo + 1e-8)


def render_compact_heatmap(original: Image.Image, heatmap: np.ndarray):
    fig, axes = plt.subplots(1, 2, figsize=(5, 2.2))
    axes[0].imshow(np.array(original))
    axes[0].set_title("Input", fontsize=7)
    axes[0].axis("off")
    axes[1].imshow(heatmap)
    axes[1].set_title("Seg. Map", fontsize=7)
    axes[1].axis("off")
    plt.tight_layout(pad=0.4)
    st.pyplot(fig)
    plt.close(fig)


#session state
for key in ("main_result", "compare_results", "_main_model"):
    if key not in st.session_state:
        st.session_state[key] = None

#sidebar
st.sidebar.title("Diabetic Retinopathy")
st.sidebar.caption("Binary classification of retinal fundus images")
_device = "cuda" if torch.cuda.is_available() else "cpu"
st.sidebar.info(f"Device: `{_device}`")

available = get_available_models()
if not available:
    st.error("No pre-trained models found in `./classifiers/`. Add `.pt` files to that directory.")
    st.stop()

#two types of tabs: main and cpmparison
main_tab, compare_tab = st.tabs(["Main", "Comparison"])

#main tab
with main_tab:
    sub_predict, sub_samples, sub_info = st.tabs(["Predict", "Dataset Samples", "Model Info"])

    # predict sub tab
    with sub_predict:
        st.subheader("Predict")

        chosen_name = st.selectbox("Model", list(available.keys()), key="main_model_sel")
        st.caption(MODEL_INFO.get(chosen_name, ""))

        # Clear stale result if model changed
        if st.session_state._main_model != chosen_name:
            st.session_state.main_result = None
            st.session_state._main_model = chosen_name

        model, device = load_model(available[chosen_name])

        col_in, col_out = st.columns(2, gap="large")

        with col_in:
            img, _ = image_source_selector("main")
            if img is not None:
                st.image(img, width="stretch")
                if st.button("Classify", type="primary", use_container_width=True, key="main_btn"):
                    with st.spinner("Running inference…"):
                        pred, probs, unet_out = run_inference(model, img, device)
                    st.session_state.main_result = {
                        "pred": pred, "probs": probs,
                        "unet_out": unet_out, "img": img,
                    }

        with col_out:
            st.subheader("Result")
            r = st.session_state.main_result
            if r:
                pred, probs, unet_out = r["pred"], r["probs"], r["unet_out"]
                label_str = CLASS_NAMES[pred]
                conf = float(probs[pred])

                if pred == 0:
                    st.success(f"### {label_str}")
                else:
                    st.error(f"### {label_str} — Anomaly Detected")

                st.metric("Confidence", f"{conf * 100:.1f}%")
                st.markdown("**Class probabilities**")
                for i, cname in enumerate(CLASS_NAMES):
                    p = float(probs[i])
                    st.progress(p, text=f"{cname}: {p * 100:.1f}%")

                if unet_out is not None:
                    st.markdown("---")
                    st.markdown("**Segmentation map**")
                    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
                    axes[0].imshow(np.array(r["img"]))
                    axes[0].set_title("Original")
                    axes[0].axis("off")
                    axes[1].imshow(unet_out)
                    axes[1].set_title("UNet Output")
                    axes[1].axis("off")
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
                else:
                    st.info("No segmentation map for EfficientNet-B0.")
            else:
                st.info("Select an image and click **Classify**.")

    # -----------------------------------------------------------------------
    # Dataset Samples
    # -----------------------------------------------------------------------
    with sub_samples:
        st.subheader("Dataset Samples")

        if not os.path.exists(DATASET_DIR):
            st.warning("Dataset not found at `./dataset/`.")
        else:
            classes = sorted(
                d for d in os.listdir(DATASET_DIR)
                if os.path.isdir(os.path.join(DATASET_DIR, d))
            )
            n = st.slider("Samples per class", 1, 20, N_SAMPLES, key="samples_n")

            for cls in classes:
                st.markdown(f"#### {cls}")
                cls_dir = os.path.join(DATASET_DIR, cls)
                files = sorted(
                    f for f in os.listdir(cls_dir) if f.lower().endswith(".png")
                )[:n]
                cols = st.columns(len(files))
                for i, fname in enumerate(files):
                    with cols[i]:
                        st.image(
                            os.path.join(cls_dir, fname),
                            caption=fname,
                            width="stretch",
                        )

  
    # model info
    with sub_info:
        st.subheader("Architecture Overview")

        for name, desc in MODEL_INFO.items():
            status = "available" if name in available else "not found"
            color = "green" if name in available else "red"
            with st.expander(f"{name}  —  :{color}[{status}]"):
                st.markdown(desc)

        st.markdown("---")
        st.markdown(
            """
### About the Project

Binary classification of diabetic retinopathy from retinal fundus photographs,
exploring U-Net variants as the primary architecture.

| Property | Value |
|---|---|
| Task | Healthy vs Severe DR |
| Dataset | ~1 190 fundus images (Kaggle) |
| Optimiser | Adam, weight_decay = 1e-5 |
| Loss | CrossEntropyLoss |
| Epochs | 20 |
| Learning rate | 0.001 |
| Batch size | 32 |
| Split | 80 % train / 10 % val / 10 % test |
| Metrics | Accuracy, F1, F2, Anomaly Detection Rate |
"""
        )

#comparison tab - showcases all the models based on the input 
with compare_tab:
    st.subheader("Compare All Models")
    st.caption(
        "Run the same image through every trained classifier and see "
        "predictions and segmentation maps side by side."
    )

    cmp_img, cmp_label = image_source_selector("cmp")

    if cmp_img is not None:
        col_prev, col_run = st.columns([1, 2])
        with col_prev:
            st.image(cmp_img, caption=cmp_label or "Input", width="stretch")
        with col_run:
            st.markdown(f"**{len(available)} models** will be evaluated.")
            if st.button("Compare All Models", type="primary", key="cmp_btn"):
                all_models, all_device = load_all_models()
                results = {}
                progress = st.progress(0, text="Starting…")
                total = len(all_models)
                for idx, (mname, mmodel) in enumerate(all_models.items()):
                    progress.progress((idx) / total, text=f"Running {mname}…")
                    pred, probs, unet_out = run_inference(mmodel, cmp_img, all_device)
                    results[mname] = {"pred": pred, "probs": probs, "unet_out": unet_out}
                progress.progress(1.0, text="Done!")
                st.session_state.compare_results = {"results": results, "img": cmp_img}

    cmp_state = st.session_state.compare_results
    if cmp_state:
        st.markdown("---")
        results = cmp_state["results"]
        orig_img = cmp_state["img"]

        cols = st.columns(3, gap="medium")
        for i, (mname, r) in enumerate(results.items()):
            pred = r["pred"]
            probs = r["probs"]
            unet_out = r["unet_out"]
            label_str = CLASS_NAMES[pred]
            conf = float(probs[pred])

            with cols[i % 3]:
                with st.container(border=True):
                    st.markdown(f"**{mname}**")

                    if pred == 0:
                        st.success(f"{label_str} &nbsp; `{conf * 100:.1f}%`")
                    else:
                        st.error(f"{label_str} &nbsp; `{conf * 100:.1f}%`")

                    for j, cname in enumerate(CLASS_NAMES):
                        p = float(probs[j])
                        st.progress(p, text=f"{cname}: {p * 100:.1f}%")

                    if unet_out is not None:
                        render_compact_heatmap(orig_img, unet_out)
                    else:
                        st.caption("No segmentation map (EfficientNet-B0).")
    elif cmp_img is None:
        st.info("Select an image above and click **Compare All Models** to see results.")
