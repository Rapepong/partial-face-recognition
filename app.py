"""Streamlit demo: upload a photo or take one with the camera, add a synthetic occlusion, see who it is.

  streamlit run app.py

Needs a trained model in models/ (src/step3_train_evaluate.py). VGG-Face models (M1/M2) also need
weights/rcmalli_vggface_tf_vgg16.h5; without it (e.g. on Streamlit Community Cloud, deploy/) they are hidden
and only the FaceNet models (M3/M4) are offered.
"""
import sys
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import joblib  # noqa: E402
from common import MODELS, UNKNOWN, VARIANT_TH, VARIANTS, WEIGHTS  # noqa: E402
from recognizer import Recognizer, draw  # noqa: E402

st.set_page_config(page_title="Partial Face Recognition", layout="wide")


@st.cache_resource(show_spinner="Loading model ...", max_entries=1)   # one backbone in RAM at a time
def load(path):
    return Recognizer(path)


@st.cache_data
def backbone_of(path):
    return joblib.load(path)["backbone"]


st.title("Face Recognition from Partial Face")
st.caption("MTCNN → Alignment → VGG-Face / FaceNet (frozen) → Softmax · ถ้าความมั่นใจต่ำกว่า τ ระบบตอบ \"ไม่รู้จัก\"")

has_vgg = (WEIGHTS / "rcmalli_vggface_tf_vgg16.h5").exists()
models = [p for p in sorted(MODELS.glob("*.joblib")) if has_vgg or backbone_of(str(p)) != "vggface"]
if not models:
    st.error("ยังไม่มีโมเดลใน models/ — รัน src/step1 ถึง step3 ก่อน (ดู README)")
    st.stop()

with st.sidebar:
    first = ["lfw+custom_M2.joblib", "lfw+custom_M4.joblib"] if has_vgg else ["lfw+custom_M4.joblib"]
    first += ["lfw_M2.joblib", "lfw_M4.joblib"]
    preferred = next((i for n in first for i, p in enumerate(models) if p.name == n), 0)
    path = st.selectbox("Model", models, index=preferred, format_func=lambda p: p.stem)
    rec = load(str(path))
    st.caption(f"{rec.bundle['method_name']} · {len(rec.names)} คน · fit on {rec.bundle['fit_on']}")
    # default: the tau at which step 3 measured that at most 5 % of LFW strangers get a name (as video.py)
    tau0 = rec.bundle.get("tau_strangers_5pct")
    tau = st.slider("เกณฑ์ความมั่นใจ τ", 0.0, 0.99, float(rec.bundle["tau"] if tau0 is None else tau0), 0.05)
    if rec.bundle.get("tau_strangers_5pct") is not None:
        st.caption(f"τ ≥ {rec.bundle['tau_strangers_5pct']:.2f}: คนแปลกหน้าถูกตอบเป็นชื่อไม่เกิน 5 % (step 3)")
    variant = st.radio("จำลองการบดบัง", list(VARIANTS), format_func=lambda v: f"{VARIANT_TH[v]} ({v})")
    source = st.radio("ภาพ", ["อัปโหลดรูป", "กล้อง"], horizontal=True)
    if not has_vgg:
        st.caption("เว็บนี้มีเฉพาะ FaceNet (M3/M4): weights ของ VGG-Face (M1/M2) 580 MB ไม่ได้อยู่บนเซิร์ฟเวอร์")

file = (st.file_uploader("รูปที่มีใบหน้า", type=["jpg", "jpeg", "png", "bmp"]) if source == "อัปโหลดรูป"
        else st.camera_input("ถ่ายรูป"))
if file is None:
    st.info("อัปโหลดรูปหรือถ่ายรูปเพื่อเริ่ม")
    st.stop()

rgb = np.array(ImageOps.exif_transpose(Image.open(file)).convert("RGB"))
with st.spinner("Recognising ..."):
    faces = rec(rgb, variant, tau=tau)

left, right = st.columns([3, 2])
known = [f.name for f in faces if f.name != UNKNOWN]
left.image(draw(rgb[..., ::-1].copy(), faces)[..., ::-1],
           caption=f"พบ {len(faces)} ใบหน้า · รู้จัก {len(known)} · ไม่รู้จัก (Unknown) {len(faces) - len(known)}",
           width="stretch")
if not faces:
    right.warning("MTCNN ไม่พบใบหน้าในรูปนี้")
for i, f in enumerate(faces, 1):
    with right.container(border=True):
        c1, c2 = st.columns([1, 2])
        c1.image(f.face, caption=f"ใบหน้า {i} ({variant})", width="stretch")
        name = "ไม่รู้จัก (Unknown)" if f.name == UNKNOWN else f.name.replace("_", " ")
        c2.markdown(f"### {name}")
        c2.markdown(f"ความมั่นใจ **{f.prob:.1%}** (τ = {tau:.2f})")
        c2.dataframe({"Top-5": [n.replace("_", " ") for n, _ in f.top], "p": [p for _, p in f.top]},
                     column_config={"p": st.column_config.ProgressColumn("p", min_value=0.0, max_value=1.0,
                                                                         format="%.2f")},
                     hide_index=True, width="stretch")
