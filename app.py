"""
Streamlit app: Microplastics Detection in Water Samples
- Upload image(s)
- Local CV pipeline to detect particles, estimate size & count
- Optional: send images to Google Gemini (Vision) for advanced image understanding (requires API key)
- Outputs: per-image gallery, detected contours overlay, size/quantity table, and charts

Notes:
- For accurate size estimates you MUST provide a scale (mm per pixel) or include a reference object in images.
- This app attempts reasonable defaults but results are approximate.

Requirements:
pip install streamlit opencv-python-headless matplotlib numpy pillow pandas google-genai

Run:
streamlit run streamlit_microplastics_detector.py

"""

import os
import io
import base64
from typing import List, Tuple, Dict

import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import cv2
import matplotlib.pyplot as plt
from dotenv import load_dotenv

load_dotenv()

# Optional: Google Gen AI (Gemini) client. If you want to enable Gemini, set GEMINI_API_KEY env var
# and ensure the google-genai SDK is installed. This import is wrapped so the app still runs without it.
try:
    from google import gen_ai  # newer package name in some docs
    GEMINI_AVAILABLE = True
except Exception:
    try:
        # fallback import names used in various examples
        import google.generativeai as genai
        GEMINI_AVAILABLE = True
    except Exception:
        genai = None
        GEMINI_AVAILABLE = False

st.set_page_config(page_title="Microplastics Detector", layout="wide")

st.title("Microplastics Detector — Image Upload + Analysis")
st.markdown("""
Upload microscope images of your water sample. The app will run a local computer-vision pipeline to detect small particles (candidate microplastics), estimate their sizes and counts, and produce a downloadable report with plots. Optionally you can enable Google Gemini (Vision) to get an AI-assisted description/classification of each particle.

**Important:** For reliable size measurements include a scale bar or enter `mm per pixel` for your imaging setup. Without that, sizes will be reported in *pixels* and are approximate.
""")

# Sidebar configuration
st.sidebar.header("Settings")
mm_per_pixel = st.sidebar.number_input("Scale: mm per pixel (enter if known)", min_value=0.0, value=0.0, step=0.0001, format="%.6f")
min_area_px = st.sidebar.number_input("Minimum area filter (pixels)", min_value=1, value=20, step=1)
apply_median = st.sidebar.checkbox("Apply median blur before threshold", value=True)
use_gemini = st.sidebar.checkbox("Use Google Gemini to analyze images (optional)", value=False)
if use_gemini and not GEMINI_AVAILABLE:
    st.sidebar.error("Gemini SDK not installed — proceed with local CV only. To enable, install google-genai or google-generativeai.")

if use_gemini:
    st.sidebar.markdown("Make sure you have set your Google API key in the environment variable `GEMINI_API_KEY` or provide below.")
    provided_gemini_key = st.sidebar.text_input("Google API Key (leave empty to read from env)", type="password")
else:
    provided_gemini_key = None

uploaded_files = st.file_uploader("Upload one or more microscope images", accept_multiple_files=True, type=["png","jpg","jpeg","tif","tiff"])

# Utility functions

def read_image(file) -> np.ndarray:
    img = Image.open(file).convert('RGB')
    return np.array(img)


def detect_particles(img_rgb: np.ndarray, min_area_px: int=20, apply_median: bool=True) -> Tuple[np.ndarray, List[Dict]]:
    """Detect candidate particles using classical CV.
    Returns overlay image and list of detections: {area_px, contour, bbox, centroid}
    """
    img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    if apply_median:
        img = cv2.medianBlur(img, 3)
    # Adaptive threshold — works well for variable illumination
    thr = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, 51, 7)
    # Morphological cleaning
    kernel = np.ones((3,3), np.uint8)
    thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    overlay = img_rgb.copy()
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area_px:
            continue
        x,y,w,h = cv2.boundingRect(c)
        M = cv2.moments(c)
        if M['m00'] != 0:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
        else:
            cx,cy = x + w//2, y + h//2
        detections.append({
            'area_px': float(area),
            'bbox': (x,y,w,h),
            'centroid': (cx,cy),
            'contour': c
        })
        # draw
        cv2.drawContours(overlay, [c], -1, (255,0,0), 1)
        cv2.rectangle(overlay, (x,y), (x+w, y+h), (0,255,0), 1)
        cv2.circle(overlay, (cx,cy), 2, (255,255,0), -1)
    return overlay, detections


def area_to_diameter_px(area_px: float) -> float:
    """Estimate equivalent diameter (pixels) assuming circular particle"""
    return 2.0 * np.sqrt(area_px / np.pi)


def detections_to_dataframe(detections: List[Dict], mm_per_pixel: float=0.0) -> pd.DataFrame:
    rows = []
    for i, d in enumerate(detections, start=1):
        dia_px = area_to_diameter_px(d['area_px'])
        dia_mm = dia_px * mm_per_pixel if mm_per_pixel and mm_per_pixel>0 else np.nan
        area_mm2 = (d['area_px'] * (mm_per_pixel**2)) if mm_per_pixel and mm_per_pixel>0 else np.nan
        rows.append({
            'id': i,
            'area_px': float(d['area_px']),
            'diameter_px': float(dia_px),
            'diameter_mm': float(dia_mm) if not np.isnan(dia_mm) else None,
            'area_mm2': float(area_mm2) if not np.isnan(area_mm2) else None,
            'centroid_x': int(d['centroid'][0]),
            'centroid_y': int(d['centroid'][1])
        })
    return pd.DataFrame(rows)


# Optional Gemini helper (best-effort stub). Real usage requires correct SDK and permissions.
def analyze_with_gemini(image_bytes: bytes, api_key: str) -> Dict:
    """
    Send image bytes to Gemini Vision for analysis. This is a best-effort wrapper —
    you may need to adapt it to your organisation's Google Gen AI client.
    Returns the parsed JSON-like response from the API or an error dict.
    """
    if not GEMINI_AVAILABLE:
        return {'error': 'Gemini SDK not available in environment.'}
    try:
        # Example: using google.generativeai (older) or google-genai (newer). Adjust based on your SDK.
        # The actual method names differ between SDK versions. This function intentionally keeps
        # a generic structure and should be modified to match the SDK you install.
        if 'genai' in globals() and getattr(genai, '__name__', '').startswith('google'):
            client = genai.Client(api_key=api_key)
            # call a hypothetical vision analyze method — please adapt per your SDK docs
            resp = client.images.analyze(image_bytes=image_bytes, features=['SEGMENTATION','OBJECT_DETECTION'])
            return resp
        else:
            # other SDKs
            return {'error': 'Unknown Gemini client installed; adapt analyze_with_gemini() to the SDK.'}
    except Exception as e:
        return {'error': str(e)}


# Main processing
if uploaded_files:
    all_results = []
    for file in uploaded_files:
        st.header(f"Image: {file.name}")
        img_np = read_image(file)
        st.image(img_np, caption=f"Original: {file.name}", use_column_width=True)

        overlay, detections = detect_particles(img_np, min_area_px=min_area_px, apply_median=apply_median)
        df = detections_to_dataframe(detections, mm_per_pixel=mm_per_pixel)

        st.subheader("Detections")
        st.write(f"Total candidates detected: {len(df)}")
        if not df.empty:
            st.dataframe(df[['id','area_px','diameter_px','diameter_mm']])

        st.subheader("Overlay (detections)")
        st.image(overlay, use_column_width=True)

        # Charts
        st.subheader("Size distribution")
        fig, ax = plt.subplots()
        if not df.empty:
            ax.hist(df['diameter_px'].dropna(), bins=20)
            ax.set_xlabel('Diameter (px)')
            if mm_per_pixel and mm_per_pixel>0:
                ax.set_title('Diameter distribution (pixels). Convert to mm using provided scale')
            else:
                ax.set_title('Diameter distribution (pixels). Provide mm/pixel for mm values')
        else:
            ax.text(0.5, 0.5, 'No detections to plot', ha='center')
        st.pyplot(fig)

        # Advanced: call Gemini if requested
        gemini_resp = None
        if use_gemini and (provided_gemini_key or os.getenv('GEMINI_API_KEY')):
            api_key = provided_gemini_key if provided_gemini_key else os.getenv('GEMINI_API_KEY')
            try:
                file.seek(0)
                image_bytes = file.read()
                gemini_resp = analyze_with_gemini(image_bytes, api_key)
                st.subheader('Gemini analysis (raw)')
                st.write(gemini_resp)
            except Exception as e:
                st.error(f'Gemini analysis failed: {e}')
        elif use_gemini:
            st.warning('Gemini requested but no API key supplied. Skipping Gemini analysis.')

        # Add to consolidated results
        all_results.append({
            'filename': file.name,
            'num_detections': len(df),
            'df': df,
            'gemini': gemini_resp
        })

    # Summary across all images
    st.header('Summary report')
    summary_rows = []
    for r in all_results:
        mean_d_px = float(r['df']['diameter_px'].mean()) if not r['df'].empty else np.nan
        summary_rows.append({'filename': r['filename'], 'num_detections': r['num_detections'], 'mean_diameter_px': mean_d_px})
    summary_df = pd.DataFrame(summary_rows)
    st.dataframe(summary_df)

    # Combined histogram
    st.subheader('Combined size distribution across all images')
    all_diams = np.concatenate([r['df']['diameter_px'].values for r in all_results if not r['df'].empty]) if any([not r['df'].empty for r in all_results]) else np.array([])
    fig2, ax2 = plt.subplots()
    if all_diams.size:
        ax2.hist(all_diams, bins=30)
        ax2.set_xlabel('Diameter (px)')
        st.pyplot(fig2)
    else:
        ax2.text(0.5, 0.5, 'No particle diameters to plot', ha='center')
        st.pyplot(fig2)

    # Exportable report
    st.subheader('Download results')
    # Create CSV summary
    csv_buf = io.StringIO()
    combined_csv = pd.concat([r['df'].assign(filename=r['filename']) for r in all_results if not r['df'].empty], ignore_index=True) if any([not r['df'].empty for r in all_results]) else pd.DataFrame()
    if not combined_csv.empty:
        combined_csv.to_csv(csv_buf, index=False)
        b64 = base64.b64encode(csv_buf.getvalue().encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="microplastics_report.csv">Download CSV report</a>'
        st.markdown(href, unsafe_allow_html=True)
    else:
        st.write('No detections — nothing to download')

else:
    st.info('Upload images to begin analysis.')

# Footer / help
st.markdown('''
---
**Notes & next steps**:
- This app uses a simple threshold+contour pipeline for particle detection. For higher accuracy, consider training a dedicated object detection/segmentation model (Vertex AI Vision / Roboflow / Detectron2 / YOLO) on labeled microplastics images.
- If you enable Gemini, adapt the `analyze_with_gemini()` function to match the SDK your environment uses (google-genai, google.generativeai, or REST calls to Vertex AI).
- For best results provide a physical scale marker in each image or the exact mm/pixel value from your imaging system.
''')
