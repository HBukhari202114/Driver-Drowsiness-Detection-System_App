"""
Driver Drowsiness Detection — Streamlit frontend (FYP I aligned).
Explicit camera consent, live EAR monitoring, audible alerts.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import av
import cv2
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer
from streamlit_webrtc.webrtc import VideoTransformerBase

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from drowsiness_core import (  # noqa: E402
    DEFAULT_ALARM_PATH,
    DrowsinessEngine,
    play_alarm_async,
    reset_alarm_gate,
)

st.set_page_config(
    page_title="Driver Drowsiness Detection | KFUEIT",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #0d9488 100%);
        padding: 1.75rem 2rem;
        border-radius: 16px;
        color: #f8fafc;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 40px rgba(15, 23, 42, 0.25);
    }
    .main-header h1 { margin: 0; font-size: 1.85rem; font-weight: 700; }
    .main-header p { margin: 0.5rem 0 0; opacity: 0.9; font-size: 1rem; }
    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1rem 1.25rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
    .status-alert { color: #059669; font-weight: 700; font-size: 1.1rem; }
    .status-drowsy { color: #dc2626; font-weight: 700; font-size: 1.1rem; animation: pulse 1s infinite; }
    @keyframes pulse { 50% { opacity: 0.6; } }
    .consent-box {
        background: #f1f5f9;
        border: 2px dashed #94a3b8;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
    }
    div[data-testid="stMetricValue"] { font-size: 1.4rem; }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

if "camera_consent" not in st.session_state:
    st.session_state.camera_consent = False
if "engine" not in st.session_state:
    st.session_state.engine = DrowsinessEngine()
if "last_alert_time" not in st.session_state:
    st.session_state.last_alert_time = 0.0


class StreamlitDrowsinessProcessor(VideoTransformerBase):
    def __init__(self) -> None:
        self.engine = st.session_state.engine
        self._alert_cooldown = 4.0

    def transform(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = self.engine.process_bgr_frame(img)
        if self.engine.metrics.status == "DROWSY":
            now = time.time()
            if now - st.session_state.last_alert_time >= self._alert_cooldown:
                st.session_state.last_alert_time = now
                self.engine.metrics.alert_count += 1
                play_alarm_async(DEFAULT_ALARM_PATH)
        else:
            reset_alarm_gate()
        return av.VideoFrame.from_ndarray(img, format="bgr24")


st.markdown(
    """
    <div class="main-header">
        <h1>Driver Drowsiness Detection System</h1>
        <p>Real-time EAR &amp; yawning analysis · OpenCV · MediaPipe · FYP I (KFUEIT)</p>
    </div>
    """,
    unsafe_allow_html=True,
)

col_side, col_main = st.columns([1, 2.2], gap="large")

with col_side:
    st.subheader("Live telemetry")
    m = st.session_state.engine.metrics
    status_class = "status-drowsy" if m.status == "DROWSY" else "status-alert"
    st.markdown(
        f'<p class="{status_class}">Status: {m.status}</p>',
        unsafe_allow_html=True,
    )
    st.metric("Eye Aspect Ratio (EAR)", f"{m.ear:.3f}")
    st.metric("Mouth opening (MAR)", f"{m.mar:.3f}")
    st.metric("Blink count", m.blink_count)
    st.metric("Alerts triggered", m.alert_count)
    st.caption(f"Last update: {m.last_updated or '—'}")

    st.divider()
    st.markdown("**How it works (FYP methodology)**")
    st.markdown(
        """
        1. Webcam captures live video  
        2. Face & eye landmarks are tracked  
        3. **EAR** detects prolonged eye closure  
        4. **MAR** detects yawning  
        5. Alarm plays when drowsiness is detected  
        """
    )

with col_main:
    if not st.session_state.camera_consent:
        st.markdown('<div class="consent-box">', unsafe_allow_html=True)
        st.markdown("### Camera access required")
        st.markdown(
            "This system needs your **webcam** to monitor eye state and detect drowsiness, "
            "as described in the FYP requirements (FR-01 video capture, FR-07 visualization)."
        )
        st.info(
            "When you click **Allow camera access**, your browser will show its own permission "
            "dialog. You must choose **Allow** there to start detection."
        )
        c1, c2, _ = st.columns([1, 1, 2])
        with c1:
            if st.button("Allow camera access", type="primary", use_container_width=True):
                st.session_state.camera_consent = True
                st.rerun()
        with c2:
            if st.button("Not now", use_container_width=True):
                st.warning("Detection stays off until you allow camera access.")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown("#### Live monitoring")
        if st.button("Revoke camera & stop", type="secondary"):
            st.session_state.camera_consent = False
            st.session_state.engine = DrowsinessEngine()
            st.rerun()

        ctx = webrtc_streamer(
            key="drowsiness-detection",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=StreamlitDrowsinessProcessor,
            media_stream_constraints={
                "video": {
                    "width": {"ideal": 640},
                    "height": {"ideal": 480},
                    "facingMode": "user",
                },
                "audio": False,
            },
            async_processing=True,
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        )

        if ctx.state.playing:
            st.success("Camera active — monitoring in progress. Press **Stop** on the player to pause.")
        elif st.session_state.camera_consent:
            st.warning(
                "Click **START** on the video widget above, then allow the browser camera prompt if shown."
            )

        metrics_placeholder = st.empty()

        @st.fragment(run_every=1)
        def refresh_metrics() -> None:
            m2 = st.session_state.engine.metrics
            with metrics_placeholder.container():
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("EAR", f"{m2.ear:.3f}")
                c2.metric("Face", "Yes" if m2.face_detected else "No")
                c3.metric("Closed frames", m2.closed_frames)
                c4.metric("Blinks", m2.blink_count)

        refresh_metrics()

st.divider()
with st.expander("Installation & dlib desktop mode"):
    st.markdown(
        f"""
        **Web app (this page):** `streamlit run streamlit_app/streamlit_app.py`  
        **Desktop launcher:** `python main.py`  
        **Classic dlib blink script:** requires `models/shape_predictor_68_face_landmarks.dat`  
        Alarm file: `{DEFAULT_ALARM_PATH.name}` at project root.
        """
    )
