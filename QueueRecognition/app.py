import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np


# =========================================================
# PAGE SETTINGS
# =========================================================

st.set_page_config(
    page_title="QueueVision AI",
    page_icon="👥",
    layout="wide"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.stApp {
    background:
        radial-gradient(
            circle at 10% 10%,
            rgba(0, 200, 255, 0.12),
            transparent 30%
        ),
        radial-gradient(
            circle at 90% 90%,
            rgba(80, 0, 255, 0.12),
            transparent 30%
        ),
        linear-gradient(
            135deg,
            #07111f,
            #0b1728,
            #07101c
        );

    color: white;
}


.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1200px;
}


/* =========================================================
   UPLOAD BOX
   ========================================================= */

[data-testid="stFileUploader"] {
    background: rgba(255, 255, 255, 0.04);
    border: 1px dashed rgba(32, 207, 255, 0.5);
    border-radius: 18px;
    padding: 15px;
}


/* =========================================================
   METRIC CARDS
   ========================================================= */

.metric-card {
    background:
        linear-gradient(
            145deg,
            rgba(255, 255, 255, 0.08),
            rgba(255, 255, 255, 0.025)
        );

    border: 1px solid rgba(255, 255, 255, 0.12);

    border-radius: 20px;

    padding: 25px;

    text-align: center;

    box-shadow:
        0 10px 30px rgba(0, 0, 0, 0.30);

    min-height: 150px;
}


.metric-title {
    color: #9db4c8;
    font-size: 14px;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 12px;
}


.metric-value {
    color: #20cfff;
    font-size: 36px;
    font-weight: 800;
}


/* =========================================================
   BUTTON
   ========================================================= */

.stButton > button {
    width: 100%;

    border-radius: 12px;

    border: none;

    background:
        linear-gradient(
            90deg,
            #08b8e8,
            #4169ff
        );

    color: white;

    font-size: 17px;

    font-weight: 700;

    padding: 12px;

    transition: 0.3s;
}


.stButton > button:hover {
    transform: translateY(-2px);

    box-shadow:
        0 8px 25px rgba(32, 207, 255, 0.35);
}


/* =========================================================
   HEADINGS
   ========================================================= */

h1,
h2,
h3 {
    color: #eaf7ff !important;
}


</style>
""", unsafe_allow_html=True)


# =========================================================
# HEADER
# =========================================================

st.html("""
<div style="
    background: linear-gradient(
        135deg,
        rgba(20,184,255,0.15),
        rgba(80,60,255,0.12)
    );

    border: 1px solid rgba(80,200,255,0.25);

    border-radius: 24px;

    padding: 35px;

    margin-bottom: 30px;

    box-shadow:
        0 15px 50px rgba(0,0,0,0.35);
">

    <div style="
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 8px;
        color: white;
    ">

        👥 <span style="color:#20cfff;">
            QueueVision
        </span> AI

    </div>


    <div style="
        color:#a9bdd0;
        font-size:17px;
    ">

        Smart People Detection and Queue Monitoring
        using Computer Vision

    </div>

</div>
""")


# =========================================================
# LOAD YOLO MODEL
# =========================================================

@st.cache_resource
def load_model():

    return YOLO("yolo11n.pt")


model = load_model()


# =========================================================
# UPLOAD SECTION
# =========================================================

st.markdown("### 📸 Upload Queue Image")


uploaded_file = st.file_uploader(
    "Choose an image containing people",
    type=["jpg", "jpeg", "png"]
)


# =========================================================
# IMAGE PROCESSING
# =========================================================

if uploaded_file is not None:

    image = Image.open(uploaded_file)


    # Show original image

    st.image(
        image,
        caption="Uploaded Queue Image",
        use_container_width=True
    )


    st.write("")


    # =====================================================
    # ANALYZE BUTTON
    # =====================================================

    if st.button("🔍 ANALYZE QUEUE"):

        with st.spinner("AI is detecting people..."):

            # Convert image

            image_array = np.array(image)


            # YOLO detection

            results = model(image_array)


            # First result

            result = results[0]


            # =================================================
            # COUNT PEOPLE
            # =================================================

            person_count = 0


            if result.boxes is not None:

                for cls in result.boxes.cls:

                    # YOLO class 0 = person

                    if int(cls) == 0:

                        person_count += 1


            # =================================================
            # DRAW BOXES
            # =================================================

            detected_image = result.plot()


        # =====================================================
        # QUEUE STATUS
        # =====================================================

        if person_count == 0:

            status = "NO QUEUE"

        elif person_count <= 5:

            status = "LOW"

        elif person_count <= 10:

            status = "MEDIUM"

        else:

            status = "HIGH"


        # =====================================================
        # ANALYSIS
        # =====================================================

        st.markdown("### 📊 Queue Analysis")


        col1, col2 = st.columns(2)


        # =====================================================
        # PEOPLE COUNT
        # =====================================================

        with col1:

            st.html(f"""
<div class="metric-card">

    <div class="metric-title">
        People Detected
    </div>

    <div class="metric-value">
        {person_count}
    </div>

</div>
""")


        # =====================================================
        # QUEUE STATUS
        # =====================================================

        with col2:

            st.html(f"""
<div class="metric-card">

    <div class="metric-title">
        Queue Status
    </div>

    <div class="metric-value">
        {status}
    </div>

</div>
""")


        # =====================================================
        # DETECTION RESULT
        # =====================================================

        st.write("")

        st.markdown("### 🎯 AI Detection Result")


        st.image(
            detected_image,
            caption="Detected People",
            use_container_width=True
        )


# =========================================================
# FOOTER
# =========================================================

st.html("""
<div style="
    text-align:center;
    color:#6f8799;
    margin-top:45px;
    padding:20px;
    font-size:13px;
">

    QueueVision AI
    <br>
    Computer Vision Queue Monitoring System

</div>
""")