import streamlit as st
from PIL import Image
import subprocess, sys
import tempfile

st.set_page_config(page_title="FaceQTH", layout="centered")

# Khởi tạo session_state
if "mode" not in st.session_state:
    st.session_state["mode"] = "None"  # None, Live, Video

# Import module face_rec_cam_web
import face_rec_cam_web

# Logo
logo = Image.open("logo.png")
st.image(logo, width=400)

st.title("FaceQTH")
st.header("Face Recognition Project")

# Chọn chế độ
mode = st.radio("Chọn chế độ:", ("None", "Live Mode", "Video Mode"))
st.session_state["mode"] = mode

if st.session_state["mode"] == "Live Mode":
    face_rec_cam_web.run_face_recognition()

elif st.session_state["mode"] == "Video Mode":
    st.subheader("Video Mode")
    video_option = st.radio("Chọn nguồn video:", ("Video mặc định", "Tải video từ máy"))

    if video_option == "Video mặc định":
        if st.button("Chạy video mặc định"):
            st.info("Đang xử lý video... \nBấm q để thoát chương trình")
            subprocess.Popen([sys.executable, "src/face_rec.py", "--path", "video/video_testing.mp4"])

    elif video_option == "Tải video từ máy":
        uploaded_file = st.file_uploader("Chọn video từ máy của bạn:", type=["mp4", "avi", "mov"])
        if uploaded_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_file.read())
            video_path = tfile.name
            st.info(f"Video đã tải lên: {video_path}")
            if st.button("Chạy video đã tải lên"):
                st.info("Đang xử lý video... \nBấm q để thoát chương trình")
                subprocess.Popen([sys.executable, "src/face_rec.py", "--path", video_path])
