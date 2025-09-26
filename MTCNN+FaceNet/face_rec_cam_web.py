import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import cv2, pickle, os, av, numpy as np, time, subprocess, sys
import tensorflow.compat.v1 as tf
sys.path.insert(0, os.path.abspath("src"))
import facenet, align.detect_face
import imutils

tf.disable_v2_behavior()

# Config
CLASSIFIER_PATH = 'Models/facemodel.pkl'
FACENET_MODEL_PATH = 'Models/20180402-114759.pb'
DATASET_RAW = "Dataset/FaceData/raw"
DATASET_PROC = "Dataset/FaceData/processed"
INPUT_IMAGE_SIZE = 160
MINSIZE = 20
THRESHOLD = [0.6, 0.7, 0.7]
FACTOR = 0.709
PROB_THRESHOLD = 0.8
FACE_MIN_HEIGHT_RATIO = 0.25  # giống file CLI: mặt phải chiếm >=25% chiều cao frame

os.makedirs(DATASET_RAW, exist_ok=True)
os.makedirs(DATASET_PROC, exist_ok=True)

# Load Facenet + MTCNN 1 lần (để tránh bị delay quá 10s)
graph = tf.Graph()
with graph.as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    with sess.as_default():
        facenet.load_model(FACENET_MODEL_PATH)

    images_placeholder = graph.get_tensor_by_name("input:0")
    embeddings = graph.get_tensor_by_name("embeddings:0")
    phase_train_placeholder = graph.get_tensor_by_name("phase_train:0")
    pnet, rnet, onet = align.detect_face.create_mtcnn(sess, "src/align")

# Load classifier 1 lần
with open(CLASSIFIER_PATH, 'rb') as file:
    model, class_names = pickle.load(file)

if "capture_count" not in st.session_state:
    st.session_state["capture_count"] = 0

# Video Processor
class FaceRecognition(VideoProcessorBase):
    def __init__(self):
        self.last_frame = None
        self.last_detected_name = "Unknown"
        self.fps = 0.0
        self.frame_count = 0
        self.fps_update_time = time.time()

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = imutils.resize(img, width=600)
        img = cv2.flip(img, 1)
        self.last_frame = img.copy()

        #Tính FPS
        self.frame_count += 1
        now = time.time()
        elapsed = now - self.fps_update_time
        if elapsed >= 1.0:
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.fps_update_time = now

        current_name = "Unknown"

        # Detect trên RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        bounding_boxes, _ = align.detect_face.detect_face(
            img_rgb, MINSIZE, pnet, rnet, onet, THRESHOLD, FACTOR
        )

        faces_found = 0 if bounding_boxes is None else bounding_boxes.shape[0]

        if faces_found > 0:
            det = bounding_boxes[:, 0:4].astype(int)
            H = img.shape[0]

            for i in range(faces_found):
                x1, y1, x2, y2 = det[i]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(img.shape[1] - 1, x2), min(img.shape[0] - 1, y2)

                if (y2 - y1) / float(H) < FACE_MIN_HEIGHT_RATIO:
                    continue

                cropped = img[y1:y2, x1:x2, :]
                if cropped.size == 0:
                    continue

                # Embedding
                scaled = cv2.resize(cropped, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)
                scaled = facenet.prewhiten(scaled)
                scaled_reshape = scaled.reshape(-1, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 3)
                feed_dict = {images_placeholder: scaled_reshape, phase_train_placeholder: False}
                emb_array = sess.run(embeddings, feed_dict=feed_dict)

                # Predict
                preds = model.predict_proba(emb_array)
                best_class_indices = np.argmax(preds, axis=1)
                best_class_probabilities = preds[np.arange(len(best_class_indices)), best_class_indices]

                if best_class_probabilities[0] > PROB_THRESHOLD:
                    current_name = class_names[best_class_indices[0]]
                    color = (0, 255, 0)
                    label = f"{current_name} ({best_class_probabilities[0]:.2f})"
                else:
                    current_name = "Unknown"
                    color = (0, 0, 255)
                    label = "Unknown"

                # Vẽ bbox + tên
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, label, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Vẽ FPS
        cv2.putText(img, f"FPS: {self.fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        self.last_detected_name = current_name
        return av.VideoFrame.from_ndarray(img, format="bgr24")

#Streamlit UI
def run_face_recognition():
    st.title("FaceQTH - Live Mode")

    ctx = webrtc_streamer(
        key="face-recognition",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration={"iceServers":[{"urls":["stun:stun.l.google.com:19302"]}]},
        async_processing=True,
        media_stream_constraints={"video": True, "audio": False},
        video_processor_factory=FaceRecognition,
    )

    if ctx.state.playing and ctx.video_processor:
        vp = ctx.video_processor

        # Thêm người mới
        new_name = st.text_input("Nếu hệ thống nhận diện bạn là Unknown, nhập tên để thêm user mới:")
        if st.button("Chụp & Lưu"):
            if new_name.strip() != "" and vp.last_frame is not None:
                save_dir = os.path.join(DATASET_RAW, new_name)
                os.makedirs(save_dir, exist_ok=True)
                existing_files = os.listdir(save_dir)
                existing_numbers = [
                    int(f.split("_")[-1].split(".")[0])
                    for f in existing_files if f.startswith(new_name + "_") and f.endswith(".jpg")
                ]
                next_index = max(existing_numbers, default=0) + 1
                filename = os.path.join(save_dir, f"{new_name}_{next_index}.jpg")
                cv2.imwrite(filename, vp.last_frame)

                st.success(f"Đã lưu ảnh cho {new_name}: {filename}")
                st.session_state["capture_count"] += 1
                st.info(f"Đã chụp {st.session_state['capture_count']} / 20 ảnh")

        if st.button("Stop & Train"):
            if st.session_state["capture_count"] > 0:
                st.info("👉 Đang align dữ liệu...")
                subprocess.run(
                    ["python", "src/align_dataset_mtcnn.py", DATASET_RAW, DATASET_PROC],
                    check=True
                )

                st.info("Đang train lại classifier...")
                subprocess.run(
                    ["python", "src/classifier.py", "TRAIN", DATASET_PROC, FACENET_MODEL_PATH, CLASSIFIER_PATH],
                    check=True
                )

                with open(CLASSIFIER_PATH, 'rb') as file:
                    global model, class_names
                    model, class_names = pickle.load(file)

                st.success("Đã align và train xong")
                st.session_state["capture_count"] = 0

# Run App
if __name__ == "__main__":
    run_face_recognition()