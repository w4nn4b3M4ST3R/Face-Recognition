from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from imutils.video import VideoStream


import argparse
import facenet
import imutils
import os
import sys
import math
import pickle
import align.detect_face
import numpy as np
import cv2
import collections
from sklearn.svm import SVC
import time

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

def capture_new_face(name, num_samples=20, save_dir="Dataset/FaceData/raw"):
    cap = cv2.VideoCapture(0)
    person_dir = os.path.join(save_dir, name)
    os.makedirs(person_dir, exist_ok=True)

    count = 0
    print(f"[INFO] Bắt đầu capture cho {name}, nhấn 'c' để chụp, 'q' để thoát.")
    while count < num_samples:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        cv2.imshow("Capture new face", frame)

        k = cv2.waitKey(1) & 0xFF
        if k == ord('c'):  # nhấn "c" để chụp
            filename = os.path.join(person_dir, f"{name}_{count}.jpg")
            cv2.imwrite(filename, frame)
            print(f"[INFO] Saved {filename}")
            count += 1
        elif k == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', help='Path of the video you want to test on.', default=0)
    args = parser.parse_args()

    # Tham số nhận diện
    MINSIZE = 20
    THRESHOLD = [0.6, 0.7, 0.7]
    FACTOR = 0.709
    INPUT_IMAGE_SIZE = 160
    CLASSIFIER_PATH = 'Models/facemodel.pkl'
    FACENET_MODEL_PATH = 'Models/20180402-114759.pb'

    # Load classifier
    with open(CLASSIFIER_PATH, 'rb') as file:
        model, class_names = pickle.load(file)
    print("[INFO] Custom Classifier loaded.")

    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))

        with sess.as_default():
            print('[INFO] Loading FaceNet model...')
            facenet.load_model(FACENET_MODEL_PATH)

            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, "src/align")

            cap = VideoStream(src=0).start()

            current_name = ""  # tên hiện tại
            #biến FPS
            fps = 0
            frame_count = 0
            start_time = time.time()

            while True:
                frame = cap.read()
                frame = imutils.resize(frame, width=600)
                frame = cv2.flip(frame, 1)

                #tính FPS
                frame_count += 1
                elapsed_time = time.time() - start_time
                if elapsed_time > 1:
                    fps = frame_count / elapsed_time
                    frame_count = 0
                    start_time = time.time()

                bounding_boxes, _ = align.detect_face.detect_face(
                    frame, MINSIZE, pnet, rnet, onet, THRESHOLD, FACTOR
                )
                faces_found = bounding_boxes.shape[0]

                current_name = "Unknown"  # reset mỗi vòng lặp

                if faces_found > 0:
                    det = bounding_boxes[:, 0:4]
                    bb = np.zeros((faces_found, 4), dtype=np.int32)

                    for i in range(faces_found):
                        bb[i][0], bb[i][1], bb[i][2], bb[i][3] = det[i][:4]

                        if (bb[i][3]-bb[i][1]) / frame.shape[0] > 0.25:
                            cropped = frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :]
                            scaled = cv2.resize(cropped, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)
                            scaled = facenet.prewhiten(scaled)
                            scaled_reshape = scaled.reshape(-1, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 3)

                            feed_dict = {images_placeholder: scaled_reshape, phase_train_placeholder: False}
                            emb_array = sess.run(embeddings, feed_dict=feed_dict)

                            predictions = model.predict_proba(emb_array)
                            best_class_indices = np.argmax(predictions, axis=1)
                            best_class_probabilities = predictions[
                                np.arange(len(best_class_indices)), best_class_indices
                            ]

                            if best_class_probabilities > 0.8:
                                current_name = class_names[best_class_indices[0]]
                                cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)
                                cv2.putText(frame, f"{current_name} ({best_class_probabilities[0]:.2f})",
                                            (bb[i][0], bb[i][3] + 20),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
                            else:
                                current_name = "Unknown"
                                cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 0, 255), 2)
                                cv2.putText(frame, current_name, (bb[i][0], bb[i][3] + 20),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

                cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                cv2.imshow("Face Recognition", frame)

                #xử lý phím bấm
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('n') and current_name == "Unknown":
                    new_name = input("[INPUT] New user - type your name: ")

                    #tắt camera trước khi align/train
                    cap.stop()
                    cv2.destroyAllWindows()
                    try:
                        cap.stream.release()
                    except:
                        pass
                    time.sleep(1)

                    capture_new_face(new_name)

                    # align dataset
                    print("[INFO] Aligning dataset...")
                    os.system(f"python src/align_dataset_mtcnn.py Dataset/FaceData/raw Dataset/FaceData/processed")

                    # retrain classifier
                    print("[INFO] Retraining classifier...")
                    os.system(f"python src/classifier.py TRAIN Dataset/FaceData/processed Models/20180402-114759.pb Models/facemodel.pkl")

                    print("[INFO] New person added. Restart the script to load updated model.")
                    break   # thoát để load lại model mới

            cap.stop()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()