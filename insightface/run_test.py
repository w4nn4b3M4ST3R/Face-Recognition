import os

import cv2
import numpy as np
from recognize_faces_image import recognize_image
from recognize_faces_video import recognize_video

encodings = np.load("encodings.npz", allow_pickle=True)
embeddings = encodings["encodings"]
labels = encodings["labels"]
test_path = "../test_images"
video = cv2.VideoCapture("../test_video.mp4")
threshold = 0.4

for img_name in os.listdir(test_path):
    img_path = os.path.join(test_path, img_name)
    img = recognize_image(
        cv2.imread(img_path),
        embeddings,
        labels,
        threshold,
    )
    cv2.imshow(img_name, img)
    cv2.waitKey(0)

recognize_video(video, embeddings, labels, threshold)
cv2.destroyAllWindows()
