import os

import cv2
import numpy as np

from insightface.app import FaceAnalysis

app = FaceAnalysis(
    name="buffalo_l",
    allowed_modules=["detection", "recognition"],
    providers=["CUDAExecutionProvider"],
)
app.prepare(ctx_id=0, det_size=(640, 640))

dataset_path = "../dataset"
encodings = []
labels = []

for person_name in os.listdir(dataset_path):
    folder = os.path.join(dataset_path, person_name)

    for img_name in os.listdir(folder):
        path = os.path.join(folder, img_name)
        img = cv2.imread(path)

        face = app.get(img)
        encodings.append(face[0].embedding)
        labels.append(person_name)

np.savez("encodings.npz", encodings=encodings, labels=labels)
