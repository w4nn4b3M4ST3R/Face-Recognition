import cv2
import numpy as np
from scipy.spatial.distance import cosine

from insightface.app import FaceAnalysis

app = FaceAnalysis(
    name="buffalo_l",
    allowed_modules=["detection", "recognition"],
    providers=["CUDAExecutionProvider"],
)
app.prepare(ctx_id=0, det_size=(640, 640))


def recognize_image(image, embeddings, labels, threshold=0.4):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    faces = app.get(image)
    names = []
    for face in faces:
        d = [cosine(face.embedding, e) for e in embeddings]
        min_idx = np.argmin(d)
        name = "Unknown" if d[min_idx] > threshold else labels[min_idx]
        names.append(name)
        x1, y1, x2, y2 = [int(i) for i in face.bbox]
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cv2.putText(
            image,
            f"{name}:{d[min_idx]:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            1,
        )
        if face.landmark is not None:
            for x, y in face.landmark.astype(int):
                cv2.circle(image, (x, y), 2, (0, 0, 255), -1)
    print(f"Recognized: {names} ")
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
