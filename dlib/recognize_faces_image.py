import pickle

import cv2
import face_recognition


def recognize_image(
    encodings_path, image_path, detection_method="cnn", threshold=0.4
):
    """
    Nhận diện khuôn mặt trên ảnh.

    Args:
        encodings_path (str): Đường dẫn file encodings.pickle.
        image_path (str): Đường dẫn ảnh test.
        detection_method (str): "cnn" hoặc "hog" để detect face.
        tolerance (float): ngưỡng so sánh mặt (mặc định 0.4).

    Returns:
        image (numpy array): Ảnh gốc đã vẽ khung và tên.
        names (list): Danh sách tên các khuôn mặt nhận diện được trong ảnh.
    """
    # Load encodings
    data = pickle.loads(open(encodings_path, "rb").read())

    # Load và chuyển ảnh sang RGB
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Không đọc được ảnh: {image_path}")
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect face và encode
    boxes = face_recognition.face_locations(rgb, model=detection_method)
    encodings = face_recognition.face_encodings(rgb, boxes)

    names = []
    for encoding in encodings:
        matches = face_recognition.compare_faces(
            data["encodings"], encoding, threshold
        )
        name = "Unknown"
        if True in matches:
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}
            for i in matchedIdxs:
                name_i = data["names"][i]
                counts[name_i] = counts.get(name_i, 0) + 1
            name = max(counts, key=counts.get)
        names.append(name)

    # Vẽ khung và tên lên ảnh
    for (top, right, bottom, left), name in zip(boxes, names):
        cv2.rectangle(
            image, (left, top), (right, bottom), (0, 255, 0), thickness=2
        )
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(
            image,
            name,
            (left, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            thickness=2,
        )

    return image, names
