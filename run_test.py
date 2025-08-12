import os

import cv2
from imutils import paths

from recognize_faces_image import recognize_image
from recognize_faces_video import recognize_video

ENCODINGS_PATH = "encodings.pickle"
TEST_IMAGES_DIR = "test_images"

imagePaths = list(paths.list_images(TEST_IMAGES_DIR))

for imagePath in imagePaths:
    print(f"[INFO] processing {imagePath}")
    try:
        image, names = recognize_image(
            ENCODINGS_PATH, imagePath, detection_method="cnn", tolerance=0.4
        )
        print(f"Detected faces: {names}")
        cv2.imshow("Result", image)
        cv2.waitKey(0)  # Nhấn phím bất kỳ để xem ảnh tiếp theo
    except Exception as e:
        print(f"[ERROR] {e}")

cv2.destroyAllWindows()
