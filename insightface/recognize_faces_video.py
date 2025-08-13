import time

import cv2
from recognize_faces_image import recognize_image


def recognize_video(
    video,
    embeddings,
    labels,
    threshold=0.4,
):
    while True:
        start = time.time()
        ret, frame = video.read()
        frame = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_AREA)
        if not ret:
            print("---END OF VIDEO---")
            break

        img = recognize_image(frame, embeddings, labels, threshold)

        end = time.time()
        fps = 1 / (end - start)
        cv2.putText(
            img,
            f"FPS: {fps:.2f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        cv2.imshow("Video", img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video.release()
