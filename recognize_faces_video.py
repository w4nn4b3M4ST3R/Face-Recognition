def recognize_video(
    video_path,
    encodings_path,
    output_path=None,
    detection_method="cnn",
    display=1,
    tolerance=0.42,
    output_width=1280,
):
    import pickle

    import cv2
    import face_recognition
    import imutils

    # Load encodings
    print("[INFO] loading encodings...")
    with open(encodings_path, "rb") as f:
        data = pickle.load(f)

    # Open video
    print("[INFO] opening video file...")
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)  # Lấy FPS gốc
    frame_size = None
    writer = None

    while True:
        ret, frame = video.read()
        if not ret:
            break

        # Resize để giảm dung lượng (tùy chọn)
        frame = imutils.resize(frame, width=output_width)
        if frame_size is None:
            frame_size = (frame.shape[1], frame.shape[0])  # (width, height)

        # Convert BGR -> RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces & encode
        boxes = face_recognition.face_locations(rgb, model=detection_method)
        encodings = face_recognition.face_encodings(rgb, boxes)

        # Nhận diện tên
        names = []
        for encoding in encodings:
            matches = face_recognition.compare_faces(
                data["encodings"], encoding, tolerance
            )
            name = "Unknown"
            if True in matches:
                matched_idxs = [i for i, b in enumerate(matches) if b]
                counts = {}
                for i in matched_idxs:
                    matched_name = data["names"][i]
                    counts[matched_name] = counts.get(matched_name, 0) + 1
                name = max(counts, key=counts.get)
            names.append(name)

        # Vẽ khung và tên
        for (top, right, bottom, left), name in zip(boxes, names):
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            y = top - 15 if top - 15 > 15 else top + 15
            cv2.putText(
                frame,
                name,
                (left, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

        # Tạo VideoWriter với FPS gốc
        if writer is None and output_path is not None:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size, True)

        # Ghi frame vào file
        if writer is not None:
            writer.write(frame)

        # Hiển thị nếu cần
        if display > 0:
            cv2.imshow("Video", frame)
            # waitKey ở đây chỉ để xem trực tiếp, không ảnh hưởng tới tốc độ output
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    video.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
