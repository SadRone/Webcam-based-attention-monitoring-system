import cv2
import time
import mediapipe as mp

# Eye landmark indices from MediaPipe FaceMesh
LEFT_EYE_IDX = [33, 133, 160, 159, 158, 157, 173, 153, 154, 155, 144, 145]
RIGHT_EYE_IDX = [362, 263, 387, 386, 385, 384, 398, 373, 374, 380, 381, 382]


def normalized_to_pixel(landmark, frame_width, frame_height):
    """Convert normalized landmark coordinates to pixel coordinates."""
    x = int(landmark.x * frame_width)
    y = int(landmark.y * frame_height)
    return (x, y)


def draw_eye_points(frame, face_landmarks, eye_indices, color):
    """Draw eye landmarks and return their pixel coordinates."""
    h, w, _ = frame.shape
    points = []

    for idx in eye_indices:
        landmark = face_landmarks.landmark[idx]
        point = normalized_to_pixel(landmark, w, h)
        points.append(point)
        cv2.circle(frame, point, 2, color, -1)

    return points


def draw_eye_outline(frame, points, color):
    """Draw outline around the eye landmarks."""
    if len(points) > 1:
        for i in range(len(points)):
            pt1 = points[i]
            pt2 = points[(i + 1) % len(points)]
            cv2.line(frame, pt1, pt2, color, 1)


def get_eye_center(points):
    """Compute the center of the eye landmarks."""
    if not points:
        return None
    x = int(sum(p[0] for p in points) / len(points))
    y = int(sum(p[1] for p in points) / len(points))
    return (x, y)


def main():
    print("RUNNING NEW camera.py")
    cap = cv2.VideoCapture(0)

    # Use the correct import style
    from mediapipe import face_mesh

    try:
        if not cap.isOpened():
            raise RuntimeError("Error: Could not open webcam.")

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        prev_time = time.time()
        window_name = "Eye Landmark Extraction + Eye Center"

        with face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as face_mesh_model:

            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Could not read frame.")
                    break

                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                results = face_mesh_model.process(rgb_frame)

                if results.multi_face_landmarks:
                    face_landmarks = results.multi_face_landmarks[0]

                    left_eye_points = draw_eye_points(
                        frame, face_landmarks, LEFT_EYE_IDX, (0, 255, 0)
                    )
                    right_eye_points = draw_eye_points(
                        frame, face_landmarks, RIGHT_EYE_IDX, (0, 255, 255)
                    )

                    draw_eye_outline(frame, left_eye_points, (0, 255, 0))
                    draw_eye_outline(frame, right_eye_points, (0, 255, 255))

                    left_center = get_eye_center(left_eye_points)
                    right_center = get_eye_center(right_eye_points)

                    if left_center:
                        cv2.circle(frame, left_center, 4, (255, 0, 0), -1)
                        cv2.putText(
                            frame,
                            f"L-eye center: {left_center}",
                            (20, 150),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (255, 0, 0),
                            2,
                        )

                    if right_center:
                        cv2.circle(frame, right_center, 4, (255, 0, 255), -1)
                        cv2.putText(
                            frame,
                            f"R-eye center: {right_center}",
                            (20, 180),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (255, 0, 255),
                            2,
                        )

                current_time = time.time()
                fps = 1 / (current_time - prev_time) if current_time != prev_time else 0
                prev_time = current_time

                cv2.putText(
                    frame,
                    "Step 5+: Eye Landmark Extraction + Eye Center",
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    frame,
                    "Press 'Q' to quit",
                    (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )
                cv2.putText(
                    frame,
                    f"FPS: {fps:.2f}",
                    (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2,
                )

                if results.multi_face_landmarks:
                    cv2.putText(
                        frame,
                        "Eyes tracked",
                        (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )
                else:
                    cv2.putText(
                        frame,
                        "No face detected",
                        (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                    )

                cv2.imshow(window_name, frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        for _ in range(5):
            cv2.waitKey(1)


if __name__ == "__main__":
    main()
