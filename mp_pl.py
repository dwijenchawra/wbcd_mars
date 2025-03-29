import argparse
import subprocess
import numpy as np
import cv2
import time

import json
import sys
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

def main(port):
    model_path = "/home/dwijen/Documents/CODE/IsaacLab/wbcd/hand_landmarker.task"  # Update with your actual model path
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        base_options=base_options,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    detector = vision.HandLandmarker.create_from_options(options)

    command = [
        'ffmpeg',
        '-listen', '1',
        '-probesize', '10000',
        '-fflags', 'nobuffer',
        '-flags', 'low_delay',
        '-strict', 'experimental',
        '-i', f'tcp://0.0.0.0:{port}',
        '-f', 'rawvideo',
        '-pix_fmt', 'uyvy422',
        'pipe:'
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    width = 640
    height = 480
    frame_size = width * height * 2
    prev_time = time.time()
    frame_rate = 30
    frame_interval = int(1000 / frame_rate)
    timestamp_ms = 0
    MARGIN = 10
    FONT_SIZE = 1
    FONT_THICKNESS = 2
    HANDEDNESS_TEXT_COLOR = (88, 205, 54)
    FPS_TEXT_COLOR = (0, 255, 255)

    def draw_landmarks_on_image(image, detection_result):
        hand_landmarks_list = detection_result.hand_landmarks
        handedness_list = detection_result.handedness
        annotated_image = np.copy(image)
        height, width, _ = image.shape

        for idx in range(len(hand_landmarks_list)):
            hand_landmarks = hand_landmarks_list[idx]
            handedness = handedness_list[idx]

            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend(
                [
                    landmark_pb2.NormalizedLandmark(
                        x=landmark.x, y=landmark.y, z=landmark.z
                    )
                    for landmark in hand_landmarks
                ]
            )

            mp.solutions.drawing_utils.draw_landmarks(
                annotated_image,
                hand_landmarks_proto,
                mp.solutions.hands.HAND_CONNECTIONS,
                mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                mp.solutions.drawing_styles.get_default_hand_connections_style()
            )

            x_coordinates = [landmark.x for landmark in hand_landmarks]
            y_coordinates = [landmark.y for landmark in hand_landmarks]
            text_x = int(min(x_coordinates) * width)
            text_y = int(min(y_coordinates) * height) - MARGIN

            text_x = max(text_x, MARGIN)
            text_y = max(text_y, MARGIN)

            cv2.putText(
                annotated_image,
                f"{handedness[0].category_name}",
                (text_x, text_y),
                cv2.FONT_HERSHEY_DUPLEX,
                FONT_SIZE,
                HANDEDNESS_TEXT_COLOR,
                FONT_THICKNESS,
                cv2.LINE_AA,
            )
        return annotated_image

    while True:
        raw_data = process.stdout.read(frame_size)
        if not raw_data or len(raw_data) < frame_size:
            break

        np_data = np.frombuffer(raw_data, dtype=np.uint8)
        frame = np_data.reshape((height, width, 2))
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_UYVY)
        rgba_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGBA)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGBA, data=rgba_frame)
        results = detector.detect_for_video(mp_image, timestamp_ms)
        timestamp_ms += frame_interval

        if results.hand_landmarks:
            bgr_frame = draw_landmarks_on_image(bgr_frame, results)
            
            key_points = []
            for hand_landmarks in results.hand_landmarks: # iterate over both hands
                key_points.append([(landmark.x, landmark.y, landmark.z) for landmark in hand_landmarks])
            
            # Serialize and print the key points
            print(json.dumps(key_points))
            sys.stdout.flush()  # Ensure the data is sent immediately


        current_time = time.time()
        elapsed_time = current_time - prev_time
        fps = 1 / elapsed_time if elapsed_time > 0 else 0
        prev_time = current_time

        cv2.putText(
            bgr_frame,
            f"FPS: {fps:.2f}",
            (MARGIN, MARGIN + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            FONT_SIZE,
            FPS_TEXT_COLOR,
            FONT_THICKNESS,
            cv2.LINE_AA,
        )
        
        # print("port", port, "showframe")
        cv2.imshow(f"MediaPipe Hand Landmarker - Port {port}", bgr_frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    process.stdout.close()
    process.stderr.close()
    process.terminate()
    cv2.destroyAllWindows()
    detector.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True, help="Port number for the FFmpeg stream")
    args = parser.parse_args()
    main(args.port)

