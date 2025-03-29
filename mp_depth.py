# import argparse
# import subprocess
# import numpy as np
# import cv2
# import time
# import json
# import sys
# import mediapipe as mp
# from mediapipe.tasks import python
# from mediapipe.tasks.python import vision
# from mediapipe.framework.formats import landmark_pb2

# def read_frame(process, width, height):
#     """
#     Read a frame from the given subprocess stream
    
#     Args:
#         process (subprocess.Popen): The subprocess reading the stream
#         width (int): Frame width
#         height (int): Frame height
#         frame_format (str): Format of the incoming video stream
    
#     Returns:
#         numpy.ndarray: Decoded frame
#     """
#     frame_size = width * height * 2
#     raw_data = process.stdout.read(frame_size)
#     if not raw_data or len(raw_data) < frame_size:
#         return None
#     np_data = np.frombuffer(raw_data, dtype=np.uint8)
#     frame = np_data.reshape((height, width, 2))
#     bgr_frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_UYVY)
#     rgba_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGBA)
#     return bgr_frame, rgba_frame
    
# def main(rgb_port, depth_port):
#     # MediaPipe Hand Landmarker Setup
#     model_path = "/home/dwijen/Documents/CODE/IsaacLab/wbcd/hand_landmarker.task"  # Update with your actual model path
#     base_options = python.BaseOptions(model_asset_path=model_path)
#     options = vision.HandLandmarkerOptions(
#         running_mode=mp.tasks.vision.RunningMode.VIDEO,
#         base_options=base_options,
#         num_hands=2,
#         min_hand_detection_confidence=0.5,
#         min_hand_presence_confidence=0.5,
#         min_tracking_confidence=0.5,
#     )
#     detector = vision.HandLandmarker.create_from_options(options)

#     # RGB Camera Stream Setup
#     rgb_command = [
#         'ffmpeg',
#         '-listen', '1',
#         '-probesize', '10000',
#         '-fflags', 'nobuffer',
#         '-flags', 'low_delay',
#         '-strict', 'experimental',
#         '-i', f'tcp://0.0.0.0:{rgb_port}',
#         '-f', 'rawvideo',
#         '-pix_fmt', 'uyvy422',
#         'pipe:'
#     ]
#     rgb_process = subprocess.Popen(rgb_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

#     # Depth Camera Stream Setup
#     depth_command = [
#         'ffmpeg',
#         '-listen', '1',
#         '-probesize', '10000',
#         '-fflags', 'nobuffer',
#         '-flags', 'low_delay',
#         '-strict', 'experimental',
#         '-i', f'tcp://0.0.0.0:{depth_port}',
#         '-f', 'rawvideo',
#         '-pix_fmt', 'uyvy422',
#         'pipe:'
#     ]
#     depth_process = subprocess.Popen(depth_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

#     width = 640
#     height = 480
#     prev_time = time.time()
#     frame_rate = 30
#     frame_interval = int(1000 / frame_rate)
#     timestamp_ms = 0

#     # Constants for visualization
#     MARGIN = 10
#     FONT_SIZE = 1
#     FONT_THICKNESS = 2
#     HANDEDNESS_TEXT_COLOR = (88, 205, 54)
#     FPS_TEXT_COLOR = (0, 255, 255)

#     def draw_landmarks_on_image(image, detection_result):
#         """Draw hand landmarks on the image"""
#         hand_landmarks_list = detection_result.hand_landmarks
#         handedness_list = detection_result.handedness
#         annotated_image = np.copy(image)
#         height, width, _ = image.shape

#         for idx in range(len(hand_landmarks_list)):
#             hand_landmarks = hand_landmarks_list[idx]
#             handedness = handedness_list[idx]

#             hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
#             hand_landmarks_proto.landmark.extend(
#                 [
#                     landmark_pb2.NormalizedLandmark(
#                         x=landmark.x, y=landmark.y, z=landmark.z
#                     )
#                     for landmark in hand_landmarks
#                 ]
#             )

#             mp.solutions.drawing_utils.draw_landmarks(
#                 annotated_image,
#                 hand_landmarks_proto,
#                 mp.solutions.hands.HAND_CONNECTIONS,
#                 mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
#                 mp.solutions.drawing_styles.get_default_hand_connections_style()
#             )

#             x_coordinates = [landmark.x for landmark in hand_landmarks]
#             y_coordinates = [landmark.y for landmark in hand_landmarks]
#             text_x = int(min(x_coordinates) * width)
#             text_y = int(min(y_coordinates) * height) - MARGIN

#             text_x = max(text_x, MARGIN)
#             text_y = max(text_y, MARGIN)

#             cv2.putText(
#                 annotated_image,
#                 f"{handedness[0].category_name}",
#                 (text_x, text_y),
#                 cv2.FONT_HERSHEY_DUPLEX,
#                 FONT_SIZE,
#                 HANDEDNESS_TEXT_COLOR,
#                 FONT_THICKNESS,
#                 cv2.LINE_AA,
#             )
#         return annotated_image

#     def replace_z_coordinates(hand_landmarks, depth_frame):
#         """
#         Replace z-coordinates with depth information
        
#         Args:
#             hand_landmarks (list): MediaPipe hand landmarks
#             depth_frame (numpy.ndarray): Depth camera frame
        
#         Returns:
#             list: Updated hand landmarks with depth-based z-coordinates
#         """
#         updated_landmarks = []
#         for landmark in hand_landmarks:
#             # Convert normalized coordinates to pixel coordinates
#             x_pixel = int(landmark.x * width)
#             y_pixel = int(landmark.y * height)
            
#             # Ensure pixel coordinates are within frame bounds
#             x_pixel = max(0, min(x_pixel, width - 1))
#             y_pixel = max(0, min(y_pixel, height - 1))
            
#             # Get depth value (assuming depth frame is grayscale or single channel)
#             # Normalize depth to a similar range as MediaPipe's original z-coordinate
#             if len(depth_frame.shape) > 2:
#                 depth_value = depth_frame[y_pixel, x_pixel, 0]  # First channel for depth
#             else:
#                 depth_value = depth_frame[y_pixel, x_pixel]
            
#             # Normalize depth value (you may need to adjust this based on your depth camera)
#             normalized_depth = depth_value / 255.0  # Assuming 8-bit depth image
            
#             # make new tuple
#             new_landmark = (landmark.x, landmark.y, normalized_depth)
            
#             updated_landmarks.append(new_landmark)
        
#         return updated_landmarks

#     while True:
#         # Read RGB and Depth frames
#         bgr_frame, rgba_frame = read_frame(rgb_process, width, height)
#         bgr_depth_frame, rgba_depth_frame = read_frame(depth_process, width, height)
        
#         if bgr_frame is None or bgr_depth_frame is None:
#             # output random landmarks
#             landmarks = np.random.rand(21, 3)
#             print(json.dumps(landmarks.tolist()))
#             sys.stdout.flush()
#             continue

#         # Convert frames for MediaPipe
#         mp_image = mp.Image(image_format=mp.ImageFormat.SRGBA, data=rgba_frame)
        
#         # Perform hand detection
#         results = detector.detect_for_video(mp_image, timestamp_ms)
#         timestamp_ms += frame_interval

#         if results.hand_landmarks:
#             # Replace z-coordinates with depth information
#             updated_hand_landmarks = []
#             key_points = []
#             for hand_landmarks in results.hand_landmarks:
#                 # Replace z-coordinates
#                 depth_adjusted_landmarks = replace_z_coordinates(hand_landmarks, bgr_depth_frame)
#                 updated_hand_landmarks.append(depth_adjusted_landmarks)
                
#                 # Prepare key points for output
#                 key_points.append(depth_adjusted_landmarks)
            
#             # # Update results with depth-adjusted landmarks
#             # results.hand_landmarks = updated_hand_landmarks
#             ## invalid now since they are tuples
            
#             # Draw OG landmarks on frame
#             # bgr_frame = draw_landmarks_on_image(bgr_frame, results)
#             # draw on depth frame
#             bgr_frame = draw_landmarks_on_image(bgr_depth_frame, results)
            
#             # Serialize and print the key points
#             print(json.dumps(key_points))
#             sys.stdout.flush()  # Ensure the data is sent immediately

#         # Calculate and display FPS
#         current_time = time.time()
#         elapsed_time = current_time - prev_time
#         fps = 1 / elapsed_time if elapsed_time > 0 else 0
#         prev_time = current_time

#         cv2.putText(
#             bgr_frame,
#             f"FPS: {fps:.2f}",
#             (MARGIN, MARGIN + 20),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             FONT_SIZE,
#             FPS_TEXT_COLOR,
#             FONT_THICKNESS,
#             cv2.LINE_AA,
#         )
        
#         # Display the frame
#         cv2.imshow(f"MediaPipe Hand Landmarker - RGB Port {rgb_port}, Depth Port {depth_port}", bgr_frame)
#         if cv2.waitKey(1) & 0xFF == 27:  # ESC key
#             break

#     # Cleanup
#     rgb_process.stdout.close()
#     rgb_process.stderr.close()
#     rgb_process.terminate()
    
#     depth_process.stdout.close()
#     depth_process.stderr.close()
#     depth_process.terminate()
    
#     cv2.destroyAllWindows()
#     detector.close()

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--rgb_port", type=int, required=True, help="Port number for the RGB camera stream", default=1234)
#     parser.add_argument("--depth_port", type=int, required=True, help="Port number for the depth camera stream", default=1235)
#     args = parser.parse_args()
#     main(args.rgb_port, args.depth_port)

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

def read_frame(process, width, height):
    """
    Read a frame from the given subprocess stream
    
    Args:
        process (subprocess.Popen): The subprocess reading the stream
        width (int): Frame width
        height (int): Frame height
        frame_format (str): Format of the incoming video stream
    
    Returns:
        numpy.ndarray: Decoded frame
    """
    frame_size = width * height * 2
    raw_data = process.stdout.read(frame_size)
    if not raw_data or len(raw_data) < frame_size:
        return None, None
    np_data = np.frombuffer(raw_data, dtype=np.uint8)
    frame = np_data.reshape((height, width, 2))
    bgr_frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_UYVY)
    rgba_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGBA)
    return bgr_frame, rgba_frame

# Global variables to track transformations
zoom_scale = 1.0
pan_x, pan_y = 0, 0
rotation_angle = 0
is_dragging = False
prev_mouse_x, prev_mouse_y = 0, 0

def plot_3d_landmarks(landmarks_list):
    """
    Create an interactive 3D plot of hand landmarks using OpenCV with Y-axis rotation
    
    Args:
        landmarks_list (list): List of hand landmarks with x, y, z coordinates.
            Can be either a single hand or a list of two hands.
    
    Returns:
        numpy.ndarray: 3D visualization of hand landmarks with interactive controls
    """
    global zoom_scale, pan_x, pan_y, rotation_angle
    
    # Create a blank white canvas
    canvas = np.ones((480, 640, 3), dtype=np.uint8) * 255
    
    print(landmarks_list)
    
    if len(landmarks_list) == 1:
        landmarks = np.array(landmarks_list[0])
    else:
        landmarks = np.concatenate([np.array(landmarks_list[0]), np.array(landmarks_list[1])])
    
    x_min, y_min, z_min = 0, 0, 0
    x_max, y_max, z_max = 1, 1, 1
    
    # Scale coordinates to canvas size
    def normalize_coord(coord, min_val, max_val, canvas_size):
        return ((coord - min_val) / (max_val - min_val)) * (canvas_size - 100) + 50
    
    # Create a list of transformed points
    transformed_points = []
    for i in range(len(landmarks)):
        # Get normalized coordinates (0-1 range)
        
        x_norm = normalize_coord(landmarks[i][0], x_min, x_max, 640)
        y_norm = normalize_coord(landmarks[i][1], y_min, y_max, 480)
        
        # force artificial z min
        z_norm = normalize_coord(landmarks[i][2], z_min, z_max, 255)
        
        # Convert to 3D space coordinates (centered)
        x_3d = (x_norm - 320) / 100  # Scale down for better rotation
        y_3d = (y_norm - 240) / 100
        z_3d = (z_norm - 127.5) / 100
        
        # Apply Y-axis rotation (around vertical axis)
        rad = np.radians(rotation_angle)
        x_rot = x_3d * np.cos(rad) + z_3d * np.sin(rad)
        z_rot = -x_3d * np.sin(rad) + z_3d * np.cos(rad)
        y_rot = y_3d  # Y coordinate remains unchanged in Y-axis rotation
        
        # Apply zoom
        x_rot *= zoom_scale
        y_rot *= zoom_scale
        z_rot *= zoom_scale
        
        # Project back to 2D (orthographic projection - just drop Z coordinate)
        x_proj = x_rot * 100 + 320 + pan_x
        y_proj = y_rot * 100 + 240 + pan_y
        
        # Store the projected point and original z for coloring
        transformed_points.append((int(x_proj), int(y_proj), z_norm))
    
    # Plot landmarks
    for i, (x, y, z) in enumerate(transformed_points):
        # Color intensity based on z-coordinate
        color = (0, 0, int(z))
        
        # Draw point
        cv2.circle(canvas, (x, y), int(5 * zoom_scale), color, -1)
        
        # Draw text with landmark index
        cv2.putText(canvas, str(i), (x, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3 * zoom_scale, 
                   (0, 0, 0), 1, cv2.LINE_AA)
    
    # Connect landmarks
    hand_connections = [
        (0,1), (1,2), (2,3), (3,4),  # Thumb
        (0,5), (5,6), (6,7), (7,8),  # Index
        (0,9), (9,10), (10,11), (11,12),  # Middle
        (0,13), (13,14), (14,15), (15,16),  # Ring
        (0,17), (17,18), (18,19), (19,20)   # Pinky
    ]
    
    for connection in hand_connections:
        start_x, start_y, _ = transformed_points[connection[0]]
        end_x, end_y, _ = transformed_points[connection[1]]
        
        cv2.line(canvas, (start_x, start_y), (end_x, end_y), (100, 100, 100), 1)
    
    # Add instructions
    cv2.putText(canvas, "Mouse Wheel: Zoom", (10, 20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(canvas, "Right Drag: Pan", (10, 40), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(canvas, "Left Drag: Rotate (Y-axis)", (10, 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(canvas, "R: Reset View", (10, 80), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    return canvas

# Mouse callback functions
def mouse_callback(event, x, y, flags, param):
    global zoom_scale, pan_x, pan_y, rotation_angle, is_dragging, prev_mouse_x, prev_mouse_y
    
    if event == cv2.EVENT_MOUSEWHEEL:
        # Zoom with mouse wheel
        if flags > 0:  # Scroll up
            zoom_scale *= 1.1
        else:  # Scroll down
            zoom_scale /= 1.1
        zoom_scale = np.clip(zoom_scale, 0.1, 5.0)
    
    elif event == cv2.EVENT_RBUTTONDOWN:
        # Start right-drag pan
        is_dragging = True
        prev_mouse_x, prev_mouse_y = x, y
    
    elif event == cv2.EVENT_RBUTTONUP:
        # End right-drag pan
        is_dragging = False
    
    elif event == cv2.EVENT_MOUSEMOVE and is_dragging and (flags & cv2.EVENT_FLAG_RBUTTON):
        # Pan with right-drag
        pan_x += x - prev_mouse_x
        pan_y += y - prev_mouse_y
        prev_mouse_x, prev_mouse_y = x, y
    
    elif event == cv2.EVENT_LBUTTONDOWN:
        # Start left-drag rotate
        is_dragging = True
        prev_mouse_x, prev_mouse_y = x, y
    
    elif event == cv2.EVENT_LBUTTONUP:
        # End left-drag rotate
        is_dragging = False
    
    elif event == cv2.EVENT_MOUSEMOVE and is_dragging and (flags & cv2.EVENT_FLAG_LBUTTON):
        # Rotate with left-drag
        rotation_angle += (x - prev_mouse_x) * 0.5
        prev_mouse_x, prev_mouse_y = x, y

def main(rgb_port, depth_port):
    # MediaPipe Hand Landmarker Setup
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

    # RGB Camera Stream Setup
    rgb_command = [
        'ffmpeg',
        '-listen', '1',
        '-probesize', '10000',
        '-fflags', 'nobuffer',
        '-flags', 'low_delay',
        '-strict', 'experimental',
        '-i', f'tcp://0.0.0.0:{rgb_port}',
        '-f', 'rawvideo',
        '-pix_fmt', 'uyvy422',
        'pipe:'
    ]
    rgb_process = subprocess.Popen(rgb_command, stdout=subprocess.PIPE, stderr=sys.stderr)

    # Depth Camera Stream Setup
    depth_command = [
        'ffmpeg',
        '-listen', '1',
        '-probesize', '10000',
        '-fflags', 'nobuffer',
        '-flags', 'low_delay',
        '-strict', 'experimental',
        '-i', f'tcp://0.0.0.0:{depth_port}',
        '-f', 'rawvideo',
        '-pix_fmt', 'gray',
        'pipe:'
    ]
    depth_process = subprocess.Popen(depth_command, stdout=subprocess.PIPE, stderr=sys.stderr)

    width = 640
    height = 480
    prev_time = time.time()
    frame_rate = 30
    frame_interval = int(1000 / frame_rate)
    timestamp_ms = 0

    # Constants for visualization
    MARGIN = 10
    FONT_SIZE = 1
    FONT_THICKNESS = 2
    HANDEDNESS_TEXT_COLOR = (88, 205, 54)
    FPS_TEXT_COLOR = (0, 255, 255)

    def draw_landmarks_on_image(image, detection_result):
        """Draw hand landmarks on the image"""
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

    def replace_z_coordinates(hand_landmarks, depth_frame):
        """
        Replace z-coordinates with depth information
        
        Args:
            hand_landmarks (list): MediaPipe hand landmarks
            depth_frame (numpy.ndarray): Depth camera frame
        
        Returns:
            list: Updated hand landmarks with depth-based z-coordinates
        """
        updated_landmarks = []
        for landmark in hand_landmarks:
            # Convert normalized coordinates to pixel coordinates
            x_pixel = int(landmark.x * width)
            y_pixel = int(landmark.y * height)
            
            # Ensure pixel coordinates are within frame bounds
            x_pixel = max(0, min(x_pixel, width - 1))
            y_pixel = max(0, min(y_pixel, height - 1))
            
            # Get depth value (assuming depth frame is grayscale or single channel)
            # Normalize depth to a similar range as MediaPipe's original z-coordinate
            if len(depth_frame.shape) > 2:
                depth_value = depth_frame[y_pixel, x_pixel, 0]  # First channel for depth
            else:
                depth_value = depth_frame[y_pixel, x_pixel]
            
            # Normalize depth value (you may need to adjust this based on your depth camera)
            normalized_depth = depth_value / 255.0  # Assuming 8-bit depth image
            
            # make new tuple
            new_landmark = (landmark.x, landmark.y, normalized_depth)
            
            updated_landmarks.append(new_landmark)
        
        return updated_landmarks


    cv2.namedWindow("3D Hand Landmarks")
    cv2.setMouseCallback("3D Hand Landmarks", mouse_callback)


    while True:
        # Read RGB and Depth frames
        bgr_frame, rgba_frame = read_frame(rgb_process, width, height)
        bgr_depth_frame, rgba_depth_frame = read_frame(depth_process, width, height)
        
        if bgr_frame is None or bgr_depth_frame is None:
            # output random landmarks, 2x21x3
            landmarks = [np.random.rand(21, 3).tolist(), np.random.rand(21, 3).tolist()]
            print(json.dumps(landmarks))
            sys.stdout.flush()
            
            landmarks_3d = plot_3d_landmarks(landmarks)
            cv2.imshow("3D Hand Landmarks", landmarks_3d)
            continue

        # Convert frames for MediaPipe
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGBA, data=rgba_frame)
        
        # Perform hand detection
        results = detector.detect_for_video(mp_image, timestamp_ms)
        timestamp_ms += frame_interval

        if results.hand_landmarks:
            # Replace z-coordinates with depth information
            updated_hand_landmarks = []
            key_points = []
            for hand_landmarks in results.hand_landmarks:
                # Replace z-coordinates
                depth_adjusted_landmarks = replace_z_coordinates(hand_landmarks, bgr_depth_frame)
                updated_hand_landmarks.append(depth_adjusted_landmarks)
                
                # Prepare key points for output
                key_points.append(depth_adjusted_landmarks)
            
            # Draw on depth frame
            bgr_frame = draw_landmarks_on_image(bgr_depth_frame, results)
            
            # Create 3D plot of landmarks
            if key_points:
                landmarks_3d = plot_3d_landmarks(key_points)
                cv2.imshow("3D Hand Landmarks", landmarks_3d)
            
            # Serialize and print the key points
            print(json.dumps(key_points))
            sys.stdout.flush()  # Ensure the data is sent immediately

        # Calculate and display FPS
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
        
        # Display the frame
        cv2.imshow(f"MediaPipe Hand Landmarker - RGB Port {rgb_port}, Depth Port {depth_port}", bgr_frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC key
            break

    # Cleanup
    rgb_process.stdout.close()
    rgb_process.stderr.close()
    rgb_process.terminate()
    
    depth_process.stdout.close()
    depth_process.stderr.close()
    depth_process.terminate()
    
    cv2.destroyAllWindows()
    detector.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb_port", type=int, required=True, help="Port number for the RGB camera stream", default=1234)
    parser.add_argument("--depth_port", type=int, required=True, help="Port number for the depth camera stream", default=1235)
    args = parser.parse_args()
    main(args.rgb_port, args.depth_port)