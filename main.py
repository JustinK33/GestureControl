import cv2
import mediapipe as mp
import pyautogui
import time
import math
from collections import deque

cap = cv2.VideoCapture(0) # 0 means default webcam
success, frame = cap.read() # reads one frame
last_boxes = []
latest_hand_result = None

# Cursor smoothing
CURSOR_SMOOTHING_ALPHA = 0.3  # Lower = more smoothing (0.1-0.5 range recommended)
last_cursor_pos = None

# Click stabilization
PINCH_HYSTERESIS_RATIO = 1.3  # Upper threshold = threshold * ratio
was_pinched = False

if not success:
    print("could not read from camera")
    cap.release()
    exit()

h, w = frame.shape[:2] # image height/width
# frame is a numpy array and opencv usually stores that as (height, width, channels)
# thats why we use .shape[:2] we only want the first 2 values of the frame

screen_w, screen_h = pyautogui.size() # get screen resolution

detector = cv2.FaceDetectorYN.create("face_detection_yunet_2023mar.onnx", "", (w, h))

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode


def print_result(result, output_image: mp.Image, timestamp_ms: int):
    global latest_hand_result
    latest_hand_result = result


def to_pixel(landmark, frame_w, frame_h):
    return int(landmark.x * frame_w), int(landmark.y * frame_h)


def detect_faces(frame, detector):
    frame_h, frame_w = frame.shape[:2]
    detector.setInputSize((frame_w, frame_h))
    _, faces = detector.detect(frame)

    boxes = []
    if faces is not None:
        for face in faces:
            x, y, w, h = face[:4].astype(int)
            boxes.append((x, y, w, h))
    return boxes


def draw_faces(frame, boxes):
    for x, y, w, h in boxes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)


def draw_hand_landmarks(frame, hand_landmarks, frame_w, frame_h):
    for landmark in hand_landmarks:
        pixel_x, pixel_y = to_pixel(landmark, frame_w, frame_h)
        cv2.circle(frame, (pixel_x, pixel_y), 2, (255, 0, 0), 2)


def draw_hand_connections(frame, hand_landmarks, frame_w, frame_h):
    hand_connections = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS

    for connection in hand_connections:
        start_landmark = hand_landmarks[connection.start]
        end_landmark = hand_landmarks[connection.end]

        start_x, start_y = to_pixel(start_landmark, frame_w, frame_h)
        end_x, end_y = to_pixel(end_landmark, frame_w, frame_h)

        cv2.line(frame, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)


def move_mouse_with_index(hand_landmarks, screen_w, screen_h):
    index_tip = hand_landmarks[8]
    screen_mouse_x = int((1 - index_tip.x) * screen_w)
    screen_mouse_y = int(index_tip.y * screen_h)
    return (screen_mouse_x, screen_mouse_y)


def smooth_cursor(target_x, target_y, alpha=CURSOR_SMOOTHING_ALPHA):
    global last_cursor_pos
    if last_cursor_pos is None:
        last_cursor_pos = (target_x, target_y)
        return last_cursor_pos
    
    smoothed_x = int(alpha * target_x + (1 - alpha) * last_cursor_pos[0])
    smoothed_y = int(alpha * target_y + (1 - alpha) * last_cursor_pos[1])
    last_cursor_pos = (smoothed_x, smoothed_y)
    return last_cursor_pos


def maybe_click_stabilized(distance, threshold, current_time, last_click_time, cooldown_ms):
    global was_pinched
    upper_threshold = threshold * PINCH_HYSTERESIS_RATIO
    
    # Transition from not-pinched to pinched
    if not was_pinched and distance < threshold:
        was_pinched = True
        if (current_time - last_click_time) > cooldown_ms:
            pyautogui.click()
            return current_time
    # Transition out of pinch (hysteresis)
    elif was_pinched and distance > upper_threshold:
        was_pinched = False
    
    return last_click_time


def get_pinch_distance(hand_landmarks, frame_w, frame_h):
    thumb_tip = hand_landmarks[4]
    index_tip = hand_landmarks[8]

    thumb_x, thumb_y = to_pixel(thumb_tip, frame_w, frame_h)
    index_x, index_y = to_pixel(index_tip, frame_w, frame_h)

    return math.hypot(thumb_x - index_x, thumb_y - index_y)


def get_pinch_threshold(hand_landmarks, frame_w, frame_h):
    wrist = hand_landmarks[0]
    index_base = hand_landmarks[5]

    wrist_x, wrist_y = to_pixel(wrist, frame_w, frame_h)
    index_base_x, index_base_y = to_pixel(index_base, frame_w, frame_h)

    hand_size = math.hypot(index_base_x - wrist_x, index_base_y - wrist_y)
    return hand_size * 0.25


def maybe_click(distance, threshold, current_time, last_click_time, cooldown_ms):
    if distance < threshold and (current_time - last_click_time) > cooldown_ms:
        pyautogui.click()
        return current_time
    return last_click_time

# creating hand landmarker instance with LIVE_STREAM mode
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_hands=4,
    result_callback=print_result)
with HandLandmarker.create_from_options(options) as landmarker: # initalizes the landmarker

# create the hands object for real-time processing
# The 'with' statement ensures resources are released automatically

    # keeping track of start time
    start_time = time.monotonic()
    last_click_time = 0

    while True:
        elapsed_seconds = time.monotonic() - start_time
        frame_timestamp_ms = int(elapsed_seconds * 1000)
        # tries to read one frame from cammera
        success, frame = cap.read()

        # success is whether it worked or not
        if not success:
            break

        frame_h, frame_w = frame.shape[:2]
        last_boxes = detect_faces(frame, detector)
        draw_faces(frame, last_boxes)

        # convert the frame from OpenCV into mediapipes image object
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

        # prefrom hand landmarks detections on the provided single image
        # the hand landmarker must be created with the livestream mode
        landmarker.detect_async(mp_image, frame_timestamp_ms)

        if latest_hand_result is not None and latest_hand_result.hand_landmarks:
            for hand_landmarks in latest_hand_result.hand_landmarks:  # type: ignore
                draw_hand_landmarks(frame, hand_landmarks, frame_w, frame_h)
                cursor_pos = move_mouse_with_index(hand_landmarks, screen_w, screen_h)
                smoothed_pos = smooth_cursor(cursor_pos[0], cursor_pos[1])
                pyautogui.moveTo(smoothed_pos[0], smoothed_pos[1])

                distance = get_pinch_distance(hand_landmarks, frame_w, frame_h)
                threshold = get_pinch_threshold(hand_landmarks, frame_w, frame_h)

                last_click_time = maybe_click_stabilized(
                    distance=distance,
                    threshold=threshold,
                    current_time=frame_timestamp_ms,
                    last_click_time=last_click_time,
                    cooldown_ms=500,
                )

                draw_hand_connections(frame, hand_landmarks, frame_w, frame_h)

        # frame is the actual frame
        cv2.imshow("Webcam", frame)

        if cv2.waitKey(1) == 27: # 27 is the ESC key
            break

cap.release()
cv2.destroyAllWindows()