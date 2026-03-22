import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0) # 0 means default webcam
success, frame = cap.read() # reads one frame
last_boxes = []
latest_hand_result = None

if not success:
    print("could not read from camera")
    cap.release()
    exit()

h, w = frame.shape[:2] # image height/width 
# frame is a numpy array and opencv usually stores that as (height, width, channels)
# thats why we use .shape[:2] we only want the first 2 values of the frame

detector = cv2.FaceDetectorYN.create("face_detection_yunet_2023mar.onnx", "", (w, h))

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

def print_result(result, output_image: mp.Image, timestamp_ms: int):
    global latest_hand_result
    latest_hand_result = result

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

    while True:
        elapsed_seconds = time.monotonic() - start_time
        frame_timestamp_ms = int(elapsed_seconds * 1000)
        # tries to read one frame from cammera
        success, frame = cap.read()

        # success is whether it worked or not
        if not success:
            break

        frame_h, frame_w = frame.shape[:2]
        detector.setInputSize((frame_w, frame_h))
        _, faces = detector.detect(frame) # runs detection
        # faces is a array of numbers
        # the first 4 values being x, y, w, h

        last_boxes = []

        if faces is not None:
            for face in faces:
                x, y, w, h = face[:4].astype(int)
                last_boxes.append((x, y, w, h))

        for x, y, w, h in last_boxes:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # convert the frame from OpenCV into mediapipes image object
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

        # prefrom hand landmarks detections on the provided single image
        # the hand landmarker must be created with the livestream mode
        landmarker.detect_async(mp_image, frame_timestamp_ms)

        if latest_hand_result is not None:
            if latest_hand_result.hand_landmarks:
                for hand_landmarks in latest_hand_result.hand_landmarks: # type: ignore
                    # stores the first normalizedlandmark
                    for r in hand_landmarks: # type: ignore becausae its gonna be unbound depending on how long i keep the camera on
                        pixel_x = int(r.x * frame_w)
                        pixel_y = int(r.y * frame_h)          

                        cv2.circle(frame, (pixel_x, pixel_y), 2, (255, 0, 0), 2)

                    hand_connections = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS

                    for connection in hand_connections:
                        start_idx = connection.start
                        end_idx = connection.end
                        
                        start_landmark = hand_landmarks[start_idx]
                        end_landmark = hand_landmarks[end_idx]

                        start_x = int(start_landmark.x * frame_w)
                        start_y = int(start_landmark.y * frame_h)
                        end_x = int(end_landmark.x * frame_w)
                        end_y = int(end_landmark.y * frame_h)

                        cv2.line(frame, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)

        # frame is the actual frame
        cv2.imshow("Webcam", frame)

        if cv2.waitKey(1) == 27: # 27 is the ESC key
            break

cap.release()
cv2.destroyAllWindows()