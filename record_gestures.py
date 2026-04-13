import os
import json
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2 
import numpy as np
from PIL import Image

# --- Configuration ---
model_path = 'models/hand_landmarker.task'
RECORD_GIF = True  # Flag to record image fragments as GIF
RECORD_INTERVAL = 5 # Record every Nth frame

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
mp_hands = mp.tasks.vision.HandLandmarksConnections
mp_drawing = mp.tasks.vision.drawing_utils
mp_drawing_styles = mp.tasks.vision.drawing_styles
MARGIN = 20  # pixels for bounding box padding
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) 

# --- Setup Data Directory ---
user_input = input("Enter the gesture class/folder number (e.g., 1, 2): ")
save_dir = os.path.join("data", str(user_input))
os.makedirs(save_dir, exist_ok=True)

# Determine the starting index based on existing files to prevent overwriting
existing_files = [f for f in os.listdir(save_dir) if f.startswith('landmarks_') and f.endswith('.json')]
gesture_index = len(existing_files)

# --- State Variables ---
is_recording = False
frame_counter = 0
recorded_landmarks = []
recorded_crops = []

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE)

cap = cv2.VideoCapture(0)

# OPTIMIZATION: Moved model initialization OUTSIDE the loop
with HandLandmarker.create_from_options(options) as landmarker:
    while True:
        suc, img = cap.read()
        if not suc:
            break
            
        img = cv2.flip(img, 1)
        h, w, _ = img.shape
        annotated_image = img.copy()

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
        detection_result = landmarker.detect(mp_image)

        # Draw landmarks and process recording
        if detection_result.hand_landmarks:
            for idx in range(len(detection_result.hand_landmarks)):
                hand_landmarks = detection_result.hand_landmarks[idx]
                handedness = detection_result.handedness[idx]

                # Draw on annotated image
                mp_drawing.draw_landmarks(
                    annotated_image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

                # Calculate bounding box for text and cropping
                x_coords = [landmark.x for landmark in hand_landmarks]
                y_coords = [landmark.y for landmark in hand_landmarks]
                
                x_min = max(0, int(min(x_coords) * w) - MARGIN)
                y_min = max(0, int(min(y_coords) * h) - MARGIN)
                x_max = min(w, int(max(x_coords) * w) + MARGIN)
                y_max = min(h, int(max(y_coords) * h) + MARGIN)

                cv2.putText(annotated_image, f"{handedness[0].category_name}",
                        (x_min, y_min - MARGIN), cv2.FONT_HERSHEY_DUPLEX,
                        FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

                # Recording logic for the *first* detected hand
                if is_recording and idx == 0:
                    if frame_counter % RECORD_INTERVAL == 0:
                        # 1. Save Landmarks as a list of dicts
                        frame_lmks = [{"x": lm.x, "y": lm.y, "z": lm.z} for lm in hand_landmarks]
                        recorded_landmarks.append(frame_lmks)
                        
                        # 2. Save Crop for GIF
                        if RECORD_GIF:
                            hand_crop = img[y_min:y_max, x_min:x_max]
                            # Ensure crop is valid before appending
                            if hand_crop.size > 0: 
                                recorded_crops.append(hand_crop)

        # Visual indicator for recording state
        if is_recording:
            cv2.circle(annotated_image, (30, 30), 10, (0, 0, 255), -1)
            cv2.putText(annotated_image, f"REC (Frames: {len(recorded_landmarks)})", 
                        (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            frame_counter += 1

        cv2.imshow("Webcam Gesture Recorder", annotated_image)

        key = cv2.waitKey(1) & 0xFF

        # SPACEBAR (Toggle Recording)
        if key == 32: 
            is_recording = not is_recording
            
            if is_recording:
                print(f"Started recording gesture #{gesture_index}...")
                recorded_landmarks = []
                recorded_crops = []
                frame_counter = 0
            else:
                print(f"Stopped. Saving gesture #{gesture_index} data...")
                
                # Save Landmarks to JSON
                if recorded_landmarks:
                    json_path = os.path.join(save_dir, f"landmarks_{gesture_index}.json")
                    with open(json_path, 'w') as f:
                        json.dump(recorded_landmarks, f, indent=4)
                    
                    # Save Crop Fragments to GIF
                    if RECORD_GIF and recorded_crops:
                        gif_path = os.path.join(save_dir, f"gesture_{gesture_index}.gif")
                        
                        # Convert BGR (OpenCV) to RGB (Pillow) and resize uniformly if needed
                        pil_images = [Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)) for crop in recorded_crops]
                        
                        # Make all crops the size of the first frame's crop so GIF builds cleanly
                        base_size = pil_images[0].size
                        resized_images = [img.resize(base_size) for img in pil_images]
                        
                        resized_images[0].save(gif_path, save_all=True, append_images=resized_images[1:], duration=100, loop=0)
                    
                    print(f"✅ Data saved to: {save_dir}")
                    gesture_index += 1
                else:
                    print("No hands detected. Data not saved.")
        
        # 'Q' or 'ESC' to Quit
        elif key == ord('q') or key == 27:
            break

cap.release()
cv2.destroyAllWindows()