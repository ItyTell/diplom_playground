import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2 
import pyautogui
import time

model_path = f"C:\Projects\diplom_playground\custom_model_mediapipeapi.task"

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a gesture recognizer instance with the image mode:
options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE)

mp_hands = mp.tasks.vision.HandLandmarksConnections
mp_drawing = mp.tasks.vision.drawing_utils
mp_drawing_styles = mp.tasks.vision.drawing_styles


def draw_gesture_and_landmarks_on_image(image, gesture, multi_hand_landmarks):
    """
    Draws the gesture category, its score, and hand landmarks on a single image.
    
    Args:
        image: The input image (either a MediaPipe Image object or a NumPy array).
        gesture: The top gesture object containing .category_name and .score.
        multi_hand_landmarks: A list of hand landmarks for the image.
        
    Returns:
        A NumPy array of the annotated image.
    """
    # Convert MediaPipe Image to a NumPy array if necessary, and create a copy
    if hasattr(image, 'numpy_view'):
        annotated_image = image.numpy_view().copy()
    else:
        annotated_image = image.copy()

    # 1. Draw hand landmarks
    if multi_hand_landmarks:
        for hand_landmarks in multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                annotated_image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

    # 2. Draw gesture name and score
    if gesture:
        title = f"{gesture.category_name[10:]} ({gesture.score:.2f})" #del train_val_
        
        # OpenCV text settings
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_color = (0, 255, 0)  # Green text (BGR format)
        thickness = 2
        position = (20, 50)       # Top-left corner coordinates
        
        cv2.putText(
            annotated_image, 
            title, 
            position, 
            font, 
            font_scale, 
            font_color, 
            thickness, 
            cv2.LINE_AA
        )

    return annotated_image


def control_computer(gesture):
    if gesture.category_name[10:] == "ok":
        pyautogui.press('right')
        time.sleep(0.5)
    if gesture.category_name[10:] == "peace":
        pyautogui.press('left')
        time.sleep(0.5)



cap = cv2.VideoCapture(0)
while True:
    suc, img = cap.read()
    img = cv2.flip(img, 1)
    with GestureRecognizer.create_from_options(options) as recognizer:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
        gestures = recognizer.recognize(mp_image)
        if len(gestures.gestures) > 0:
            top_gesture = gestures.gestures[0][0]
            hand_landmarks = gestures.hand_landmarks
            if top_gesture.score > 0.9:
                mp_image = draw_gesture_and_landmarks_on_image(img, top_gesture, hand_landmarks)
                control_computer(top_gesture)
            else:
                mp_image = img
        else:
            mp_image = img


    cv2.imshow("Veb",  mp_image)


    key = cv2.waitKey(1)

    if key == ord('q') or key == 27:
        break





