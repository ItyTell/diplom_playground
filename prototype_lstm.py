import cv2
import torch
import torch.nn as nn
import numpy as np
import mediapipe as mp
from collections import deque

MODEL_PATH = 'gesture_lstm.pth'
MEDIAPIPE_MODEL_PATH = 'hand_landmarker.task'
MAX_SEQ_LENGTH = 8
INPUT_SIZE = 21 * 3 
HIDDEN_SIZE = 64
NUM_LAYERS = 2
NUM_CLASSES = 10 


CLASS_MAP = {i: str(i + 1) for i in range(NUM_CLASSES)}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GestureLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(GestureLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                            batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :] 
        out = self.fc(out)
        return out

model = GestureLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval() #
print("PyTorch Model loaded successfully.")

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
mp_drawing = mp.tasks.vision.drawing_utils
mp_hands = mp.tasks.vision.HandLandmarksConnections
mp_drawing_styles = mp.tasks.vision.drawing_styles

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MEDIAPIPE_MODEL_PATH),
    running_mode=VisionRunningMode.IMAGE)

landmark_buffer = deque(maxlen=MAX_SEQ_LENGTH)
current_prediction = "Waiting for hand..."

cap = cv2.VideoCapture(0)

with HandLandmarker.create_from_options(options) as landmarker:
    while True:
        suc, img = cap.read()
        if not suc:
            break
            
        img = cv2.flip(img, 1)
        annotated_image = img.copy()

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
        detection_result = landmarker.detect(mp_image)

        if detection_result.hand_landmarks:
            hand_landmarks = detection_result.hand_landmarks[0]
            
            mp_drawing.draw_landmarks(
                annotated_image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            frame_features = []
            for lm in hand_landmarks:
                frame_features.extend([lm.x, lm.y, lm.z])
                
            landmark_buffer.append(frame_features)
            
            if len(landmark_buffer) == MAX_SEQ_LENGTH:
                input_tensor = torch.tensor(np.array(landmark_buffer), dtype=torch.float32).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    outputs = model(input_tensor)
                    
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)
                    confidence, predicted_class = torch.max(probabilities, 1)
                    
                    gesture_idx = predicted_class.item()
                    conf_score = confidence.item()
                    
                    if conf_score > 0.60:
                        gesture_name = CLASS_MAP.get(gesture_idx, f"Unknown ({gesture_idx})")
                        current_prediction = f"{gesture_name} ({conf_score*100:.1f}%)"
                    else:
                        current_prediction = "Uncertain..."

        else:
            landmark_buffer.clear()
            current_prediction = "No hand detected"

        cv2.rectangle(annotated_image, (0, 0), (640, 60), (0, 0, 0), -1)
        cv2.putText(annotated_image, current_prediction, (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("LSTM Gesture Prototype", annotated_image)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

cap.release()
cv2.destroyAllWindows()