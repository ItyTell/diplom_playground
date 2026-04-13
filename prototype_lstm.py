import cv2
import torch
import torch.nn as nn
import numpy as np
import mediapipe as mp
from collections import deque

MODEL_PATH = 'models/gesture_lstm2.0.pth'
MEDIAPIPE_MODEL_PATH = 'models/hand_landmarker.task'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Model architecture (must match training) ───────────────────────

class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1, bias=False),
        )

    def forward(self, lstm_out):
        scores = self.attn(lstm_out)
        weights = torch.softmax(scores, dim=1)
        context = (lstm_out * weights).sum(dim=1)
        return context, weights.squeeze(-1)


class GestureLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.4):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.input_proj = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
        )

        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )

        self.layer_norm = nn.LayerNorm(hidden_size * 2)
        self.attention = TemporalAttention(hidden_size * 2)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x):
        x = self.input_proj(x)
        lstm_out, _ = self.lstm(x)
        lstm_out = self.layer_norm(lstm_out)
        context, _ = self.attention(lstm_out)
        logits = self.classifier(context)
        return logits


# ── Load checkpoint ────────────────────────────────────────────────

checkpoint = torch.load(MODEL_PATH, map_location=device)

MAX_SEQ_LENGTH = checkpoint['max_seq_length']
INPUT_SIZE     = checkpoint['input_size']
HIDDEN_SIZE    = checkpoint['hidden_size']
NUM_LAYERS     = checkpoint['num_layers']
NUM_CLASSES    = checkpoint['num_classes']
LABEL_NAMES    = checkpoint['label_names']    # {new_label: original_gesture_id}

GESTURE_DISPLAY = {
    1: "Right  →",
    2: "Left  ←",
    6: "Slide Right  ⇒",
    7: "Slide Left  ⇐",
    8: "Closer  ↓",
    9: "Scale Up  ↑",
}

model = GestureLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print(f"Model loaded — {NUM_CLASSES} active gestures, seq_length={MAX_SEQ_LENGTH}")

# ── MediaPipe setup ────────────────────────────────────────────────

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

# ── Inference loop ─────────────────────────────────────────────────

landmark_buffer = deque(maxlen=MAX_SEQ_LENGTH)
current_prediction = "Waiting for hand..."

consecutive_frames = 0
current_candidate = None
confirmed_gesture = ""

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
                input_tensor = (torch.tensor(np.array(landmark_buffer),
                                             dtype=torch.float32)
                                .unsqueeze(0).to(device))

                with torch.no_grad():
                    outputs = model(input_tensor)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)
                    confidence, predicted_class = torch.max(probabilities, 1)

                    new_label = predicted_class.item()
                    conf_score = confidence.item()

                    if conf_score >= 0.90:
                        if new_label == current_candidate:
                            consecutive_frames += 1
                        else:
                            current_candidate = new_label
                            consecutive_frames = 1

                        if consecutive_frames >= 3:
                            orig_id = LABEL_NAMES[new_label]
                            display = GESTURE_DISPLAY.get(orig_id, f"Gesture {orig_id}")
                            confirmed_gesture = display
                            current_prediction = f"Tracking... ({conf_score*100:.1f}%)"
                        else:
                            current_prediction = f"Verifying... ({conf_score*100:.1f}%)"
                    else:
                        current_candidate = None
                        consecutive_frames = 0
                        confirmed_gesture = ""
                        current_prediction = "Uncertain..."

        else:
            landmark_buffer.clear()
            current_candidate = None
            consecutive_frames = 0
            confirmed_gesture = ""
            current_prediction = "No hand detected"

        cv2.rectangle(annotated_image, (0, 0), (640, 60), (0, 0, 0), -1)
        cv2.putText(annotated_image, current_prediction, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        if confirmed_gesture:
            cv2.putText(annotated_image, confirmed_gesture, (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 4, cv2.LINE_AA)

        cv2.imshow("LSTM Gesture Prototype", annotated_image)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

cap.release()
cv2.destroyAllWindows()