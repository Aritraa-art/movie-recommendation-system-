import cv2
import mediapipe as mp
import math

# Initialize Mediapipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
mp_draw = mp.solutions.drawing_utils

# Eye landmark indices (Mediapipe)
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def eye_aspect_ratio(eye_landmarks, landmarks, frame_shape):
    # Convert normalized landmarks to pixel coordinates
    coords = [(int(landmarks.landmark[i].x * frame_shape[1]),
               int(landmarks.landmark[i].y * frame_shape[0])) for i in eye_landmarks]
    
    # Compute EAR
    A = math.dist(coords[1], coords[5])
    B = math.dist(coords[2], coords[4])
    C = math.dist(coords[0], coords[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Blink detection threshold
EAR_THRESHOLD = 0.25

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    
    if results.multi_face_landmarks:
        for landmarks in results.multi_face_landmarks:
            # Draw landmarks (optional)
            mp_draw.draw_landmarks(frame, landmarks, mp_face_mesh.FACEMESH_TESSELATION)
            
            left_ear = eye_aspect_ratio(LEFT_EYE, landmarks, frame.shape)
            right_ear = eye_aspect_ratio(RIGHT_EYE, landmarks, frame.shape)
            avg_ear = (left_ear + right_ear) / 2.0
            
            if avg_ear < EAR_THRESHOLD:
                cv2.putText(frame, "BLINK", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    
    cv2.imshow("Blink Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
