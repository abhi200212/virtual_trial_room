# project
import cv2
import mediapipe as mp
import numpy as np
import os


GARMENT_PATH = "shirt.png"   # Replace with path to your transparent PNG shirt
SNAPSHOT_DIR = "snapshots"

# Create snapshot directory
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

# Load transparent shirt image
garment = cv2.imread(GARMENT_PATH, cv2.IMREAD_UNCHANGED)  # Keep alpha channel
if garment is None:
    raise FileNotFoundError("Garment PNG not found! Please check GARMENT_PATH")

# Initialize mediapipe pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils


# Helper functions
def overlay_transparent(background, overlay, x, y, overlay_size=None):
    """Overlay transparent PNG on background at (x, y)."""
    bg = background.copy()

 if overlay_size is not None:
        overlay = cv2.resize(overlay, overlay_size, interpolation=cv2.INTER_AREA)

 h, w = overlay.shape[:2]
    if x >= bg.shape[1] or y >= bg.shape[0]:
        return bg

   # Clip overlay region
   h = min(h, bg.shape[0] - y)
    w = min(w, bg.shape[1] - x)
    if h <= 0 or w <= 0:
        return bg

  overlay_img = overlay[:h, :w, :3]
    mask = overlay[:h, :w, 3:] / 255.0

  bg[y:y+h, x:x+w] = (1.0 - mask) * bg[y:y+h, x:x+w] + mask * overlay_img
    return bg


cap = cv2.VideoCapture(0)
overlay_on = True
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

frame = cv2.flip(frame, 1)
 rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
 results = pose.process(rgb)

  if results.pose_landmarks and overlay_on:
        h, w, _ = frame.shape

    # Get shoulder and hip points
left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]

        # Convert normalized coords to pixel coords
 l_sh = np.array([int(left_shoulder.x * w), int(left_shoulder.y * h)])
        r_sh = np.array([int(right_shoulder.x * w), int(right_shoulder.y * h)])
        l_hip = np.array([int(left_hip.x * w), int(left_hip.y * h)])
        r_hip = np.array([int(right_hip.x * w), int(right_hip.y * h)])

        # Estimate center, width and height for garment
shoulders_center = (l_sh + r_sh) // 2
        hips_center = (l_hip + r_hip) // 2
        garment_width = int(np.linalg.norm(r_sh - l_sh) * 2.0)
        garment_height = int(np.linalg.norm(hips_center - shoulders_center) * 1.2)

        # Top-left corner for placing garment
x1 = shoulders_center[0] - garment_width // 2
        y1 = shoulders_center[1]

        # Overlay garment
 frame = overlay_transparent(frame, garment, x1, y1, (garment_width, garment_height))

cv2.imshow("Virtual Trial Room", frame)
    key = cv2.waitKey(1) & 0xFF

if key == ord("q"):
        break
    elif key == ord("v"):
        overlay_on = not overlay_on
    elif key == ord("s"):
        path = os.path.join(SNAPSHOT_DIR, f"snapshot_{frame_count}.png")
        cv2.imwrite(path, frame)
        print(f"Snapshot saved: {path}")

 frame_count += 1

cap.release()
cv2.destroyAllWindows()
