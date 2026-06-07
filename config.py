import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_DIR = os.path.join(BASE_DIR, "input")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
MODELS_DIR = os.path.join(BASE_DIR, "models")

VIDEO_FILENAME = "crowd_video7.mp4"
VIDEO_PATH = os.path.join(INPUT_DIR, VIDEO_FILENAME)

OUTPUT_FILENAME = "crowd_output.mp4"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)

REPORT_FILENAME = "crowd_report.csv"
REPORT_PATH = os.path.join(OUTPUT_DIR, REPORT_FILENAME)

HIGH_RISK_DIR = os.path.join(OUTPUT_DIR, "high_risk_frames")

# ---------------- PROCESSING ----------------
REAL_TIME_MODE = True

NORMAL_PROCESS_EVERY_N_FRAMES = 5
DANGER_PROCESS_EVERY_N_FRAMES = 2
PROCESS_EVERY_N_FRAMES = NORMAL_PROCESS_EVERY_N_FRAMES

PRINT_EVERY_N_FRAMES = 5

# ---------------- YOLO ----------------
YOLO_MODEL_NAME = "yolov8m.pt"
YOLO_MODEL_PATH = os.path.join(MODELS_DIR, YOLO_MODEL_NAME)

PERSON_CLASS_ID = 0

CONFIDENCE_THRESHOLD = 0.10
YOLO_IMAGE_SIZE = 1280

# Keep tiled detection ON for CCTV crowd videos
ENABLE_TILED_DETECTION = True

TILE_ROWS = 2
TILE_COLS = 3
TILE_OVERLAP = 0.25
TILE_CONFIDENCE = 0.07
TILE_IMAGE_SIZE = 960

DUPLICATE_IOU_THRESHOLD = 0.35

MIN_PERSON_BOX_WIDTH = 3
MIN_PERSON_BOX_HEIGHT = 8

MAX_DETECTIONS = 5000

# ---------------- ROI ----------------
DRAW_ROI_BOUNDARY = False
ROI_BOUNDARY_COLOR = (255, 255, 0)

# ---------------- DENSITY ----------------
DENSITY_LOW_MAX = 35
DENSITY_MEDIUM_MAX = 90

# ---------------- ZONES ----------------
ZONE_ROWS = 2
ZONE_COLS = 3

# Reduced from 18 so zones with 10-12 people also become crowded
ZONE_CROWD_THRESHOLD = 10

SHOW_ZONE_LINES = True
SHOW_ZONE_TEXT = True
SHOW_ZONE_TOTAL = True

# ---------------- MOTION ----------------
MOTION_FAST_MIN = 0.40
MOTION_ABNORMAL_MIN = 1.20

CHAOS_MOTION_MIN = 0.35
ACCELERATION_THRESHOLD = 0.18
DIRECTION_CONSISTENCY_THRESHOLD = 0.60
COUNT_INSTABILITY_RATIO = 0.25

# ---------------- STAMPEDE ----------------
SURGE_COUNT_RATIO = 1.20
DENSE_CROWD_IS_NOT_STAMPEDE = True

# ---------------- RISK LEVELS ----------------
RISK_SAFE = "Safe"
RISK_DENSE = "Dense Crowd"
RISK_MEDIUM = "Medium Risk"
RISK_HIGH = "High Risk"

# ---------------- ALERT ----------------
ALERT_FRAME_THRESHOLD = 4
ALERT_TEXT = "POSSIBLE STAMPEDE RISK!"
ALERT_COLOR = (0, 0, 255)
ENABLE_SOUND = False

# ---------------- HEATMAP ----------------
HEATMAP_RADIUS = 10
HEATMAP_ALPHA = 0.05

# ---------------- OUTPUT ----------------
SAVE_OUTPUT = False
ENABLE_REPORT = False
SAVE_HIGH_RISK_FRAMES = False

HIGH_RISK_SAVE_INTERVAL = 30
PRINT_RISK_REASONS_IN_LOGS = True

# ---------------- DISPLAY ----------------
SHOW_LIVE = True

BBOX_COLOR = (0, 255, 120)
TEXT_COLOR = (255, 255, 255)
FONT_SCALE = 0.60
BBOX_THICKNESS = 1

DRAW_CENTER_DOTS = False
CENTER_DOT_COLOR = (0, 255, 255)
CENTER_DOT_RADIUS = 1

SHOW_RISK_REASONS = True