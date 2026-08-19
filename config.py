import os

# Server & storage settings
HOST = os.environ.get("FLASK_HOST", "0.0.0.0")
PORT = int(os.environ.get("FLASK_PORT", 5004))
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "videos")
ZONES_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "zones_config.json")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "yolov8n-pose.pt")

DEFAULT_VIDEO = os.path.join(UPLOAD_FOLDER, "salon_video_1.mp4")

# Detection and tracking parameters
CONFIDENCE_THRESHOLD = 0.35
SITTING_ASPECT_RATIO_FALLBACK = 0.62
STATION_OCCUPIED_STREAK = 12
STATION_VACANT_STREAK = 375        # ~15 seconds tolerance for empty chair before vacating

# Waiting Lounge Occlusion & Persistence Settings
WAITING_SLOT_MAX_DISTANCE = 95.0
WAITING_SLOT_MIN_SEPARATION = 75.0
WAITING_SLOT_DROP_FRAMES = 1250     # ~50 seconds occlusion tolerance
WAITING_SLOT_ACTIVE_WINDOW = 1000   # ~40 seconds live queue persistence

