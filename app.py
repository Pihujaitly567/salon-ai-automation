import cv2
import time
import os
import threading
from flask import Flask, render_template, Response, jsonify, request
from werkzeug.utils import secure_filename

import config
from tracker import SalonTracker, save_zones_database

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = config.UPLOAD_FOLDER
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

tracker = SalonTracker()

# Thread frame buffers & synchronization
latest_frame = None
latest_clean_frame = None
frame_lock = threading.Lock()
video_reload_flag = False


def video_processing_worker():
    """Background thread that captures video frames and runs pose analytics."""
    global latest_frame, latest_clean_frame, video_reload_flag

    while True:
        target_video = tracker.current_video_file
        video_reload_flag = False

        cap = cv2.VideoCapture(target_video)
        if not cap.isOpened():
            print(f"Failed to open video source: {target_video}. Retrying...")
            time.sleep(1.0)
            continue

        print(f"Analyzing video stream: {os.path.basename(target_video)}")

        while cap.isOpened() and not video_reload_flag:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            with frame_lock:
                latest_clean_frame = frame.copy()

            processed_frame = tracker.process_frame(frame)
            display_frame = cv2.resize(processed_frame, (800, 600))
            success, buffer = cv2.imencode('.jpg', display_frame)

            if success:
                with frame_lock:
                    latest_frame = buffer.tobytes()

            time.sleep(0.035)

        cap.release()


# ----------------------------------------------------------------------
# Routes & API Endpoints
# ----------------------------------------------------------------------

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    def frame_generator():
        while True:
            with frame_lock:
                if latest_frame is not None:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + latest_frame + b'\r\n')
            time.sleep(0.04)

    return Response(frame_generator(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/snapshot')
def api_snapshot():
    with frame_lock:
        if latest_clean_frame is not None:
            ret, buffer = cv2.imencode('.jpg', latest_clean_frame)
            if ret:
                return Response(buffer.tobytes(), mimetype='image/jpeg')
    return Response(b'', mimetype='image/jpeg')


@app.route('/api/data')
def api_data():
    return jsonify(tracker.get_api_payload())


@app.route('/api/select_video', methods=['POST'])
def select_video():
    global video_reload_flag
    data = request.get_json() or {}
    video_name = data.get("video")

    if not video_name:
        return jsonify({"error": "Video parameter missing"}), 400

    target_path = os.path.join(config.UPLOAD_FOLDER, secure_filename(video_name))
    if not os.path.exists(target_path):
        return jsonify({"error": "Selected video file does not exist"}), 404

    tracker.current_video_file = target_path
    tracker.reset_state()
    video_reload_flag = True

    return jsonify({"success": True, "current_video": video_name})


@app.route('/api/upload_video', methods=['POST'])
def upload_video():
    global video_reload_flag
    if 'file' not in request.files:
        return jsonify({"error": "No file payload received"}), 400

    uploaded_file = request.files['file']
    if uploaded_file.filename == '':
        return jsonify({"error": "Filename is empty"}), 400

    filename = secure_filename(uploaded_file.filename)
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    uploaded_file.save(save_path)

    tracker.current_video_file = save_path
    tracker.reset_state()
    video_reload_flag = True

    return jsonify({"success": True, "filename": filename})


@app.route('/api/save_zones', methods=['POST'])
def save_zones():
    data = request.get_json() or {}
    video_name = data.get("video") or os.path.basename(tracker.current_video_file)
    zones = data.get("zones")

    if not zones:
        return jsonify({"error": "Zone payload empty"}), 400

    tracker.zones_db[video_name] = zones
    save_zones_database(tracker.zones_db)
    tracker.sync_stations()

    return jsonify({"success": True, "saved_for": video_name})


@app.route('/api/get_zones', methods=['GET'])
def get_zones():
    video_name = request.args.get("video", os.path.basename(tracker.current_video_file))
    zones = tracker.zones_db.get(video_name, tracker.zones_db.get("default", {}))
    return jsonify({"video": video_name, "zones": zones})


@app.route('/api/toggle_skeletons', methods=['POST'])
def toggle_skeletons():
    tracker.show_skeletons = not tracker.show_skeletons
    return jsonify({"show_skeletons": tracker.show_skeletons})


if __name__ == '__main__':
    worker = threading.Thread(target=video_processing_worker, daemon=True)
    worker.start()

    app.run(host=config.HOST, port=config.PORT, debug=False)
