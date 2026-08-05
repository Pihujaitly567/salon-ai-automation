import cv2
import numpy as np
import time
import math
import threading
from flask import Flask, render_template, Response, jsonify
from ultralytics import YOLO

app = Flask(__name__)

# --- LOAD YOLO MODEL ---
model = YOLO("yolov8n.pt")

# --- GLOBAL STREAMS LOCKS & BUFFERS ---
latest_frame = None
frame_lock = threading.Lock()

# --- STATE DATABASE ---
CHAIRS = {
    1: {"label": "Station 01", "status": "Vacant", "current_customer": None, "occupancy_start": None, "durations": []},
    2: {"label": "Station 02", "status": "Vacant", "current_customer": None, "occupancy_start": None, "durations": []},
    3: {"label": "Station 03", "status": "Vacant", "current_customer": None, "occupancy_start": None, "durations": []}
}

BARBERS = {
    1: {"name": "Alex (Senior Barber)", "status": "Idle", "current_chair": None, "service_start": None, "total_service_time": 0.0},
    2: {"name": "Jordan (Stylist)", "status": "Idle", "current_chair": None, "service_start": None, "total_service_time": 0.0},
    3: {"name": "Taylor (Color Specialist)", "status": "Idle", "current_chair": None, "service_start": None, "total_service_time": 0.0}
}

metrics = {
    "total_entries": 0,
    "active_barbers": 3,
}

# AI classification & queue states
tracked_people = set()
stylists = set()
waiting_clients = {}    # track_id -> wait_start_time
wait_durations = []     # List of historical wait durations
waiting_streaks = {}    # track_id -> {"occupied": 0, "vacant": 0}

# Streak counters for smoothing out detections (flickering prevention)
station_streaks = {
    1: {"occupied": 0, "vacant": 0},
    2: {"occupied": 0, "vacant": 0}
}

# --- BACKGROUND AI PROCESSING LOOP (YOLOv8) ---
def process_video():
    global latest_frame, tracked_people, stylists, waiting_clients, wait_durations, waiting_streaks, station_streaks
    
    cap = cv2.VideoCapture("videos/salon_video_1.mp4")
    if not cap.isOpened():
        print("Error: Could not open videos/salon_video_1.mp4")
        return
        
    frame_counter = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            # Reset active state timers when the video loops
            CHAIRS[1]["occupancy_start"] = None
            CHAIRS[2]["occupancy_start"] = None
            BARBERS[1]["service_start"] = None
            BARBERS[2]["service_start"] = None
            station_streaks[1] = {"occupied": 0, "vacant": 0}
            station_streaks[2] = {"occupied": 0, "vacant": 0}
            
            # Clear active wait list and reset classifier counters
            waiting_clients.clear()
            waiting_streaks.clear()
            stylists.clear()
            frame_counter = 0
            continue
            
        frame_counter += 1
        fh, fw, fc = frame.shape
        
        # Run YOLOv8 + ByteTrack
        results = model.track(frame, persist=True, classes=[0], verbose=False)
        
        s1_detected = False
        s2_detected = False
        waiting_detected = set()
        
        if results and len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                track_id = int(box.id[0]) if box.id is not None else None
                if track_id is None:
                    continue
                
                # Center point of bounding box
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                
                # STYLIST CLASSIFICATION:
                # In the first 50 frames, anyone detected inside Station 01 or Station 02 zones
                # is flagged as a staff stylist. This filters them out from client stats.
                if frame_counter < 50:
                    if (100 <= cx <= 380 and 120 <= cy <= 360) or (340 <= cx <= 450 and 140 <= cy <= 290):
                        stylists.add(track_id)
                
                # Client-only tracking logic
                if track_id not in stylists:
                    # Update global entries
                    if track_id not in tracked_people:
                        tracked_people.add(track_id)
                        metrics["total_entries"] = len(tracked_people)
                    
                    # Coordinates in 640x360 frame
                    if 100 <= cx <= 380 and 120 <= cy <= 360:
                        s1_detected = True
                        
                    # Check Station 2 ROI (Middle chair)
                    # Disjoint y-range to avoid overlap with waiting bench
                    if 340 <= cx <= 450 and 155 <= cy <= 290:
                        s2_detected = True
                        
                    # Check Waiting Lounge (Background bench)
                    # Mapped to the bench zone (x: 200-380, y: 70-150)
                    if 200 <= cx <= 380 and 70 <= cy <= 150:
                        waiting_detected.add(track_id)
                
                # Draw client bounding box (Luxury gold tone)
                if track_id not in stylists:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 160, 120), 1)
                    cv2.putText(frame, f"Client #{track_id}", (x1, y1 - 8), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 160, 120), 1)
                else:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (180, 140, 160), 1)
                    cv2.putText(frame, "Stylist", (x1, y1 - 8), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 140, 160), 1)
                    
        # --- STATE SMOOTHING & HYSTERESIS FOR STATION 01 ---
        if s1_detected:
            station_streaks[1]["occupied"] += 1
            station_streaks[1]["vacant"] = 0
        else:
            station_streaks[1]["vacant"] += 1
            station_streaks[1]["occupied"] = 0

        if CHAIRS[1]["status"] == "Vacant":
            if station_streaks[1]["occupied"] >= 15:
                CHAIRS[1]["status"] = "Occupied"
                CHAIRS[1]["occupancy_start"] = time.time()
                BARBERS[1]["status"] = "Servicing"
                BARBERS[1]["current_chair"] = 1
                BARBERS[1]["service_start"] = time.time()
        else:
            if station_streaks[1]["vacant"] >= 45:
                CHAIRS[1]["status"] = "Vacant"
                if CHAIRS[1]["occupancy_start"]:
                    duration = time.time() - CHAIRS[1]["occupancy_start"]
                    # Only append significant sessions (> 5s) to avoid loop-flicker anomalies
                    if duration >= 5.0:
                        CHAIRS[1]["durations"].append(duration)
                CHAIRS[1]["occupancy_start"] = None
                BARBERS[1]["status"] = "Idle"
                if BARBERS[1]["service_start"]:
                    s_dur = time.time() - BARBERS[1]["service_start"]
                    if s_dur >= 5.0:
                        BARBERS[1]["total_service_time"] += s_dur
                BARBERS[1]["service_start"] = None

        # --- STATE SMOOTHING & HYSTERESIS FOR STATION 02 ---
        if s2_detected:
            station_streaks[2]["occupied"] += 1
            station_streaks[2]["vacant"] = 0
        else:
            station_streaks[2]["vacant"] += 1
            station_streaks[2]["occupied"] = 0

        if CHAIRS[2]["status"] == "Vacant":
            if station_streaks[2]["occupied"] >= 15:
                CHAIRS[2]["status"] = "Occupied"
                CHAIRS[2]["occupancy_start"] = time.time()
                BARBERS[2]["status"] = "Servicing"
                BARBERS[2]["current_chair"] = 2
                BARBERS[2]["service_start"] = time.time()
        else:
            if station_streaks[2]["vacant"] >= 45:
                CHAIRS[2]["status"] = "Vacant"
                if CHAIRS[2]["occupancy_start"]:
                    duration = time.time() - CHAIRS[2]["occupancy_start"]
                    if duration >= 5.0:
                        CHAIRS[2]["durations"].append(duration)
                CHAIRS[2]["occupancy_start"] = None
                BARBERS[2]["status"] = "Idle"
                if BARBERS[2]["service_start"]:
                    s_dur = time.time() - BARBERS[2]["service_start"]
                    if s_dur >= 5.0:
                        BARBERS[2]["total_service_time"] += s_dur
                BARBERS[2]["service_start"] = None

        # --- WAITING LOUNGE REGISTRY TRACKING ---
        for cid in waiting_detected:
            if cid not in waiting_clients:
                waiting_clients[cid] = time.time()
                waiting_streaks[cid] = {"occupied": 1, "vacant": 0}
            else:
                waiting_streaks[cid]["occupied"] += 1
                waiting_streaks[cid]["vacant"] = 0
                
        # Handle clients leaving the waiting lounge (moving to station or leaving)
        active_waiting_ids = list(waiting_clients.keys())
        for cid in active_waiting_ids:
            if cid not in waiting_detected:
                if cid not in waiting_streaks:
                    waiting_streaks[cid] = {"occupied": 0, "vacant": 0}
                waiting_streaks[cid]["vacant"] += 1
                waiting_streaks[cid]["occupied"] = 0
                
                # If they have been missing from the lounge for 45 frames, finalize wait session
                if waiting_streaks[cid]["vacant"] >= 45:
                    start_time = waiting_clients.pop(cid)
                    duration = time.time() - start_time
                    if duration >= 5.0:
                        wait_durations.append(duration)
                    waiting_streaks.pop(cid, None)

        # Draw ROIs on the frame
        # Station 1
        color1 = (80, 90, 220) if CHAIRS[1]["status"] == "Occupied" else (180, 180, 180)
        cv2.rectangle(frame, (100, 120), (380, 350), color1, 1)
        cv2.putText(frame, "STATION 01", (105, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color1, 1)
        
        # Station 2
        color2 = (80, 90, 220) if CHAIRS[2]["status"] == "Occupied" else (180, 180, 180)
        cv2.rectangle(frame, (340, 155), (450, 290), color2, 1)
        cv2.putText(frame, "STATION 02", (345, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color2, 1)
        
        # Waiting Lounge
        cv2.rectangle(frame, (200, 70), (380, 150), (197, 168, 128), 1)
        cv2.putText(frame, "WAITING LOUNGE", (205, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (197, 168, 128), 1)

        # Scale frame up to 800x600 for sharp look in dashboard
        frame_resized = cv2.resize(frame, (800, 600))
        
        # Encoding image to JPEG
        ret, buffer = cv2.imencode('.jpg', frame_resized)
        if ret:
            with frame_lock:
                latest_frame = buffer.tobytes()
                
        time.sleep(0.04) # ~25 FPS

# --- WEB APP ENDPOINTS ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            with frame_lock:
                if latest_frame is not None:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + latest_frame + b'\r\n')
            time.sleep(0.04)
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/data')
def api_data():
    chairs_data = {}
    for cid, ch in CHAIRS.items():
        avg_time = np.mean(ch["durations"]) if ch["durations"] else 0.0
        current_dur = 0.0
        if ch["status"] == "Occupied" and ch["occupancy_start"]:
            current_dur = time.time() - ch["occupancy_start"]
            
        chairs_data[cid] = {
            "status": ch["status"],
            "current_customer": ch["current_customer"],
            "current_duration": round(current_dur, 1),
            "average_duration": round(avg_time, 1),
            "total_services": len(ch["durations"])
        }
        
    barbers_data = {}
    for bid, b in BARBERS.items():
        current_service_dur = 0.0
        if b["status"] == "Servicing" and b["service_start"]:
            current_service_dur = time.time() - b["service_start"]
            
        barbers_data[bid] = {
            "name": b["name"],
            "status": b["status"],
            "current_chair": b["current_chair"],
            "total_service_time": round(b["total_service_time"] + current_service_dur, 1)
        }

    # Queue data
    waiting_queue = [
        {"client_id": cid, "duration": round(time.time() - start_time, 1)}
        for cid, start_time in waiting_clients.items()
    ]
    avg_wait = np.mean(wait_durations) if wait_durations else 0.0

    return jsonify({
        "total_entries": metrics["total_entries"],
        "active_barbers_count": metrics["active_barbers"],
        "chairs": chairs_data,
        "barbers": barbers_data,
        "waiting": {
            "count": len(waiting_queue),
            "avg_wait_time": round(avg_wait, 1),
            "queue": waiting_queue
        }
    })

if __name__ == '__main__':
    # Start AI processing background thread
    sim_thread = threading.Thread(target=process_video, daemon=True)
    sim_thread.start()
    
    app.run(host='0.0.0.0', port=5002, debug=False)
