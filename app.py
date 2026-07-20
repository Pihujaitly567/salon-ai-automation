import cv2
import numpy as np
import time
import math
import threading
from flask import Flask, render_template, Response, jsonify
from ultralytics import YOLO

app = Flask(__name__)

# --- LOAD YOLO MODEL ---
# Using the lightweight nano model (downloads automatically on first run ~6MB)
model = YOLO("yolov8n.pt")

# --- SERVER START TIME ---
SERVER_START_TIME = time.time()

# --- GLOBAL STREAMS LOCKS & BUFFERS ---
WIDTH, HEIGHT = 768, 432
latest_frame_main = None
latest_frame_spa = None
frame_lock_main = threading.Lock()
frame_lock_spa = threading.Lock()

# --- STATE DATABASE ---

# Room 1: Main Styling Lounge (YOLO Tracking)
ROOM_MAIN = {
    "name": "Main Styling Lounge",
    "total_entries": 0,
    "active_barbers_count": 3,
    "chairs": {
        1: {"roi": (120, 100, 260, 320), "label": "Station 01", "status": "Vacant", "current_customer": None, "occupancy_start": None, "durations": []},
        2: {"roi": (300, 100, 460, 320), "label": "Station 02", "status": "Vacant", "current_customer": None, "occupancy_start": None, "durations": []},
        3: {"roi": (500, 100, 660, 320), "label": "Station 03", "status": "Vacant", "current_customer": None, "occupancy_start": None, "durations": []}
    },
    "barbers": {
        1: {"name": "Alex (Senior Barber)", "status": "Idle", "current_chair": None, "service_start": None, "total_service_time": 0.0},
        2: {"name": "Jordan (Stylist)", "status": "Idle", "current_chair": None, "service_start": None, "total_service_time": 0.0},
        3: {"name": "Taylor (Color Specialist)", "status": "Idle", "current_chair": None, "service_start": None, "total_service_time": 0.0}
    }
}

# Room 2: Spa & Therapy Room (Simulated Tracking)
ROOM_SPA = {
    "name": "Spa & Therapy Room",
    "total_entries": 0,
    "active_barbers_count": 2,
    "chairs": {
        1: {"pos": (250, 230), "roi_box": (195, 150, 305, 330), "label": "Massage Table 01", "status": "Vacant", "current_customer": None, "occupancy_start": None, "durations": []},
        2: {"pos": (500, 230), "roi_box": (445, 150, 555, 330), "label": "Massage Table 02", "status": "Vacant", "current_customer": None, "occupancy_start": None, "durations": []}
    },
    "barbers": {
        1: {"name": "Morgan (Therapist A)", "rest_pos": (250, 100), "pos": [250, 100], "status": "Idle", "current_chair": None, "service_start": None, "total_service_time": 0.0},
        2: {"name": "Casey (Therapist B)", "rest_pos": (500, 100), "pos": [500, 100], "status": "Idle", "current_chair": None, "service_start": None, "total_service_time": 0.0}
    }
}

# Unique trackers
tracked_people_main = set()

# Helper function to check if a point is inside a bounding box
def is_in_roi(point, roi):
    px, py = point
    rx1, ry1, rx2, ry2 = roi
    return rx1 <= px <= rx2 and ry1 <= py <= ry2

# --- BACKGROUND AI PROCESSING LOOP: MAIN LOUNGE (YOLOv8) ---
def process_video_main():
    global latest_frame_main, tracked_people_main
    
    cap = cv2.VideoCapture("people-detection.mp4")
    if not cap.isOpened():
        print("Error: Could not open people-detection.mp4 video file.")
        return
        
    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
            
        frame = cv2.resize(frame, (WIDTH, HEIGHT))
        results = model.track(frame, persist=True, classes=[0], verbose=False)
        
        chairs_occupied_this_frame = {1: False, 2: False, 3: False}
        chairs_customer_this_frame = {1: None, 2: None, 3: None}
        
        if results and len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                track_id = None
                if box.id is not None:
                    track_id = int(box.id[0])
                    
                if track_id is not None:
                    if track_id not in tracked_people_main:
                        tracked_people_main.add(track_id)
                        ROOM_MAIN["total_entries"] = len(tracked_people_main)
                        
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    
                    for cid, chair in ROOM_MAIN["chairs"].items():
                        if is_in_roi((cx, cy), chair["roi"]):
                            chairs_occupied_this_frame[cid] = True
                            chairs_customer_this_frame[cid] = track_id
                            
                    # Draw client bounding box (Luxury gold tone)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 160, 120), 1)
                    cv2.putText(frame, f"Client #{track_id}", (x1, y1 - 8), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 160, 120), 1)
                                
        # Update station occupancy
        for cid, chair in ROOM_MAIN["chairs"].items():
            occupied = chairs_occupied_this_frame[cid]
            cust_id = chairs_customer_this_frame[cid]
            rx1, ry1, rx2, ry2 = chair["roi"]
            
            if occupied:
                cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (100, 110, 200), 2)  # Occupied (Rose Gold color)
                cv2.putText(frame, f"Station 0{cid} (Occupied by Client #{cust_id})", 
                            (rx1, ry1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 110, 200), 1)
            else:
                cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (180, 180, 180), 1)  # Vacant
                cv2.putText(frame, f"Station 0{cid} (Vacant)", 
                            (rx1, ry1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 140, 140), 1)
            
            # Occupancy timers & Barbers updates
            if occupied:
                if chair["status"] == "Vacant":
                    chair["status"] = "Occupied"
                    chair["current_customer"] = cust_id
                    chair["occupancy_start"] = time.time()
                    
                    for bid, b in ROOM_MAIN["barbers"].items():
                        if b["status"] == "Idle":
                            b["status"] = "Servicing"
                            b["current_chair"] = cid
                            b["service_start"] = time.time()
                            break
            else:
                if chair["status"] == "Occupied":
                    chair["status"] = "Vacant"
                    chair["current_customer"] = None
                    if chair["occupancy_start"]:
                        duration = time.time() - chair["occupancy_start"]
                        chair["durations"].append(duration)
                        chair["occupancy_start"] = None
                        
                    for bid, b in ROOM_MAIN["barbers"].items():
                        if b["current_chair"] == cid:
                            b["status"] = "Idle"
                            b["current_chair"] = None
                            if b["service_start"]:
                                s_dur = time.time() - b["service_start"]
                                b["total_service_time"] += s_dur
                                b["service_start"] = None
                            break

        # Draw Live overlay details
        cv2.rectangle(frame, (10, 10), (220, 60), (255, 255, 255), -1)
        cv2.rectangle(frame, (10, 10), (220, 60), (220, 220, 220), 1)
        cv2.putText(frame, f"Total entries: {ROOM_MAIN['total_entries']}", (20, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 60, 60), 1)
        cv2.putText(frame, f"Active Staff: {ROOM_MAIN['active_barbers_count']}", (20, 48), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 60, 60), 1)

        ret, jpeg = cv2.imencode('.jpg', frame)
        if ret:
            with frame_lock_main:
                latest_frame_main = jpeg.tobytes()
                
        time.sleep(0.035)

# --- BACKGROUND PROCESS SIMULATION: SPA ROOM ---
class SpaCustomer:
    def __init__(self, cid):
        self.id = cid
        self.pos = [50, 380]
        self.state = "ENTERING" # ENTERING, WAITING, GOING_TO_TABLE, BEING_SERVICED, LEAVING, EXITED
        self.assigned_table = None
        self.speed = 4

    def move_towards(self, target):
        tx, ty = target
        cx, cy = self.pos
        dx, dy = tx - cx, ty - cy
        dist = math.sqrt(dx**2 + dy**2)
        if dist < self.speed:
            self.pos = [tx, ty]
            return True
        else:
            self.pos[0] += (dx / dist) * self.speed
            self.pos[1] += (dy / dist) * self.speed
            return False

def process_video_spa():
    global latest_frame_spa
    
    # Colors
    COLOR_FLOOR = (245, 245, 240)      # Linen/Sage floor
    COLOR_TABLE = (120, 160, 150)      # Sage green massage tables
    COLOR_DIVIDER = (197, 168, 128)    # Brass frames
    
    spa_customers = []
    spa_customer_id = 200
    last_spawn = time.time()
    
    while True:
        time.sleep(0.05) # ~20 FPS simulation loop
        
        # 1. Update Simulation State
        current_time = time.time()
        active_cust = [c for c in spa_customers if c.state != "EXITED"]
        
        # Spawn Spa Client every 10-18 seconds
        if len(active_cust) < 4 and (current_time - last_spawn) > np.random.uniform(10.0, 18.0):
            new_c = SpaCustomer(spa_customer_id)
            spa_customers.append(new_c)
            ROOM_SPA["total_entries"] += 1
            spa_customer_id += 1
            last_spawn = current_time

        empty_tables = [tid for tid, t in ROOM_SPA["chairs"].items() if t["status"] == "Vacant" and t["current_customer"] is None]

        # Update Clients
        for cust in spa_customers:
            if cust.state == "EXITED":
                continue
                
            if cust.state == "ENTERING":
                # Move to waiting spot (150, 360)
                if cust.move_towards((150, 360)):
                    cust.state = "WAITING"
            elif cust.state == "WAITING":
                if empty_tables:
                    target_table_id = empty_tables.pop(0)
                    ROOM_SPA["chairs"][target_table_id]["status"] = "Claimed"
                    ROOM_SPA["chairs"][target_table_id]["current_customer"] = cust.id
                    cust.assigned_table = target_table_id
                    cust.state = "GOING_TO_TABLE"
            elif cust.state == "GOING_TO_TABLE":
                table_pos = ROOM_SPA["chairs"][cust.assigned_table]["pos"]
                # Walk to table seat
                if cust.move_towards((table_pos[0], table_pos[1] + 20)):
                    cust.state = "BEING_SERVICED"
                    ROOM_SPA["chairs"][cust.assigned_table]["status"] = "Occupied"
                    ROOM_SPA["chairs"][cust.assigned_table]["occupancy_start"] = time.time()
                    
                    # Assign a therapist
                    free_therapists = [bid for bid, b in ROOM_SPA["barbers"].items() if b["status"] == "Idle"]
                    if free_therapists:
                        bid = free_therapists[0]
                        ROOM_SPA["barbers"][bid]["status"] = "Moving"
                        ROOM_SPA["barbers"][bid]["current_chair"] = cust.assigned_table
                        ROOM_SPA["barbers"][bid]["service_start"] = time.time()
            elif cust.state == "BEING_SERVICED":
                table = ROOM_SPA["chairs"][cust.assigned_table]
                elapsed = time.time() - table["occupancy_start"]
                
                servicing_therapist = None
                for bid, b in ROOM_SPA["barbers"].items():
                    if b["current_chair"] == cust.assigned_table and b["status"] == "Servicing":
                        servicing_therapist = b
                        break
                        
                # Treatment completed in 15-22 seconds
                if elapsed > np.random.uniform(15.0, 22.0) and servicing_therapist:
                    table["status"] = "Vacant"
                    table["current_customer"] = None
                    duration = time.time() - table["occupancy_start"]
                    table["durations"].append(duration)
                    table["occupancy_start"] = None
                    
                    for bid, b in ROOM_SPA["barbers"].items():
                        if b["current_chair"] == cust.assigned_table:
                            b["status"] = "Returning"
                            b["current_chair"] = None
                            if b["service_start"]:
                                s_dur = time.time() - b["service_start"]
                                b["total_service_time"] += s_dur
                                b["service_start"] = None
                    
                    cust.state = "LEAVING"
                    cust.assigned_table = None
            elif cust.state == "LEAVING":
                # Exit at bottom left (50, 380)
                if cust.move_towards((50, 380)):
                    cust.state = "EXITED"

        # Update Therapists
        for bid, b in ROOM_SPA["barbers"].items():
            if b["status"] == "Idle":
                b["pos"] = list(b["rest_pos"])
            elif b["status"] == "Moving":
                table_pos = ROOM_SPA["chairs"][b["current_chair"]]["pos"]
                target = (table_pos[0] + 30, table_pos[1] - 5)
                dx, dy = target[0] - b["pos"][0], target[1] - b["pos"][1]
                dist = math.sqrt(dx**2 + dy**2)
                if dist < 5:
                    b["pos"] = list(target)
                    b["status"] = "Servicing"
                else:
                    b["pos"][0] += (dx / dist) * 5
                    b["pos"][1] += (dy / dist) * 5
            elif b["status"] == "Servicing":
                table_pos = ROOM_SPA["chairs"][b["current_chair"]]["pos"]
                wobble_y = math.sin(time.time() * 10) * 1.5
                b["pos"] = [table_pos[0] + 30, table_pos[1] - 5 + wobble_y]
            elif b["status"] == "Returning":
                dx, dy = b["rest_pos"][0] - b["pos"][0], b["rest_pos"][1] - b["pos"][1]
                dist = math.sqrt(dx**2 + dy**2)
                if dist < 5:
                    b["pos"] = list(b["rest_pos"])
                    b["status"] = "Idle"
                else:
                    b["pos"][0] += (dx / dist) * 5
                    b["pos"][1] += (dy / dist) * 5

        # 2. Draw Spa Layout
        frame = np.ones((HEIGHT, WIDTH, 3), dtype=np.uint8)
        frame[:, :, 0] = COLOR_FLOOR[0]
        frame[:, :, 1] = COLOR_FLOOR[1]
        frame[:, :, 2] = COLOR_FLOOR[2]
        
        # Grid/Flooring divider lines
        for i in range(0, WIDTH, 60):
            cv2.line(frame, (i, 0), (i, HEIGHT), (230, 230, 225), 1)

        # Entrance
        cv2.line(frame, (20, 340), (120, 340), COLOR_DIVIDER, 2)
        cv2.putText(frame, "SPA ENTRANCE", (25, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 80, 80), 1)

        # Draw Massage Tables (Regions of Interest)
        for tid, table in ROOM_SPA["chairs"].items():
            tx, ty = table["pos"]
            rx1, ry1, rx2, ry2 = table["roi_box"]
            
            # Draw Table (Sage rounded box)
            cv2.rectangle(frame, (tx - 25, ty - 50), (tx + 25, ty + 50), COLOR_TABLE, -1)
            cv2.rectangle(frame, (tx - 25, ty - 50), (tx + 25, ty + 50), (80, 110, 100), 1)
            
            # Headrest pillow
            cv2.circle(frame, (tx, ty - 38), 10, (230, 230, 220), -1)
            cv2.circle(frame, (tx, ty - 38), 10, (80, 110, 100), 1)

            # Draw AI tracking ROI border
            border_color = (200, 200, 200)
            if table["status"] == "Occupied":
                border_color = (100, 110, 200) # Soft red
            elif table["status"] == "Claimed":
                border_color = COLOR_DIVIDER
                
            cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), border_color, 1)
            cv2.putText(frame, table["label"], (rx1, ry1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1)
            
            if table["status"] == "Occupied" and table["occupancy_start"]:
                dur = time.time() - table["occupancy_start"]
                cv2.putText(frame, f"Active: {dur:.1f}s", (tx - 32, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (40, 40, 40), 1)
            else:
                cv2.putText(frame, "VACANT", (tx - 22, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 80, 80), 1)

        # Draw Spa Clients (Green circular tracking nodes)
        for cust in spa_customers:
            if cust.state == "EXITED":
                continue
            cx, cy = int(cust.pos[0]), int(cust.pos[1])
            
            # Draw tracking bounding box simulation
            cv2.rectangle(frame, (cx - 15, cy - 25), (cx + 15, cy + 15), (100, 150, 80), 1)
            cv2.circle(frame, (cx, cy - 8), 10, (150, 200, 130), -1)
            cv2.circle(frame, (cx, cy - 8), 10, (80, 120, 60), 1)
            cv2.putText(frame, f"Client {cust.id}", (cx - 24, cy - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (80, 120, 60), 1)

        # Draw Spa Therapists (Staff)
        for bid, b in ROOM_SPA["barbers"].items():
            bx, by = int(b["pos"][0]), int(b["pos"][1])
            # Draw tracking bounding box
            cv2.rectangle(frame, (bx - 12, by - 22), (bx + 12, by + 12), (180, 120, 140), 1)
            cv2.circle(frame, (bx, by - 8), 10, (220, 170, 190), -1)
            cv2.circle(frame, (bx, by - 8), 10, (140, 90, 110), 1)
            cv2.putText(frame, f"Therapist {bid}", (bx - 28, by - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (140, 90, 110), 1)

        # Draw local stats details
        cv2.rectangle(frame, (10, 10), (220, 60), (255, 255, 255), -1)
        cv2.rectangle(frame, (10, 10), (220, 60), (220, 220, 220), 1)
        cv2.putText(frame, f"Total entries: {ROOM_SPA['total_entries']}", (20, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 60, 60), 1)
        cv2.putText(frame, f"Active Staff: {ROOM_SPA['active_barbers_count']}", (20, 48), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 60, 60), 1)

        ret, jpeg = cv2.imencode('.jpg', frame)
        if ret:
            with frame_lock_spa:
                latest_frame_spa = jpeg.tobytes()

# --- WEB APP ENDPOINTS ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed/<room>')
def video_feed(room):
    def frame_generator(room_name):
        while True:
            if room_name == "main":
                with frame_lock_main:
                    if latest_frame_main is not None:
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + latest_frame_main + b'\r\n')
            elif room_name == "spa":
                with frame_lock_spa:
                    if latest_frame_spa is not None:
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + latest_frame_spa + b'\r\n')
            time.sleep(0.04)
            
    return Response(frame_generator(room), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/data')
def api_data():
    # Process Room 1 (Main Lounge) Data
    main_chairs = {}
    total_busy_chairs = 0
    total_stations = 0
    
    for cid, ch in ROOM_MAIN["chairs"].items():
        total_stations += 1
        avg_time = np.mean(ch["durations"]) if ch["durations"] else 0.0
        current_dur = 0.0
        if ch["status"] == "Occupied" and ch["occupancy_start"]:
            total_busy_chairs += 1
            current_dur = time.time() - ch["occupancy_start"]
            
        main_chairs[cid] = {
            "status": ch["status"],
            "current_customer": ch["current_customer"],
            "current_duration": round(current_dur, 1),
            "average_duration": round(avg_time, 1),
            "total_services": len(ch["durations"])
        }
        
    main_barbers = {}
    for bid, b in ROOM_MAIN["barbers"].items():
        current_service_dur = 0.0
        if b["status"] == "Servicing" and b["service_start"]:
            current_service_dur = time.time() - b["service_start"]
            
        main_barbers[bid] = {
            "name": b["name"],
            "status": b["status"],
            "current_chair": b["current_chair"],
            "total_service_time": round(b["total_service_time"] + current_service_dur, 1)
        }

    # Process Room 2 (Spa Room) Data
    spa_chairs = {}
    for cid, ch in ROOM_SPA["chairs"].items():
        total_stations += 1
        avg_time = np.mean(ch["durations"]) if ch["durations"] else 0.0
        current_dur = 0.0
        if ch["status"] == "Occupied" and ch["occupancy_start"]:
            total_busy_chairs += 1
            current_dur = time.time() - ch["occupancy_start"]
            
        spa_chairs[cid] = {
            "status": ch["status"],
            "current_customer": ch["current_customer"],
            "current_duration": round(current_dur, 1),
            "average_duration": round(avg_time, 1),
            "total_services": len(ch["durations"])
        }
        
    spa_barbers = {}
    for bid, b in ROOM_SPA["barbers"].items():
        current_service_dur = 0.0
        if b["status"] == "Servicing" and b["service_start"]:
            current_service_dur = time.time() - b["service_start"]
            
        spa_barbers[bid] = {
            "name": b["name"],
            "status": b["status"],
            "current_chair": b["current_chair"],
            "total_service_time": round(b["total_service_time"] + current_service_dur, 1)
        }

    # Global aggregation
    global_entries = ROOM_MAIN["total_entries"] + ROOM_SPA["total_entries"]
    global_active_staff = ROOM_MAIN["active_barbers_count"] + ROOM_SPA["active_barbers_count"]
    occupancy_rate = round((total_busy_chairs / total_stations) * 100, 1) if total_stations > 0 else 0.0
    uptime = time.time() - SERVER_START_TIME

    return jsonify({
        "global": {
            "total_entries": global_entries,
            "active_staff": global_active_staff,
            "occupancy_rate": occupancy_rate,
            "total_busy_stations": total_busy_chairs,
            "total_stations": total_stations,
            "uptime": round(uptime, 1)
        },
        "rooms": {
            "main": {
                "name": ROOM_MAIN["name"],
                "total_entries": ROOM_MAIN["total_entries"],
                "active_barbers_count": ROOM_MAIN["active_barbers_count"],
                "chairs": main_chairs,
                "barbers": main_barbers
            },
            "spa": {
                "name": ROOM_SPA["name"],
                "total_entries": ROOM_SPA["total_entries"],
                "active_barbers_count": ROOM_SPA["active_barbers_count"],
                "chairs": spa_chairs,
                "barbers": spa_barbers
            }
        }
    })

if __name__ == '__main__':
    # Start Main Lounge YOLO background thread
    main_thread = threading.Thread(target=process_video_main, daemon=True)
    main_thread.start()
    
    # Start Spa Room background thread
    spa_thread = threading.Thread(target=process_video_spa, daemon=True)
    spa_thread.start()
    
    app.run(host='0.0.0.0', port=5001, debug=False)
