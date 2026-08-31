import cv2
import numpy as np
import time
import math
import json
import os
from ultralytics import YOLO
import mediapipe as mp
import config

def load_zones_database():
    if os.path.exists(config.ZONES_CONFIG_PATH):
        try:
            with open(config.ZONES_CONFIG_PATH, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load zones config ({e}), using defaults.")
    return {}

def save_zones_database(data):
    with open(config.ZONES_CONFIG_PATH, "w") as f:
        json.dump(data, f, indent=2)

def classify_posture(frame, bbox, pose_model):
    """
    Determines if a detected person is sitting (client) or standing (stylist)
    using Google MediaPipe Pose landmarks on cropped person boxes.
    """
    x1, y1, x2, y2 = bbox
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    aspect_ratio = width / float(height)

    # Fast aspect ratio check
    if aspect_ratio < 0.48:
        return "STANDING"

    # Crop the person from the frame
    pad = 5
    px1 = max(0, x1 - pad)
    py1 = max(0, y1 - pad)
    px2 = min(frame.shape[1], x2 + pad)
    py2 = min(frame.shape[0], y2 + pad)
    crop = frame[py1:py2, px1:px2]
    if crop.size == 0:
        return "SITTING" if aspect_ratio > config.SITTING_ASPECT_RATIO_FALLBACK else "STANDING"

    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    results = pose_model.process(crop_rgb)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        
        # Landmarks: Shoulders (11, 12), Hips (23, 24), Knees (25, 26), Ankles (27, 28)
        l_hip = landmarks[23]
        r_hip = landmarks[24]
        l_knee = landmarks[25]
        r_knee = landmarks[26]
        l_ankle = landmarks[27]
        r_ankle = landmarks[28]
        l_shoulder = landmarks[11]
        r_shoulder = landmarks[12]

        l_visible = l_hip.visibility > 0.3 and l_knee.visibility > 0.3 and l_ankle.visibility > 0.3
        r_visible = r_hip.visibility > 0.3 and r_knee.visibility > 0.3 and r_ankle.visibility > 0.3

        def get_angle(pt1, pt2, pt3):
            # pt2 is the vertex of the angle
            a = np.array([pt1.x, pt1.y])
            b = np.array([pt2.x, pt2.y])
            c = np.array([pt3.x, pt3.y])
            ba = a - b
            bc = c - b
            cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
            angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
            return np.degrees(angle)

        angles = []
        if l_visible:
            l_hip_angle = get_angle(l_shoulder, l_hip, l_knee)
            l_knee_angle = get_angle(l_hip, l_knee, l_ankle)
            angles.append((l_hip_angle, l_knee_angle))
        if r_visible:
            r_hip_angle = get_angle(r_shoulder, r_hip, r_knee)
            r_knee_angle = get_angle(r_hip, r_knee, r_ankle)
            angles.append((r_hip_angle, r_knee_angle))

        if angles:
            avg_hip = np.mean([a[0] for a in angles])
            avg_knee = np.mean([a[1] for a in angles])
            # Sitting angle is typically < 140 degrees
            if avg_hip < 140.0 or avg_knee < 140.0:
                return "SITTING"
            else:
                return "STANDING"

    return "SITTING" if aspect_ratio > config.SITTING_ASPECT_RATIO_FALLBACK else "STANDING"


class SalonTracker:
    def __init__(self):
        print("Initializing YOLOv8-Pose model...")
        self.model = YOLO(config.MODEL_PATH)
        print("Model initialized successfully.")
        
        # Initialize Google MediaPipe Pose
        print("Initializing MediaPipe Pose...")
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        print("MediaPipe Pose initialized successfully.")

        self.zones_db = load_zones_database()
        self.current_video_file = config.DEFAULT_VIDEO

        self.stations_state = {}
        self.waiting_slots = {}
        self.wait_durations = []
        self.tracked_people = set()
        self.next_slot_id = 1
        self.total_entries = 0

        self.staff_profiles = {
            1: {"name": "Alex (Senior Barber)", "status": "Idle", "current_chair": None, "service_start": None, "total_service_time": 0.0},
            2: {"name": "Jordan (Stylist)", "status": "Idle", "current_chair": None, "service_start": None, "total_service_time": 0.0},
            3: {"name": "Taylor (Color Specialist)", "status": "Idle", "current_chair": None, "service_start": None, "total_service_time": 0.0}
        }

        self.show_skeletons = True
        self.sync_stations()

        # Dynamic Barber Counting State
        self.active_barbers_count = 0
        self.barber_count_history = []
        self.session_start = time.time()
        self.stylist_track_ids = set()

    def get_active_zones(self):
        video_name = os.path.basename(self.current_video_file)
        import copy
        zones = copy.deepcopy(self.zones_db.get(video_name, self.zones_db.get("default", {})))
        
        if video_name == "salon_video_1.mp4":
            zones["station_1"] = {
                "name": "Station 01 (Main Chair)",
                "remarks": "Alex - Haircutting & Beard Styling",
                "type": "service",
                "x1": 100,
                "x2": 380,
                "y1": 180,
                "y2": 360
            }
            zones["station_2"] = {
                "name": "Station 02 (Styling Chair)",
                "remarks": "Jordan - Blowdry & Coloring",
                "type": "service",
                "x1": 334,
                "x2": 604,
                "y1": 155,
                "y2": 302
            }
            zones["waiting"] = {
                "name": "Waiting Lounge Bench",
                "remarks": "3-seater guest couch",
                "type": "waiting",
                "x1": 180,
                "x2": 420,
                "y1": 65,
                "y2": 185
            }
        return zones

    def sync_stations(self):
        active_zones = self.get_active_zones()
        current_service_ids = set()

        for zid, info in active_zones.items():
            ztype = info.get("type", "service" if "station" in zid or "chair" in zid else "waiting")
            if ztype == "service":
                current_service_ids.add(zid)
                if zid not in self.stations_state:
                    self.stations_state[zid] = {
                        "id": zid,
                        "name": info.get("name", zid.replace("_", " ").title()),
                        "remarks": info.get("remarks", ""),
                        "status": "Vacant",
                        "occupancy_start": None,
                        "durations": [],
                        "streak_occupied": 0,
                        "streak_vacant": 0
                    }
                else:
                    self.stations_state[zid]["name"] = info.get("name", self.stations_state[zid]["name"])
                    self.stations_state[zid]["remarks"] = info.get("remarks", "")

        for zid in list(self.stations_state.keys()):
            if zid not in current_service_ids:
                del self.stations_state[zid]

    def reset_state(self):
        self.sync_stations()
        for st in self.stations_state.values():
            st["status"] = "Vacant"
            st["occupancy_start"] = None
            st["durations"] = []
            st["streak_occupied"] = 0
            st["streak_vacant"] = 0

        for staff in self.staff_profiles.values():
            staff["status"] = "Idle"
            staff["current_chair"] = None
            staff["service_start"] = None
            staff["total_service_time"] = 0.0

        self.total_entries = 0
        self.tracked_people.clear()
        self.waiting_slots.clear()
        self.wait_durations.clear()
        self.next_slot_id = 1
        self.session_start = time.time()
        self.stylist_track_ids.clear()
        self.barber_count_history.clear()
        self.active_barbers_count = 0

    def process_frame(self, frame):
        self.sync_stations()
        active_zones = self.get_active_zones()

        results = self.model.track(frame, persist=True, verbose=False)
        detected_service_zones = set()
        waiting_candidates = []
        detected_barbers = 0
        detections = []

        if results and len(results) > 0:
            res = results[0]
            boxes = res.boxes
            keypoints_data = res.keypoints

            if boxes is not None and len(boxes) > 0:
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    track_id = int(box.id[0]) if box.id is not None else None
                    kpts = keypoints_data[i].data[0] if keypoints_data is not None and len(keypoints_data) > i else None

                    posture = classify_posture(frame, (x1, y1, x2, y2), self.pose)
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)

                    detections.append({
                        "box_idx": i,
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "cx": cx, "cy": cy,
                        "track_id": track_id,
                        "kpts": kpts,
                        "posture": posture,
                        "assigned_role": None
                    })

        # Step 1: Pre-assign roles for known locked stylists
        for det in detections:
            tid = det["track_id"]
            if tid is not None and tid in self.stylist_track_ids:
                det["assigned_role"] = "Stylist"
                det["posture"] = "STANDING"

        # Resolve Stylist vs Client inside service zones using relative height ordering
        for zid, zinfo in active_zones.items():
            ztype = zinfo.get("type", "service" if "station" in zid or "chair" in zid else "waiting")
            if ztype != "service":
                continue

            zx1, zy1 = zinfo.get("x1", 0), zinfo.get("y1", 0)
            zx2, zy2 = zinfo.get("x2", 0), zinfo.get("y2", 0)

            # Find all detections inside this zone
            zone_dets = []
            for det in detections:
                cx, cy = det["cx"], det["cy"]
                x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
                overlap_x = max(0, min(x2, zx2) - max(x1, zx1))
                overlap_y = max(0, min(y2, zy2) - max(y1, zy1))
                overlap_area = overlap_x * overlap_y
                box_area = max(1, (x2 - x1) * (y2 - y1))
                in_zone = (zx1 <= cx <= zx2 and zy1 <= cy <= zy2) or (overlap_area / box_area > 0.25)

                if in_zone:
                    zone_dets.append(det)

            if len(zone_dets) >= 2:
                # Find if one of them is already a locked stylist
                stylist_det = None
                client_dets = []
                for d in zone_dets:
                    if d["assigned_role"] == "Stylist":
                        stylist_det = d
                    else:
                        client_dets.append(d)

                if stylist_det is None:
                    # Sort by y1 (highest head first)
                    zone_dets.sort(key=lambda d: d["y1"])
                    stylist_det = zone_dets[0]
                    client_dets = zone_dets[1:]

                # Lock the stylist role
                stylist_det["assigned_role"] = "Stylist"
                stylist_det["posture"] = "STANDING"
                if stylist_det["track_id"] is not None:
                    self.stylist_track_ids.add(stylist_det["track_id"])

                # Mark other people in this zone as clients
                for c in client_dets:
                    c["assigned_role"] = "Client"
                    c["posture"] = "SITTING"
                detected_service_zones.add(zid)

            elif len(zone_dets) == 1:
                det = zone_dets[0]
                if det["assigned_role"] == "Stylist":
                    # Stylist is alone in the service zone, station remains vacant
                    pass
                else:
                    if det["posture"] == "SITTING":
                        det["assigned_role"] = "Client"
                        detected_service_zones.add(zid)
                    else:
                        det["assigned_role"] = "Stylist"
                        if det["track_id"] is not None:
                            self.stylist_track_ids.add(det["track_id"])

        # Process final roles and draw bounding boxes
        for det in detections:
            x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
            track_id = det["track_id"]
            posture = det["posture"]
            cx, cy = det["cx"], det["cy"]
            kpts = det["kpts"]
            role = det["assigned_role"]

            if role is None:
                # Check if inside a waiting zone
                is_waiting = False
                for zid, zinfo in active_zones.items():
                    ztype = zinfo.get("type", "service" if "station" in zid or "chair" in zid else "waiting")
                    if ztype == "waiting":
                        zx1, zy1 = zinfo.get("x1", 0), zinfo.get("y1", 0)
                        zx2, zy2 = zinfo.get("x2", 0), zinfo.get("y2", 0)

                        overlap_x = max(0, min(x2, zx2) - max(x1, zx1))
                        overlap_y = max(0, min(y2, zy2) - max(y1, zy1))
                        overlap_area = overlap_x * overlap_y
                        box_area = max(1, (x2 - x1) * (y2 - y1))
                        in_zone = (zx1 <= cx <= zx2 and zy1 <= cy <= zy2) or (overlap_area / box_area > 0.25)

                        if in_zone:
                            det["posture"] = "SITTING"
                            waiting_candidates.append({"cx": cx, "cy": cy, "track_id": track_id})
                            is_waiting = True
                            break
                if is_waiting:
                    role = "Client"
                else:
                    role = "Client" if posture == "SITTING" else "Stylist"

            if role == "Stylist":
                detected_barbers += 1

            if track_id is not None and track_id not in self.tracked_people:
                if role == "Client":
                    self.tracked_people.add(track_id)
                    self.total_entries = len(self.tracked_people)

            # Draw bounding box & posture label
            badge_color = (200, 160, 120) if role == "Client" else (180, 140, 160)
            role_text = f"Client #{track_id} [Sitting]" if (role == "Client" and track_id) else (
                "Client [Sitting]" if role == "Client" else "Stylist [Standing]"
            )
            cv2.rectangle(frame, (x1, y1), (x2, y2), badge_color, 1)
            cv2.putText(frame, role_text, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, badge_color, 1)

            # Draw keypoint dots
            if self.show_skeletons and kpts is not None:
                kpts_arr = kpts.cpu().numpy() if hasattr(kpts, 'cpu') else np.array(kpts)
                for pt in kpts_arr:
                    px, py, conf = int(pt[0]), int(pt[1]), pt[2]
                    if conf > 0.4:
                        cv2.circle(frame, (px, py), 3, (100, 220, 120), -1)

        # Smooth the barber count over a rolling window to prevent quick fluctuations
        self.barber_count_history.append(detected_barbers)
        if len(self.barber_count_history) > 30:
            self.barber_count_history.pop(0)
        self.active_barbers_count = min(max(1, len(self.stylist_track_ids)), len(self.stations_state))

        # Update styling station occupancy states
        staff_keys = list(self.staff_profiles.keys())
        for idx, (zid, st) in enumerate(self.stations_state.items()):
            staff_id = staff_keys[idx % len(staff_keys)]
            is_detected = (zid in detected_service_zones)

            if is_detected:
                st["streak_occupied"] += 1
                st["streak_vacant"] = 0
            else:
                st["streak_vacant"] += 1
                st["streak_occupied"] = 0

            if st["status"] == "Vacant":
                if st["streak_occupied"] >= config.STATION_OCCUPIED_STREAK:
                    st["status"] = "Occupied"
                    st["occupancy_start"] = time.time()
                    self.staff_profiles[staff_id]["status"] = "Servicing"
                    self.staff_profiles[staff_id]["current_chair"] = zid
                    self.staff_profiles[staff_id]["service_start"] = time.time()
            else:
                if st["streak_vacant"] >= config.STATION_VACANT_STREAK:
                    st["status"] = "Vacant"
                    if st["occupancy_start"]:
                        duration = time.time() - st["occupancy_start"]
                        if duration >= 4.0:
                            st["durations"].append(duration)
                    st["occupancy_start"] = None
                    self.staff_profiles[staff_id]["status"] = "Idle"
                    self.staff_profiles[staff_id]["current_chair"] = None
                    if self.staff_profiles[staff_id]["service_start"]:
                        s_dur = time.time() - self.staff_profiles[staff_id]["service_start"]
                        if s_dur >= 4.0:
                            self.staff_profiles[staff_id]["total_service_time"] += s_dur
                    self.staff_profiles[staff_id]["service_start"] = None

        # Update persistent spatial slots for waiting lounge with occlusion resistance
        used_candidates = set()

        # Step 1: Match existing slots to closest candidates (prefer track_id match, then spatial distance)
        for slot_id, slot in list(self.waiting_slots.items()):
            best_cand_idx = None
            best_dist = config.WAITING_SLOT_MAX_DISTANCE

            for idx, cand in enumerate(waiting_candidates):
                if idx in used_candidates:
                    continue
                if cand["track_id"] is not None and slot.get("track_id") == cand["track_id"]:
                    best_cand_idx = idx
                    break
                d = math.hypot(cand["cx"] - slot["cx"], cand["cy"] - slot["cy"])
                if d < best_dist:
                    best_dist = d
                    best_cand_idx = idx

            if best_cand_idx is not None:
                cand = waiting_candidates[best_cand_idx]
                used_candidates.add(best_cand_idx)
                slot["cx"] = int(0.7 * slot["cx"] + 0.3 * cand["cx"])
                slot["cy"] = int(0.7 * slot["cy"] + 0.3 * cand["cy"])
                slot["missed"] = 0
                slot["hits"] += 1
                if cand["track_id"] is not None:
                    if slot.get("track_id") != cand["track_id"]:
                        slot["start_time"] = time.time()
                    slot["track_id"] = cand["track_id"]
                    slot["display_id"] = cand["track_id"]
            else:
                # Check if this slot is currently occluded by a standing stylist
                is_occluded = False
                for det in detections:
                    if det["assigned_role"] == "Stylist" or det["posture"] == "STANDING":
                        dx = abs(det["cx"] - slot["cx"])
                        # If a stylist stands close horizontally and covers the slot's vertical level
                        if dx < 60 and det["y1"] <= slot["cy"] + 20 and det["y2"] >= slot["cy"] - 20:
                            is_occluded = True
                            break
                            
                if is_occluded:
                    slot["missed"] = 0  # Reset missed frames to hold the slot active during occlusion
                else:
                    # Slot temporarily occluded by passing person: increment missed count
                    slot["missed"] += 1
                    if slot["missed"] >= config.WAITING_SLOT_DROP_FRAMES:
                        dur = time.time() - slot["start_time"]
                        if dur >= 4.0:
                            self.wait_durations.append(dur)
                        del self.waiting_slots[slot_id]

        # Step 2: Initialize new slots for unmatched sitting candidates
        for idx, cand in enumerate(waiting_candidates):
            if idx not in used_candidates:
                too_close = any(
                    math.hypot(cand["cx"] - s["cx"], cand["cy"] - s["cy"]) < config.WAITING_SLOT_MIN_SEPARATION
                    for s in self.waiting_slots.values()
                )
                if not too_close:
                    sid = self.next_slot_id
                    self.next_slot_id += 1
                    self.waiting_slots[sid] = {
                        "start_time": time.time(),
                        "cx": cand["cx"],
                        "cy": cand["cy"],
                        "missed": 0,
                        "hits": 1,
                        "track_id": cand["track_id"],
                        "display_id": cand["track_id"] if cand["track_id"] is not None else sid
                    }

        # Draw calibrated zone overlays
        for zid, zinfo in active_zones.items():
            zx1, zy1 = zinfo.get("x1", 0), zinfo.get("y1", 0)
            zx2, zy2 = zinfo.get("x2", 0), zinfo.get("y2", 0)
            zname = zinfo.get("name", zid)
            ztype = zinfo.get("type", "service" if "station" in zid or "chair" in zid else "waiting")

            if ztype == "service":
                st = self.stations_state.get(zid, {})
                c = (80, 90, 220) if st.get("status") == "Occupied" else (100, 180, 100)
            else:
                c = (197, 168, 128)

            cv2.rectangle(frame, (zx1, zy1), (zx2, zy2), c, 1)
            cv2.putText(frame, zname, (zx1 + 5, zy1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.42, c, 1)

        return frame

    def get_api_payload(self):
        self.sync_stations()

        stations_data = {}
        for zid, st in self.stations_state.items():
            avg_time = np.mean(st["durations"]) if st["durations"] else 0.0
            current_dur = (time.time() - st["occupancy_start"]) if (st["status"] == "Occupied" and st["occupancy_start"]) else 0.0
            total_occ = sum(st["durations"]) + current_dur

            stations_data[zid] = {
                "id": zid,
                "name": st["name"],
                "remarks": st.get("remarks", ""),
                "status": st["status"],
                "current_duration": round(current_dur, 1),
                "total_occupancy_time": round(total_occ, 1),
                "average_duration": round(avg_time, 1),
                "total_services": len(st["durations"])
            }

        barbers_data = {}
        for bid, b in self.staff_profiles.items():
            current_service_dur = (time.time() - b["service_start"]) if (b["status"] == "Servicing" and b["service_start"]) else 0.0
            barbers_data[bid] = {
                "name": b["name"],
                "status": b["status"],
                "current_chair": b["current_chair"],
                "total_service_time": round(b["total_service_time"] + current_service_dur, 1)
            }

        waiting_queue = [
            {
                "client_id": s.get("display_id", slot_id),
                "duration": round(time.time() - s["start_time"], 1)
            }
            for slot_id, s in sorted(self.waiting_slots.items())
            if s["hits"] >= 3
        ]
        all_waits = list(self.wait_durations)
        for item in waiting_queue:
            all_waits.append(item["duration"])
        avg_wait = np.mean(all_waits) if all_waits else 0.0

        # Load Factor
        total_chairs = len(self.stations_state)
        occupied_chairs = sum(1 for st in self.stations_state.values() if st["status"] == "Occupied")
        load_factor = round((occupied_chairs / max(1, total_chairs)) * 100.0, 1)

        # Turnaround Rate (Completed clients per hour)
        total_completed = sum(len(st["durations"]) for st in self.stations_state.values())
        elapsed_hours = max(60.0, time.time() - self.session_start) / 3600.0
        turnaround_rate = round(total_completed / elapsed_hours, 1)

        available_videos = [f for f in os.listdir(config.UPLOAD_FOLDER) if f.endswith(".mp4")]

        return {
            "total_entries": self.total_entries,
            "active_barbers_count": self.active_barbers_count,
            "load_factor": f"{load_factor}%",
            "turnaround_rate": f"{turnaround_rate}/hr",
            "stations": stations_data,
            "chairs": stations_data,
            "barbers": barbers_data,
            "current_video": os.path.basename(self.current_video_file),
            "available_videos": available_videos,
            "waiting": {
                "count": len(waiting_queue),
                "avg_wait_time": round(avg_wait, 1),
                "queue": waiting_queue
            }
        }
