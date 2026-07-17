# Salon Intelligence Hub (YOLOv8 & ByteTrack PoC)

This repository contains a **real-time AI Video Analytics** system for the **Salon AI Automation** proof of concept. 

Unlike a simple mock simulation, this version runs a real **YOLOv8 Object Detection** model and a **ByteTrack** tracking algorithm on a sample video to count entries, monitor chair occupancy, and measure stylist service times.

---

## 🛠️ Tech Stack & AI Architecture

1. **AI Model**: **YOLOv8** (`yolov8n.pt` nano model) runs real-time person detection.
2. **Object Tracking**: **ByteTrack** (via Ultralytics YOLO `model.track()`) assigns persistent unique tracking IDs to each person as they walk.
3. **Region of Interest (ROI)**: **OpenCV** defines static coordinates in the video coordinate system corresponding to the styling chairs:
   - **Station 01**: Left Zone
   - **Station 02**: Middle Zone
   - **Station 03**: Right Zone
4. **Backend Server**: **Flask** runs the background thread for frame-by-frame inference and serves a live MJPEG stream + a REST API (`/api/data`) for stats.
5. **Frontend**: A **luxury salon themed dashboard** styled with custom cream palettes, soft rose-gold accents, and premium serif typography.

---

## 🚀 How to Run the App

1. **Activate the environment**:
   ```bash
   source venv/bin/activate
   ```

2. **Start the Flask server**:
   ```bash
   python3 app.py
   ```

3. **Open the Dashboard**:
   Open your browser and navigate to:
   👉 **`http://localhost:5001`**

---

## 📋 Mapping to the 6 Requirements

The app solves all 6 supervisor requirements using actual computer vision:

1. **Count how many people entered the salon**:
   - *Implementation*: YOLOv8 tracks unique IDs. Every time a new `track_id` is detected on screen, the system increments the **Total Entries** counter.
2. **Count barbers**:
   - *Implementation*: Displays the active staff list (Alex, Jordan, and Taylor) tracking their dynamic statuses (Idle or Servicing).
3. **Assign each chair a unique number**:
   - *Implementation*: Static bounding box coordinates (ROIs) are labeled as **Station 01**, **Station 02**, and **Station 03** in the video feed.
4. **Find chair occupancy time**:
   - *Implementation*: When a client's tracked bounding box center falls inside a Station ROI, the chair status becomes `Occupied` and a session timer starts.
5. **Track each chair's average time**:
   - *Implementation*: When the person leaves the ROI, the session duration is logged and added to the running average displayed on the card.
6. **Track each barber's service time**:
   - *Implementation*: When a station becomes occupied, the assigned barber's status changes to `Servicing`, and their cumulative service timer starts ticking.

---

## 📁 Repository Structure

* `app.py`: Backend containing Flask server, background thread for YOLOv8/ByteTrack video processing, and `/api/data` JSON API.
* `templates/index.html`: Responsive luxury spa-themed dashboard with live AJAX data poller.
* `people-detection.mp4`: Sample video downloaded from Intel IoT-Devkit for pedestrian tracking tests.
* `yolov8n.pt`: Pre-trained YOLOv8 weights (auto-downloaded, ~6MB).
* `venv/`: Virtual environment containing `flask`, `opencv-python`, `numpy`, and `ultralytics`.
