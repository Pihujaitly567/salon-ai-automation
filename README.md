# salon-ai-automation

tracks people in a salon using a camera feed and shows some stats on a dashboard.

## what it does
- counts how many people enter
- checks if chairs are occupied (and for how long)
- tracks barber work times
- has a spa room view too (privacy mode so no raw video stream, just a floor plan map)

## tech stack
- **yolov8 & bytetrack** - for detecting and tracking people
- **opencv** - for video handling and chair boundaries (ROIs)
- **python (flask)** - backend server & API
- **html/css/js** - simple luxury-styled dashboard UI

## how to run
1. activate the env:
   `source venv/bin/activate`
2. start the server:
   `python3 app.py`
3. open: `http://localhost:5001`
