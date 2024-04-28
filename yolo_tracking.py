from collections import defaultdict
import sys
import os
import cv2
import numpy as np
import time

from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8n_chess_300.pt')

if len(sys.argv) != 2:
    print("Usage: python yolo_tracker.py <video_file.mp4>")
    sys.exit(1)

# Open the video file
file_path = sys.argv[1]
filename, file_ext = os.path.splitext(sys.argv[1])

cap = cv2.VideoCapture(sys.argv[1])

# Store the track history
track_history = defaultdict(lambda: [])

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        #results = model.predict(frame)  # Run YOLOv8 detection
        results = model.predict(frame)
        # Render the detections
        results.render()  # This line modifies results.imgs in place with the drawn detections
        # Access the annotated image
        annotated_frame = results.imgs[0]

        # Get the boxes and track IDs
        boxes = results[0].boxes.xywh.cpu()
        if results[0].boxes.id != None:
            track_ids = results[0].boxes.id.int().cpu().tolist()

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Plot the tracks
            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                track = track_history[track_id]
                track.append((float(x), float(y)))  # x, y center point
                if len(track) > 30:  # retain 90 tracks for 90 frames
                    track.pop(0)

                # Draw the tracking lines
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(annotated_frame, [points], isClosed=False, color=(50, 230, 50), thickness=2)

            # Display the annotated frame
            cv2.imshow("YOLOv8 Tracking", annotated_frame)
            time.sleep(.1)
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()