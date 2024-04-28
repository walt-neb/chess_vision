import cv2
import os
import datetime

# Create a directory to store recordings and snapshots
output_directory = "recordings"
os.makedirs(output_directory, exist_ok=True)

# Set up video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video stream from webcam.")
    exit()

# Define the codec and create VideoWriter object for recording video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 files
recording = False
out = None

try:
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read from webcam.")
            break

        # Display the resulting frame
        cv2.imshow('Webcam', frame)

        # Handle keyboard inputs
        key = cv2.waitKey(1)

        # Spacebar to start/stop recording
        if key & 0xFF == 32:  # Spacebar key code
            if recording:
                # Stop recording
                out.release()
                print(f"Stopped recording. Video saved as '{output_filename}'")
                recording = False
            else:
                # Start recording
                current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                output_filename = os.path.join(output_directory, f"video_{current_time}.mp4")
                out = cv2.VideoWriter(output_filename, fourcc, 20.0, (640, 480))
                print(f"Started recording to '{output_filename}'...")
                recording = True

        # 'p' to take a snapshot
        if key & 0xFF == ord('p'):
            snapshot_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            snapshot_filename = os.path.join(output_directory, f"snapshot_{snapshot_time}.jpg")
            cv2.imwrite(snapshot_filename, frame)
            print(f"Snapshot saved as '{snapshot_filename}'")

        # Write the frame to file if recording
        if recording:
            out.write(frame)

        # Quit the program when 'q' key is pressed
        if key & 0xFF == ord('q'):
            if recording:
                # If we were recording, stop and save the file
                out.release()
                print(f"Stopped recording. Video saved as '{output_filename}'")
            break
finally:
    # When everything is done, release the capture
    cap.release()
    cv2.destroyAllWindows()
