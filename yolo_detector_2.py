from ultralytics import YOLO
import cv2
import sys

# Load the YOLOv8 model
model = YOLO('train22_best.pt')  # No stream argument
#model = YOLO('train12_best.pt')  # No stream argument

# Determine if we're using a webcam or video file
source = 0 if len(sys.argv) < 2 or sys.argv[1] == '0' else sys.argv[1]
output_name = 'webcam_annotated_22.mp4' if source == 0 else source.replace('.mp4', '_annotated.mp4')

# Process video or webcam
results = model(source)  # Results objects for video files are automatically streamed

# Set up video writer if not using webcam
if source != 0:
    video_capture = cv2.VideoCapture(source)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_name, fourcc, video_capture.get(cv2.CAP_PROP_FPS),
                                   (int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                    int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))))

# Process results
for result in results:
    # Annotate the image with the detections using the plot method
    annotated_frame = result.plot()  # You may need to specify parameters for plot()

    # Ensure that plot() modifies the image in place or returns the annotated image
    # If plot() returns the annotated image, use that. If it modifies in place, use result.imgs[0]

    # Display or write frame
    if source == 0:  # Webcam
        cv2.imshow('YOLOv8 Detection', annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:  # Video file
        video_writer.write(annotated_frame)

# Release everything when done
if source != 0:
    video_capture.release()
    video_writer.release()
cv2.destroyAllWindows()

