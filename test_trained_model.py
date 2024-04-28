#filename: test_trained_model.py

from ultralytics import YOLO
import cv2
import os

# Load the trained model
model = YOLO('/home/walt/src/yolo/train12_best.pt')

# Directory containing images
image_dir = '/home/walt/src/yolo/datasets/ultralytics/chess_yolov8/test/images'

# Loop through all images in the directory
for image_name in os.listdir(image_dir):
    if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):  # Check for image file extensions
        image_path = os.path.join(image_dir, image_name)
        image = cv2.imread(image_path)

        # Perform inference
        results = model.predict(image)
        # If results is mistakenly a list, try getting the first item
        first_result = results[0]

        # Annotate the image with the detections
        annotated_frame = first_result.plot()

        # Display the annotated image
        cv2.imshow('YOLOv8 Detection', annotated_frame) #results.imgs[0])

        # Wait for the space bar to be pressed to move to the next image
        while True:
            key = cv2.waitKey(1)
            if key == 32:  # Space bar key code
                break
            elif key == 27:  # ESC key to exit
                cv2.destroyAllWindows()
                exit()

cv2.destroyAllWindows()
