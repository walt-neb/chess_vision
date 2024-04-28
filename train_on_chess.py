#filename: train_on_chess.py

from ultralytics import YOLO

# Load a pre-trained model
model = YOLO('/home/walt/src/yolo/yolov8n.pt')  # Starting with a pre-trained base can help

# Prepare the dataset
#train_data = Datasets.load('/home/walt/src/datasets/ultralytics/chess_yolov8/test/images', img_size=640, batch_size=16, augment=True)
#val_data = Datasets.load('/home/walt/src/datasets/ultralytics/chess_yolov8/valid/images', img_size=640, batch_size=16)

# Configure training parameters
model.train(
    data='/home/walt/src/yolo/datasets/sp2.v3i.yolov8/data.yaml',
    epochs=300,           # Number of training epochs
    batch=16,       # Training batch size
#    img_size=640,        # Image size (ensure this matches the model's expected input size)
#    cache_images=True    # Cache images for faster training
)

# Start training
#model.fit()