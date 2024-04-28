import pyrealsense2 as rs
import cv2
import numpy as np

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# This example assumes 640x480 resolution.
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'mp4v' is the codec for .mp4 files
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (640, 480))

try:
    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # Write the frame into the file 'output.mp4'
        out.write(color_image)

        # Display the resulting frame
        cv2.imshow('RealSense', color_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # Stop streaming
    pipeline.stop()
    # Release everything if job is finished
    out.release()
    cv2.destroyAllWindows()
