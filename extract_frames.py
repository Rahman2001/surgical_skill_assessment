import cv2
import os

# Specify the path to the video file and the folder to save the frames

video_path = "video_path"  # Replace with the actual path to your video file

output_folder = "your_output_folder_to_keep_frames_of_the_video"  # Replace with the desired output folder path

# Create the output folder if it does not exist

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Create a VideoCapture object to read the video file
cap = cv2.VideoCapture(video_path)
# Frame rate to 30 fps
cap.set(cv2.CAP_PROP_FPS, 10)
# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
# Extract frames and save them in the output folder
frame_number = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Save the frame as an image in the output folder
    frame_filename = os.path.join(output_folder, f"frame_{frame_number}.png")
    cv2.imwrite(frame_filename, frame)
    frame_number += 1
