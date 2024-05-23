import cv2
import numpy as np
from PIL import Image

def extract_frames(video_path, num_frames):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    
    for i in range(num_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * frame_count // num_frames)
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    cap.release()
    return frames

def blend_frames(frames):
    blended_frame = np.zeros_like(frames[0], dtype=np.float32)
    num_frames = len(frames)
    
    for i, frame in enumerate(frames):
        alpha = (i + 1) / num_frames
        blended_frame += frame * alpha
    
    blended_frame = np.clip(blended_frame / num_frames, 0, 255).astype(np.uint8)
    return blended_frame

def save_image(image, output_path):
    Image.fromarray(image).save(output_path)

video_path = 'video_dqn_3.3_trunc200.mp4'
output_path = 'image_dqn_3.3_trunc200.png'
num_frames = 10  # Number of frames to extract and blend

frames = extract_frames(video_path, num_frames)
blended_image = blend_frames(frames)
save_image(blended_image, output_path)
