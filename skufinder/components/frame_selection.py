import cv2
from pathlib import Path
from tqdm import tqdm
from skufinder.utils import logger
import numpy as np
from ensure import ensure_annotations


@ensure_annotations
def get_frames(video_path:str,frame_interval=10) -> np.ndarray:
    """reads video and return frames
    
    Args:
        video_path (str): path of the video

    Returns:
        frames (np.ndarray): array of frames of video
    """ 
        
    video_obj = cv2.VideoCapture(video_path)
    total_frames = int(video_obj.get(cv2.CAP_PROP_FRAME_COUNT))
    video_path = Path(video_path)
    
    logger.info(f"Video Name: {video_path}")
    logger.info(f"Total frames in the video: {total_frames}")
    logger.info(f"Extracting every {frame_interval}th frame...")

    frames = []
    for i in tqdm(range(total_frames)):
        
        ret, frame = video_obj.read()
        
        # If frame read is unsuccessful, break the loop
        if not ret:
            logger.error("frame unsuccessful")
            break
        
        if (i+1) % frame_interval == 0:
            frames.append(frame[:,:,::-1])
            
    video_obj.release()
    frames = np.array(frames)

    return frames

@ensure_annotations
def filter_frames(frames:np.ndarray) -> np.ndarray:
    """filter unusable/blurry frames
    
    TODO: filtering logic based on blur

    Args:
        frames (np.ndarray): frames of video

    Returns:
        frames (np.ndarray): filtered out frames     
    """
    
    return frames