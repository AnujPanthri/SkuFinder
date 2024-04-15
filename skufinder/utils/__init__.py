import logging
import os
from pathlib import Path
import sys
import torch
import numpy as np
import cv2
import shutil

# ======================================================general======================================================
# logging
__LOGFILE_PATH = Path('logs/dev.log')
if not os.path.exists(__LOGFILE_PATH):
    os.makedirs(__LOGFILE_PATH.parent,exist_ok=True)
    open(__LOGFILE_PATH,"w").close()


__logging_str = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"

logging.basicConfig(
    level= logging.INFO,
    format= __logging_str,

    handlers=[
        logging.FileHandler(__LOGFILE_PATH),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)



# pytorch device
device = "cuda:0" if torch.cuda.is_available() else "cpu"


def create_new_dir(path:str):
    """used to create directory

    Args:
        path(str) : folder path 
    """
    if os.path.exists(path):
        logger.info(f"Deleting old DIRECTORY:\t{path}")
        shutil.rmtree(path)
    logger.info(f"Creating DIRECTORY:\t{path}")
    os.makedirs(path,exist_ok=True)
    logger.info(f"Created DIRECTORY:\t{path}")


def save_images(images:list[np.ndarray], folder_name:str):
    """ used to save images
    
    Args:
        images (list[np.ndarray]|np.ndarray) : list/np.array of images
        folder_name (str) : folder name where we want to save the images at
    """
    
    create_new_dir(folder_name)

    for i,image in enumerate(images):
        try:

            filename = Path(folder_name) / Path(f"{i}.jpg")
            cv2.imwrite(str(filename),image[:,:,::-1])
            logger.info(f"saved image at {filename}")

        except Exception as e:
            logger.error(f"error while saving image with shape:\t{image.shape}")
            raise e
# ======================================================frame selection======================================================

# ======================================================product detection======================================================

def get_crops(image:np.ndarray, results:dict) -> list[np.ndarray]:
    """ used to get crops from results
        
        Args:
            image (np.ndarray) : rgb image np.array 
            results (dict["boxes","scores"]) : dictionary with keys boxes and scores of bounding boxes

        Results:
            crops (list[np.ndarray]) : list of crops
    """
    crops = []
    for box in results["boxes"]:
        try:
            crop = image[box[1]:box[3],box[0]:box[2]]
            if(crop.shape[0]==0 or crop.shape[1]==0):   continue    # not including crop with size like (0, 64, 3) or (42, 0, 3)
            crops.append(crop)
        except Exception as e:
            logger.error(f"Error while cropping with coordinates:\t{box}")
            raise e
    return crops

def draw_boxes(image:np.ndarray, results:dict) -> np.ndarray:
    """ used to get crops from results
        
        Args:
            image (np.ndarray) : rgb image np.array 
            results (dict["boxes","scores"]) : dictionary with keys boxes and scores of bounding boxes

        Results:
            crops (list[np.ndarray]) : list of crops
    """
    image = image.copy()

    for box in results["boxes"]:
        try:
            crop = image[box[1]:box[3],box[0]:box[2]]
            if(crop.shape[0]==0 or crop.shape[1]==0):   continue    # not including crop with size like (0, 64, 3) or (42, 0, 3)
            image = cv2.rectangle(image, (box[0],box[1]), (box[2],box[3]), (0,225,215), 4)
        except Exception as e:
            logger.error(f"Error while drawing box with coordinates:\t{box}")
            raise e
    return image

