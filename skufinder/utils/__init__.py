import logging
import os
from pathlib import Path
import sys
import torch
import numpy as np

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


# get crops
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
            crops.append(crop)
        except Exception as e:
            logger.error(f"Error while cropping with coordinates:\t{box}")
            raise e
    return crops
