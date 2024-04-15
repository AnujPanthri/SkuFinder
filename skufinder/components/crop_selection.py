import numpy as np
from skufinder.utils import logger

def is_not_low_res(crop:np.ndarray,min_res:int=5):
    """used to check if a crop is not low resolution
    
    Args:
        crop (np.ndarray) : crop image
        min_res (int) :  minimum allowed resolution
    
    Returns:
        bool (Bool) : True/False
    
    """
    h,w = crop.shape[:2]
    if(h<min_res or w<min_res):
        return False
    return True

def filter_crops(crops:list[np.ndarray]):
    """ used to filter out unusable crops
    
    Args:
        crops (list[np.ndarray]) : list of crops
    
    Returns:
        valid_crops (np.ndarray) : filtered out crops
    """

    valid_crops = []
    for crop in crops:
        is_valid = True
        is_valid &= is_not_low_res(crop)
        
        if is_valid:    valid_crops.append(crop)
    
    return valid_crops
