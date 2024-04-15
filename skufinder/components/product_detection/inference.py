from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image, ImageOps
import requests
import numpy as np
from skufinder.utils import device,logger,get_crops
from ensure import ensure_annotations

# - [x] model creation code 
# - [x] model weights loading code
# - [x] model's output to xmin,ymin,xmax,ymax bounding box coordinates for each detection
# - [x] bounding box to crops array




class DetrModel:

    def __init__(self):
        self.MODEL_TYPE = "detr-hf"
        self.create_load_model()

    def create_load_model(self):
        """create model and load weights

        Returns:
            model (DetrForObjectDetection): Detr object detection model trained on sku110k
        """

        logger.info("loading detr model")
        processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-101-dc5")
        model = DetrForObjectDetection.from_pretrained("isalia99/detr-resnet-101-dc5-sku110k")
        model = model.eval().to(device)
        logger.info("loaded detr model")

        self.processor = processor
        self.model = model
        return model

    
    def detect_objects(self,image:np.ndarray, thres:float|int=0.8) -> dict:
        """ detect objects in image

        Args:
            image (np.ndarray) : rgb image np.array 
            thres (int|float) : 0 to 1 confidence threshold
        
        Returns:
            results (dict["boxes","scores"]) : dictionary with keys boxes and scores of bounding boxes,
            each box is xmin, ymin, xmax, ymax

        """
        inputs = self.processor(images=image, return_tensors="pt").to(device)
        # print(type(inputs))
        outputs = self.model(**inputs)

        target_sizes = torch.tensor([image.shape[:2]])
        results = self.processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=thres)[0]
        
        results["scores"]=results["scores"].cpu().detach().numpy()
        results["boxes"]=results["boxes"].cpu().detach().numpy().astype("int32")
        results["labels"]=results["labels"].cpu().detach().numpy()
        
        del results["labels"]

        return results

    def detect_crops(self,image:np.ndarray, thres:float|int=0.8):
        """ used to detect objects and return their crops
        
        Args:
            image (np.ndarray) : rgb image np.array 
            thres (int|float) : 0 to 1 confidence threshold
        
        Returns:
            crops (list[np.ndarray]) : list of crops

        """
        results = self.detect_objects(image=image,thres=thres)
        crops = get_crops(image,results)
        return crops