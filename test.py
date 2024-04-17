import skufinder
from skufinder.utils import get_crops,save_images,draw_boxes
import cv2


# image = cv2.imread("J6R17B.jpg")[:,:,::-1]
image = cv2.imread("IMG_0491_frame_490.jpg")[:,:,::-1]

model = skufinder.product_detection.inference.DetrModel()

results = model.detect_objects(image)

detection_image = draw_boxes(image,results)
save_images([detection_image],"artifacts/image/")

crops = get_crops(image,results)
crops = skufinder.crop_selection.filter_crops(crops)

print(len(crops))
save_images(crops,"artifacts/crops/")