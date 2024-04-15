import skufinder
from skufinder.utils import get_crops
import cv2


image = cv2.imread("J6R17B.jpg")[:,:,::-1]

model = skufinder.product_detection.inference.DetrModel()

# results = model.detect_objects(image)
# crops = get_crops(image,results)
crops = model.detect_crops(image)

print(len(crops))
print(crops[0].shape)
