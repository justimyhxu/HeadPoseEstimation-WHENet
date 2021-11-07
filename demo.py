import numpy as np
import cv2
from whenet import WHENet
from utils import draw_axis

def crop_and_pred(img_path, bbox, model):
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x_min, y_min, x_max, y_max = bbox
    img_rgb = img_rgb[y_min:y_max, x_min:x_max]
    img_rgb = cv2.resize(img_rgb,(224,224))
    img_rgb = np.expand_dims(img_rgb, axis=0)
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0,0,0), 1)
    yaw, pitch, roll = model.get_angle(img_rgb)
    draw_axis(img, yaw, pitch, roll, tdx=(x_min+x_max)/2, tdy=(y_min+y_max)/2, size = abs(x_max-x_min))
    x_cent = int((x_min+x_max)/2) 
    y_cent = int((y_min+y_max)/2)
    cv2.putText(img, "yaw: {}".format(np.round(yaw)), (int(x_cent), int(y_cent)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 0), 1)
    cv2.putText(img, "pitch: {}".format(np.round(pitch)), (int(x_cent), int(y_cent) - 
15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 0), 1)
    cv2.putText(img, "roll: {}".format(np.round(roll)), (int(x_cent), int(y_cent)-30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 0), 1)
    cv2.imwrite(f'{img_path[:-4]}_infer.jpg',img)

if __name__ == "__main__":
    model = WHENet('WHENet.h5')
    root = 'SSample/'
    print(model.model.summary())

    with open(f'{root}/annotation.txt', 'r') as f:
        lines = f.readlines()

    for l in lines:
        filename, bbox =l.split(',')
        bbox = bbox.split(' ')
        # bbox = [int(b) for b in bbox]
        bbox = [0,0,256,256]
        crop_and_pred(root+filename,bbox, model)
