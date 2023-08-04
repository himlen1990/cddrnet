import os
import numpy as np
from PIL import Image
import glob
import cv2

color_list = [[0, 128, 192], [128, 0, 0], [64, 0, 128],
              [192, 192, 128], [64, 64, 128], [64, 64, 0],
              [128, 64, 128], [0, 0, 192], [192, 128, 128],
              [128, 128, 128], [128, 128, 0]]

output_path = 'labels_process'
if not os.path.exists(output_path):
    os.makedirs(output_path)

input_path   = "labels_color"
files = glob.glob(input_path+'/*.png')
for label_file in files:

    new_img_file = label_file.replace(input_path, output_path)
    color_map = cv2.imread(label_file)
    color_map = cv2.cvtColor(color_map, cv2.COLOR_BGR2RGB)
    label = np.ones(color_map.shape[:2])*255
    for i, v in enumerate(color_list):
        label[(color_map == v).sum(2)==3] = i
    print(new_img_file)
    cv2.imwrite(new_img_file,label)
