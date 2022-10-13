import cv2
import numpy as np
import os

# choose codec according to format needed
fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
video = cv2.VideoWriter('video.avi', fourcc, 30, (1920, 1080))


path = './img1/'
img_list = os.listdir(path)
final_img_list = []
for i in img_list:
   if i[-4:] == '.jpg':
      final_img_list += i,
#print(final_img_list)

for j in final_img_list:
  img = cv2.imread(path + j)
  video.write(img)

cv2.destroyAllWindows()
video.release()
