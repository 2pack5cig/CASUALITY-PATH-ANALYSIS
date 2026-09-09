import cv2
import numpy as np
import matplotlib.pyplot as plt

img=cv2.imread('images/IMG-20260831-WA0027.jpg')
img_rgb=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
hsv=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#obstacle ka mask
lower_black=np.array([0, 0, 0])
upper_black=np.array([65, 65, 65])   
obstacle_mask=cv2.inRange(hsv, lower_black, upper_black)
lower_blue=np.array([125, 200, 220])
upper_blue=np.array([130, 230, 235])
river_mask=cv2.inRange(hsv, lower_blue, upper_blue)
#green regions ka mask
lower_light_green=np.array([35, 40, 206])
upper_light_green=np.array([85, 255, 255])
light_mask=cv2.inRange(hsv, lower_light_green, upper_light_green)

lower_med_green=np.array([35, 40, 131])
upper_med_green=np.array([85, 255, 205])
med_mask=cv2.inRange(hsv, lower_med_green, upper_med_green)

lower_dark_green=np.array([35, 40, 0])
upper_dark_green=np.array([85, 255, 130])
dark_mask=cv2.inRange(hsv, lower_dark_green, upper_dark_green)
#dd the masks which are traversable and rem
traversable_mask=cv2.bitwise_or(light_mask, med_mask)
traversable_mask=cv2.bitwise_or(traversable_mask, dark_mask)
non_traversable_mask=cv2.bitwise_or(obstacle_mask, river_mask)
final_mask=cv2.bitwise_not(non_traversable_mask)

plt.imshow(final_mask, cmap='gray')
plt.title("Final")
plt.axis('on')
plt.savefig('outputs/mask_output.png')
plt.show()