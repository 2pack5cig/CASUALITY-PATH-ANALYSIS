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

#now casuality detection
lower_red1=np.array([0,100,100])
upper_red1=np.array([10,255,255])
mask_red1=cv2.inRange(hsv,lower_red1,upper_red1)

lower_red2=np.array([170,100,100])
upper_red2=np.array([180,255,255])
mask_red2=cv2.inRange(hsv,lower_red2,upper_red2)

red_mask=cv2.bitwise_or(mask_red1,mask_red2)

lower_yellow=np.array([20,100,100])
upper_yellow=np.array([34,250,255])
yellow_mask=cv2.inRange(hsv,lower_yellow,upper_yellow)

lower_white=np.array([0,0,200])
upper_white=np.array([180,40,255])
white_mask=cv2.inRange(hsv,lower_white,upper_white)
#ab casuality detection

colors=[("Red",red_mask,3),("Yellow",yellow_mask,2),("White",white_mask,1)]

casualties=[]

for color_name,mask,severity_score in colors:
    contours,hierarchy=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area=cv2.contourArea(cnt)
        if area<100:
            continue
        M=cv2.moments(cnt)
        if M["m00"]==0:
            continue
        cx=int(M["m10"]/M["m00"])
        cy=int(M["m01"]/M["m00"])
        peri=cv2.arcLength(cnt,True)
        approx=cv2.approxPolyDP(cnt,0.02*peri,True)
        corners=len(approx)
        hull=cv2.convexHull(cnt)
        hull_area=cv2.contourArea(hull)
        solidity=area/hull_area
        
        if corners==4:
            shape="square"
            age_score=2
        elif solidity<0.85:
            shape="star"
            age_score=1
        else:
            shape="circle"
            age_score=3
        
        px,py=approx[0][0]
        dx=px-cx
        dy=py-cy
        nx=px+dx+1
        ny=py+dy+1
        if light_mask[ny,nx]==255:
            level= 0
        elif med_mask[ny,nx]==255:
            level= 1
        elif dark_mask[ny,nx]==255:
            level= 2
        else:
            level= -1
        
        priority_score=severity_score*age_score
        
        casualties.append({"color":color_name,"shape":shape,"coords":(cx,cy),"level":level,"priority":priority_score})

for c in casualties:
    print(c.values())




