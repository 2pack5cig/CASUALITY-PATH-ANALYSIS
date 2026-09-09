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
#the masks which are traversable and not
traversable_mask=cv2.bitwise_or(light_mask, med_mask)
traversable_mask=cv2.bitwise_or(traversable_mask, dark_mask)
non_traversable_mask=cv2.bitwise_or(obstacle_mask, river_mask)
kernel=np.ones((20,20),np.uint8)
inflated=cv2.dilate(non_traversable_mask,kernel,iterations=1)
final_mask=cv2.bitwise_not(inflated)
plt.imshow(final_mask, cmap='gray')
plt.title("Final")
plt.axis('on')
plt.savefig('outputs/mask_output.png')
plt.show()
#ab casuality detection
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
#ab contouring
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
        nx=px+dx
        ny=py+dy
        nx=max(0,min(1279,nx))
        ny=max(0,min(719,ny))
        if light_mask[ny,nx]==255:
            level=0
        elif med_mask[ny,nx]==255:
            level=1
        elif dark_mask[ny,nx]==255:
            level=2
        else:
            level=-1
        priority_score=severity_score*age_score
        casualties.append({"color":color_name,"shape":shape,"coords":(cx,cy),"level":level,"priority":priority_score})
for c in casualties:
    print(c.values())
#orange start ka mask
lower_orange=np.array([8,100,100])
upper_orange=np.array([20,255,255])
mask_orange=cv2.inRange(hsv,lower_orange,upper_orange)
#purple destination ka mask
lower_purple=np.array([130,80,80])
upper_purple=np.array([160,255,255])
mask_purple=cv2.inRange(hsv,lower_purple,upper_purple)
#ab start goal detection
masks=[("orange",mask_orange),("purple",mask_purple)]
points={}
for name,mask in masks:
    contours,hierarchy=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area=cv2.contourArea(cnt)
        if area<300:
            continue
        M=cv2.moments(cnt)
        if M["m00"]==0:
            continue
        cx=int(M["m10"]/M["m00"])
        cy=int(M["m01"]/M["m00"])
        peri=cv2.arcLength(cnt,True)
        approx=cv2.approxPolyDP(cnt,0.02*peri,True)
        corners=len(approx)
        if corners!=3:
            continue
        print(name,"at",cx,cy,"corners:",corners,"area:",round(area,1))
        points[name]=np.array([cx,cy])
start=points.get("orange")
goal=points.get("purple")
scale_o=0.25
small_o=cv2.resize(final_mask,None,fx=scale_o,fy=scale_o,interpolation=cv2.INTER_NEAREST)
ho,wo=small_o.shape
ordered=[]
remaining=casualties[:]
cur_pt=(int(start[0]),int(start[1]))
while len(remaining)>0:
    best=None
    best_len=1000000000
    for cand in remaining:
        sx=int(cur_pt[0]*scale_o)
        sy=int(cur_pt[1]*scale_o)
        gx=int(cand["coords"][0]*scale_o)
        gy=int(cand["coords"][1]*scale_o)
        s0=(sx,sy)
        g0=(gx,gy)
        q=[s0]
        head=0
        came={s0:None}
        visited=set([s0])
        while head<len(q):
            x,y=q[head]
            head+=1
            if (x,y)==g0:
                break
            if x+1<wo and small_o[y,x+1]!=0 and (x+1,y) not in visited:
                visited.add((x+1,y))
                came[(x+1,y)]=(x,y)
                q.append((x+1,y))
            if x-1>=0 and small_o[y,x-1]!=0 and (x-1,y) not in visited:
                visited.add((x-1,y))
                came[(x-1,y)]=(x,y)
                q.append((x-1,y))
            if y+1<ho and small_o[y+1,x]!=0 and (x,y+1) not in visited:
                visited.add((x,y+1))
                came[(x,y+1)]=(x,y)
                q.append((x,y+1))
            if y-1>=0 and small_o[y-1,x]!=0 and (x,y-1) not in visited:
                visited.add((x,y-1))
                came[(x,y-1)]=(x,y)
                q.append((x,y-1))
        if g0 not in came:
            l=1000000000
        else:
            l=0
            temp=g0
            while temp!=s0:
                temp=came[temp]
                l+=1
        if l<best_len:
            best_len=l
            best=cand
    ordered.append(best)
    remaining.remove(best)
    cur_pt=best["coords"]
s=(int(start[0]),int(start[1]))
g=(int(goal[0]),int(goal[1]))
targets=[]
for c in ordered:
    targets.append((int(c["coords"][0]),int(c["coords"][1])))
targets.append(g)
full_path=[]
travelled=0
prev=s
scores=[]
h,w=final_mask.shape
idx=0
for cur in targets:
    q=[prev]
    head=0
    came={prev:None}
    visited=set([prev])
    while head<len(q):
        x,y=q[head]
        head+=1
        if (x,y)==cur:
            break
        if x+1<w and final_mask[y,x+1]!=0 and (x+1,y) not in visited:
            visited.add((x+1,y))
            came[(x+1,y)]=(x,y)
            q.append((x+1,y))
        if x-1>=0 and final_mask[y,x-1]!=0 and (x-1,y) not in visited:
            visited.add((x-1,y))
            came[(x-1,y)]=(x,y)
            q.append((x-1,y))
        if y+1<h and final_mask[y+1,x]!=0 and (x,y+1) not in visited:
            visited.add((x,y+1))
            came[(x,y+1)]=(x,y)
            q.append((x,y+1))
        if y-1>=0 and final_mask[y-1,x]!=0 and (x,y-1) not in visited:
            visited.add((x,y-1))
            came[(x,y-1)]=(x,y)
            q.append((x,y-1))
        if x+1<w and y+1<h and final_mask[y+1,x+1]!=0 and (x+1,y+1) not in visited:
            visited.add((x+1,y+1))
            came[(x+1,y+1)]=(x,y)
            q.append((x+1,y+1))
        if x-1>=0 and y+1<h and final_mask[y+1,x-1]!=0 and (x-1,y+1) not in visited:
            visited.add((x-1,y+1))
            came[(x-1,y+1)]=(x,y)
            q.append((x-1,y+1))
        if x+1<w and y-1>=0 and final_mask[y-1,x+1]!=0 and (x+1,y-1) not in visited:
            visited.add((x+1,y-1))
            came[(x+1,y-1)]=(x,y)
            q.append((x+1,y-1))
        if x-1>=0 and y-1>=0 and final_mask[y-1,x-1]!=0 and (x-1,y-1) not in visited:
            visited.add((x-1,y-1))
            came[(x-1,y-1)]=(x,y)
            q.append((x-1,y-1))
    seg=[cur]
    temp=cur
    while temp!=prev:
        temp=came[temp]
        seg.append(temp)
    seg.reverse()
    #smooth only this leg, keeps cur waypoint
    sseg=[seg[0]]
    a=0
    while a<len(seg)-1:
        b=len(seg)-1
        while b>a+1:
            x0,y0=seg[a]
            x1,y1=seg[b]
            dx=x1-x0
            dy=y1-y0
            steps=max(abs(dx),abs(dy))
            ok=True
            for k in range(steps+1):
                x=int(x0+dx*k/steps)
                y=int(y0+dy*k/steps)
                if final_mask[y,x]==0:
                    ok=False
                    break
            if ok:
                break
            b-=1
        sseg.append(seg[b])
        a=b
    seg=sseg
    if len(full_path)==0:
        full_path=seg
    else:
        full_path=full_path+seg[1:]
    travelled+=(len(seg)-1)
    if idx<len(ordered):
        c=ordered[idx]
        disp=np.linalg.norm(np.array(cur)-np.array(s))
        score=(disp/travelled)*c["priority"] if travelled>0 else 0
        c["disp"]=round(float(disp),1)
        c["dist"]=travelled
        c["score"]=round(float(score),2)
        scores.append(c["score"])
    prev=cur
    idx+=1
time_light=0
time_med=0
time_dark=0
for x,y in full_path:
    if light_mask[y,x]!=0:
        time_light+=1.0/20.0
    elif med_mask[y,x]!=0:
        time_med+=1.0/15.0
    elif dark_mask[y,x]!=0:
        time_dark+=1.0/10.0
total_time=time_light+time_med+time_dark
print("Number of casualties =",len(ordered))
print("Casualty coordinates =",[c["coords"] for c in ordered])
for c in ordered:
    print(c["coords"],c["shape"],c["color"],"priority",c["priority"],"disp",c["disp"],"dist",c["dist"],"score",c["score"])
print("Rover Path =",full_path)
print("Casualty Scores =",scores)
print("Total Path Score =",round(sum(scores),2))
print("Total Time =",round(total_time,2),"seconds")
full_path=np.array(full_path)
plt.imshow(img_rgb)
plt.plot(full_path[:,0],full_path[:,1],'r-',linewidth=2)
plt.scatter([s[0]],[s[1]],c='orange',s=100)
plt.scatter([g[0]],[g[1]],c='purple',s=100)
for c in ordered:
    plt.scatter([c["coords"][0]],[c["coords"][1]],c='cyan',s=40)
plt.axis('off')
plt.savefig('outputs/path_full.png')
plt.show()


