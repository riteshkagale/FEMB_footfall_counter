from picamera.array import *
from picamera import PiCamera
import numpy as np
import cv2
import person
import time

#Input and Output Counter
cnt_up=0
cnt_down=0

#initialze the camera and grab a reference to the raw camera capture
camera=PiCamera()
w=640
h=480
camera.resolution=(w,h)
#camera.framerate=32
rawCapture=PiRGBArray(camera,size=(w,h))
#allow the camera to warmup
time.sleep(0.1)
frameArea=h*w
areaTH=frameArea/250
print 'Area Threshold',areaTH

#Input/Output lines
line_up=int(2*(h/5))#192
line_down=int(3*(h/5))#288

up_limit=int(1*(h/5))#96
down_limit=int(4*(h/5))#384

print "Red line y:",str(line_down)#displays Red Line=288
print "Blue line y:", str(line_up)#displays Blue Line=192
line_down_color=(255,0,0)#color blue
line_up_color=(0,0,255)#color red

pt1=[0,line_down];#[0,288]
pt2=[w,line_down];
pts_L1=np.array([pt1,pt2],np.int32)
pts_L1=pts_L1.reshape((-1,1,2))
pt3=[0,line_up];
pt4=[w,line_up];
pts_L2=np.array([pt3,pt4],np.int32)
pts_L2=pts_L2.reshape((-1,1,2))

pt5=[0,up_limit];
pt6=[w,up_limit];
pts_L3=np.array([pt5,pt6],np.int32)
pts_L3=pts_L3.reshape((-1,1,2))
pt7=[0,down_limit];
pt8=[w,down_limit];
pts_L4=np.array([pt7,pt8],np.int32)
pts_L4=pts_L4.reshape((-1,1,2))

fgbg=cv2.BackgroundSubtractorMOG(history=3,nmixtures=5,backgroundRatio=0.0001)

kernel0p=np.ones((5,5),np.uint8)
kernelC1=np.ones((11,11),np.uint8)

#variables
font=cv2.FONT_HERSHEY_SIMPLEX
persons=[]
max_p_age=5
pid=1

#capture frames from camera
for frame in camera.capture_continuous(rawCapture,format="bgr",use_video_port=True):
    image=frame.array

    for i in persons:
        i.age_one()
        
    fgmask=fgbg.apply(image)
    try:
        ret,imBin=cv2.threshold(fgmask,50,255,cv2.THRESH_BINARY)
        #opening (erode->dilate)
        mask=cv2.morphologyEx(imBin,cv2.MORPH_OPEN,kernel0p)
        #closing (dilate->erode)
        mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernelC1)
    except:
        #if there are no frames.......
        print('EOF')
        print'UP:',cnt_up
        print 'DOWN:',cnt_down
        break

    #RTER_EXTERNAL returns only extreme outer flags. All child contours are left behind
    (contours0,hierarchy)=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    for cnt in contours0:
        area=cv2.contourArea(cnt)
        if area>areaTH:
            M=cv2.moments(cnt)#moments helps to calculate features like center of mass of the object, area of the object..etc
            cx=int(M['m10']/M['m00'])#from this moments,extract useful  data like area,centroid.
            cy=int(M['m01']/M['m00'])
            x,y,w,h=cv2.boundingRect(cnt)

            new=True
            if cy in range(up_limit,down_limit):
                for i in persons:
                    if abs(cx-i.getX())<=w and abs(cy-i.getY())<=h:

                        #The object is near one already detected before
                        new=False
                        i.updateCoords(cx,cy)
                        if i.going_up(line_down,line_up)==True:
                            cnt_up+=1;
                            print "ID:",i.getId(),'crossed going up at',time.strftime("%c")
                        elif i.going_down(line_down,line_up)==True:
                            cnt_down+=1;
                            print "ID:",i.getId(),'crossed going down at',time.strftime("%c")
                        break

                    if i.getState()=='1':
                        if i.getDir()=='down' and i.getY()>down_limit:
                            i.setDone()
                        elif i.getDir()=='up' and i.getY()<up_limit:
                            i.setDone()

                    if i.timedOut():
                        #remove from the list persons
                        index=persons.index(i)
                        persons.pop(index)
                        del i #Free i memory
                        
                        
                if new==True:
                    p=person.MyPerson(pid,cx,cy,max_p_age)
                    persons.append(p)
                    pid+=1

            cv2.circle(image,(cx,cy),5,(0,0,255),-1)
            img=cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
           
    for i in persons:
        cv2.putText(image,str(i.getId()),(i.getX(),i.getY()),font,0.3,i.getRGB(),1)

    str_up='UP: '+str(cnt_up)
    str_down='DOWN: '+str(cnt_down)
    frame=cv2.polylines(image,[pts_L1],False,line_down_color,thickness=2)
    frame=cv2.polylines(image,[pts_L2],False,line_up_color,thickness=2)
    frame=cv2.polylines(image,[pts_L3],False,(255,255,255),thickness=1)
    frame=cv2.polylines(image,[pts_L4],False,(255,255,255),thickness=1)
    cv2.putText(image,str_up,(10,40),font,0.5,(255,255,255),2)
    cv2.putText(image,str_up,(10,40),font,0.5,(0,0,255),1)
    cv2.putText(image,str_down,(10,90),font,0.5,(255,255,255),2)
    cv2.putText(image,str_down,(10,90),font,0.5,(255,0,0),1)

    #show the frame
    cv2.imshow("Frame",image)
    key=cv2.waitKey(1) & 0xFF
    #clear the stream in preparation for the next frame
    rawCapture.truncate(0)

    #if the 'q' key was pressed, break from the loop
    if key == ord("q"):
        break
            
cv2.destroyAllWindows()


