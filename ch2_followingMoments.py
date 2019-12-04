from picamera.array import *
from picamera import PiCamera
import numpy as np
import cv2
import person
import time

#initialze the camera and grab a reference to the raw camera capture
camera=PiCamera()
camera.resolution=(640,480)
#camera.framerate=32
rawCapture=PiRGBArray(camera,size=(640,480))
#allow the camera to warmup
time.sleep(0.1)

fgbg=cv2.BackgroundSubtractorMOG()

kernel0p=np.ones((3,3),np.uint8)
kernelC1=np.ones((11,11),np.uint8)

#variables
font=cv2.FONT_HERSHEY_SIMPLEX
persons=[]
max_p_age=5
pid=1
areaTH=500

#capture frames from camera
for frame in camera.capture_continuous(rawCapture,format="bgr",use_video_port=True):
    image=frame.array
    fgmask=fgbg.apply(image)
    try:
        ret,imBin=cv2.threshold(fgmask,210,255,cv2.THRESH_BINARY)
        #opening (erode->dilate)
        mask=cv2.morphologyEx(imBin,cv2.MORPH_OPEN,kernel0p)
        #closing (dilate->erode)
        mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernelC1)
    except:
        #if there are no frames.......
        print('EOF')
        break

    (contours0,hierarchy)=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    for cnt in contours0:
        cv2.drawContours(image,cnt,-1,(0,255,0),3,8)
        area=cv2.contourArea(cnt)
        if area>areaTH:
            M=cv2.moments(cnt)
            cx=int(M['m10']/M['m00'])
            cy=int(M['m01']/M['m00'])
            x,y,w,h=cv2.boundingRect(cnt)

            new=True
            for i in persons:
                if abs(cx-i.getX())<=w and abs(cy-i.getY())<=h:

                    new=False
                    i.updateCoords(cx,cy)
                    break
                if new==True:
                    p=person.MyPerson(pid,cx,cy,max_p_age)
                    persons.append(p)
                    pid+=1

            cv2.circle(image,(cx,cy),5,(0,0,255),-1)
            img=cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.drawContours(image,cnt,-1,(0,255,0),3)

    for i in persons:
        if len(i.getTracks())>=5:
            pts=np.array(i.getTracks(),np.int32)
            pts=pts.reshape((-1,1,2))
            image=cv2.polylines(image,[pts],False,i,getRGB())
        if i.getId()==9:
            print str(i.getX()), ',' ,str(i.getY())
        cv2.putText(image,str(i.getId()),(i.getX(),i.getY()),font,0.3,i.getRGB(),1)

#show the frame
    cv2.imshow("Frame",image)
    key=cv2.waitKey(1) & 0xFF
    #clear the stream in preparation for the next frame
    rawCapture.truncate(0)

    #if the 'q' key was pressed, break from the loop
    if key == ord("q"):
        break
            
cv2.destroyAllWindows()


