import jetson.inference
import jetson.utils
import time
import cv2
import numpy as np 

timeStamp=time.time()
fpsFilt=0
#net=jetson.inference.detectNet('ssd-mobilenet-v1',threshold=.5)
net=jetson.inference.detectNet('ssd-mobilenet-v2',threshold=.5)
#net=jetson.inference.detectNet('ssd-inception-v2',threshold=.5)
#net=jetson.inference.detectNet('coco-dog',threshold=.5)
#net=jetson.inference.detectNet('coco-bottle',threshold=.5)
#net=jetson.inference.detectNet('coco-chair',threshold=.5)
#net=jetson.inference.detectNet('coco-airplane',threshold=.5)
#net=jetson.inference.detectNet('pednet',threshold=.5)
#net=jetson.inference.detectNet('multiped',threshold=.5)
#net=jetson.inference.detectNet('facenet',threshold=.5)


dispW=1280
dispH=720
#dispW=640
#dispH=480
flip=2
font=cv2.FONT_HERSHEY_SIMPLEX

cam=cv2.VideoCapture('/dev/video0')
cam.set(cv2.CAP_PROP_FRAME_WIDTH, dispW)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, dispH)

while True:
    _,img = cam.read()
    height=img.shape[0]
    width=img.shape[1]

    frame=cv2.cvtColor(img,cv2.COLOR_BGR2RGBA).astype(np.float32)
    frame=jetson.utils.cudaFromNumpy(frame)

    detections=net.Detect(frame, width, height)
    for detect in detections:
        ID=detect.ClassID
        top=int(detect.Top)
        left=int(detect.Left)
        bottom=int(detect.Bottom)
        right=int(detect.Right)
        item=net.GetClassDesc(ID)
        cv2.rectangle(img,(left,top),(right,bottom),(0,255,0),3)
        cv2.putText(img,item,(left,top+20),font,.95,(0,0,255),2)
    dt=time.time()-timeStamp
    timeStamp=time.time()
    fps=1/dt
    fpsFilt=.9*fpsFilt + .1*fps
    cv2.putText(img,str(round(fpsFilt,1))+' fps',(0,30),font,1,(0,0,255),2)
    cv2.imshow('detCam',img)
    cv2.moveWindow('detCam',0,0)
    if cv2.waitKey(1)==ord('q'):
        break
cam.release()
cv2.destroyAllWindows()
