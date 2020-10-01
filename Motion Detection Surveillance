
#imports
import sys
import cv2
import time
import datetime
import imutils




def motion_detection():
    video_capture = cv2.VideoCapture(0) # value (0) selects the devices default camera
    time.sleep(2)

    first_frame = None # instinate the first fame

    while True:
        frame = video_capture.read()[1] # gives 2 outputs retval,frame - [1] selects frame
        text = 'Unoccupied'
	
        greyscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 

        gaussian_frame = cv2.GaussianBlur(greyscale_frame, (21,21),0)
        blur_frame = cv2.blur(gaussian_frame, (5,5)) 
        greyscale_image = blur_frame 
        if first_frame is None:
            first_frame = greyscale_image 
        else:
            pass


        frame = imutils.resize(frame, width=500)
        frame_delta = cv2.absdiff(first_frame, greyscale_image) 
	thresh = cv2.threshold(frame_delta, 100, 255, cv2.THRESH_BINARY)[1]
        dilate_image = cv2.dilate(thresh, None, iterations=2)
        cnt = cv2.findContours(dilate_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
        
        for c in cnt:
            if cv2.contourArea(c) > 800: 
                (x, y, w, h) = cv2.boundingRect(c) 

                cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 2) 
                
                text = 'Occupied'
                
		
            else:

                pass


        ''' now draw text and timestamp on security feed '''
        font = cv2.FONT_HERSHEY_SIMPLEX 
	
	
		

        cv2.putText(frame, '{+} Room Status: %s' % (text), 
            (10,20), cv2.FONT_HERSHEY_SIMPLEX , 0.5, (0, 0, 255), 2)
        
        cv2.imshow('Security Feed', frame)
        

        key = cv2.waitKey(1) & 0xFF 
        if key == ord('q'):
            cv2.destroyAllWindows()
            break
                


if __name__=='__main__':    
    motion_detection()











