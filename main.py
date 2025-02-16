import cv2 as cv
import time
import numpy as np

print('\nCamera ON!!')
cam = cv.VideoCapture(1)

while True:
    ret, frame = cam.read()
    if not ret:
        break
 
    width, height, _ = frame.shape
    center = [height // 2, width // 2]

    # ||==================[ creating a mask ]==================||
    blank = np.zeros(frame.shape[:2], dtype='uint8')
    mask = cv.rectangle(blank.copy(), 
                        (center[0] - 270, center[1] - 150), 
                        (center[0] + 270, center[1] + 150), 
                        255, thickness=-1)  

    frame = cv.flip(frame, 1)
    
    # ||==================[ time and date ]==================||
    t = time.time()
    curr_date = time.strftime('%d-%m-%y', time.localtime(t)) 
    curr_time = time.strftime('%H:%M:%S', time.localtime(t))  
    
    masked_vid = cv.bitwise_and(frame, frame, mask=mask)

    cv.rectangle(masked_vid, 
                 (center[0] - 275, center[1] - 155), 
                 (center[0] + 275, center[1] + 155), 
                 (255, 255, 255), thickness=1)  


    (text_width, text_height), _ = cv.getTextSize(curr_date, cv.FONT_HERSHEY_TRIPLEX, 0.75, 1)
    text_x = center[0] - (text_width // 2)
    text_y = center[1] + (text_height // 2)

    cv.putText(masked_vid, 
               curr_date,
               (text_x - 200, text_y - 200), 
               fontFace=cv.FONT_HERSHEY_PLAIN, 
               fontScale=1, 
               thickness=1, 
               color=(255, 255, 255))
    
    cv.putText(masked_vid, 
               curr_time,
               (text_x - 200, text_y - 185), 
               fontFace=cv.FONT_HERSHEY_PLAIN, 
               fontScale=1, 
               thickness=1, 
               color=(255, 255, 255))

    cv.imshow('||====| Testing |====||', masked_vid)

    if cv.waitKey(1) == ord('q'):
        break

print('Camera OFF!!\n')
cam.release()
cv.destroyAllWindows()
