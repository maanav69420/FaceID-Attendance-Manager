import cv2 as cv
import time

print('\nCamera ON!!')
cam = cv.VideoCapture(1)

"""
0 ----> for using camo-studio
1 ----> for using laptop's camera
"""

while True:
    _ , frame = cam.read()

    height , width , _ = frame.shape

    frame =cv.flip(frame , 1)

    curr_time = time.strftime("%H:%M:%S", time.localtime())
    cv.putText(frame 
               , str(curr_time)
               , (18 , 70) 
               , fontFace= cv.FONT_HERSHEY_TRIPLEX
               , fontScale= 1
               , thickness= 3 
               , color= (98 , 37 , 250)
               )

    if cv.waitKey(1) == ord('q'):
        break
    
    cv.imshow('||====| Testing |====||', frame)

print('Camera OFF!!\n')
cam.release()
cv.destroyAllWindows()