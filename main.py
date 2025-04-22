import cv2 as cv
import numpy as np
from datetime import datetime, timedelta
import os

def is_valid(now, time_end, frame, save_path, curr_time):
    if now >= time_end:
        # Replace colons with dashes for the filename
        filename = f"{curr_time.replace(':', '-')}.png"
        cv.imwrite(os.path.join(save_path, filename), frame)
        return False
    return True

print('\nCamera ON!!')
cam = cv.VideoCapture(1)  # Ensure the correct camera index

time_start = datetime.now()
delta = timedelta(seconds=6)
time_end = time_start + delta 

# ||==============[ directory made ]==============||
save_directory = f"D:\\Entries\\{datetime.today().date()}"  # Use double backslashes
os.makedirs(save_directory, exist_ok=True)

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
    now = datetime.now()
    curr_date = now.strftime('%d-%m-%y') 
    curr_time = now.strftime('%H-%M-%S')  # Replace colons with dashes for time formatting
    
    # || ====[ is valid function ]==== ||
    if not is_valid(now, time_end, frame, save_directory, curr_time):
        break  
    
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

    # Wait for a key press for a short duration to allow the window to update
    if cv.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        break

print('Camera OFF!!\n')
cam.release()
cv.destroyAllWindows()