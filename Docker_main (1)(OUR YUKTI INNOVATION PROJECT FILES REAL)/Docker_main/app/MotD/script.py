import cv2

print("[INFO] Loaded Dependencies")
# Set Buffer Region
print("[USER] Set Buffer Region")
# Initialize webcam
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
# Set initial line coordinates
line1 = len(frame[0])//3
line2 = line1 + len(frame[0])//3
print("[INFO] Opening Camera...")
while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if line1 <0 or line1>=  len(frame[0]):
        line1 =0
    # Draw the line on the frame
    cv2.line(frame, (line1, 0), (line1, len(frame)), (0, 255, 0), 2)
    cv2.putText(frame, 'Left Boundary', (20, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    cv2.putText(frame, 'A <- Move Left', (20, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    cv2.putText(frame, 'D -> Move Right', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    cv2.putText(frame, 'Enter : Confirm', (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Display the frame
    cv2.imshow("Set Buffer Region", frame)

    # Wait for a key event
    key = cv2.waitKey(1) & 0xFF

    # Handle key events
    if key == 13:  # Press 'Enter' key to exit
        break
    if key == 27:  # Press 'Esc' key to exit
        break
    # elif key == 119:  # Press 'Up' arrow key
    #     y1 -= 10
    #     y2 -= 10
    # elif key == 115:  # Press 'Down' arrow key
    #     y1 += 10
    #     y2 += 10
    elif key == 97:  # Press 'Left' arrow key
        line1 -= 10
    elif key == 100:  # Press 'Right' arrow key
        line1 += 10
    elif key !=255:
        print(key)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if line2<=line1 or line2 >= len(frame[0]):
        line2 =line1+1
    # Draw the line on the frame
    cv2.line(frame, (line2, 0), (line2, len(frame)), (0, 255, 0), 2)
    cv2.line(frame, (line1, 0), (line1, len(frame)), (0, 255, 255), 2)
    cv2.putText(frame, 'Right Boundary', (20, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    cv2.putText(frame, 'A <- Move Left', (20, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    cv2.putText(frame, 'D -> Move Right', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    cv2.putText(frame, 'Enter : Confirm', (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)



    # Display the frame
    cv2.imshow("Set Buffer Region", frame)

    # Wait for a key event
    key = cv2.waitKey(1) & 0xFF

    # Handle key events
    if key == 13:  # Press 'Enter' key to exit
        break
    if key == 27:  # Press 'Esc' key to exit
        break
    # elif key == 119:  # Press 'Up' arrow key
    #     y1 -= 10
    #     y2 -= 10
    # elif key == 115:  # Press 'Down' arrow key
    #     y1 += 10
    #     y2 += 10
    elif key == 97:  # Press 'Left' arrow key
        line2 -= 10
    elif key == 100:  # Press 'Right' arrow key
        line2 += 10
    elif key !=255:
        print(key)

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()

#print(x1,x2, y1, y2)

# W   119
# S   115
# A   97
# D   100
#enter  13

print("[USER] Buffer Region Set...")

# Initialize video capture
video_capture = cv2.VideoCapture(0)

print("[INFO] Loading cv2.BackgroundSubtractorMOG2...")
bg_subtractor = cv2.createBackgroundSubtractorMOG2()
print("[INFO] Opening Camera...")

print("[INFO] Press Esc to close")
while True:
    ret, frame = video_capture.read()
    if not ret:
        break
    RoI = frame[0:len(frame), line1:line2]
    fg_mask = bg_subtractor.apply(RoI)
    cv2.imshow("Structured element", fg_mask)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    thresh_frame = cv2.threshold(fg_mask, 30, 255, cv2.THRESH_BINARY)[1]
    thresh_frame = cv2.dilate(thresh_frame, None, iterations = 2)
    cv2.imshow("Pre contor", thresh_frame)
    contours, _ = cv2.findContours(thresh_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        if cv2.contourArea(contour) > 5000:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (line1 + x, y), (line1 + x + w, y + h), (0, 255, 0), 2)
    cv2.line(frame, (line2, 0), (line2, len(frame)), (0, 255, 255), 1)
    cv2.line(frame, (line1, 0), (line1, len(frame)), (0, 255, 255), 1)

    cv2.imshow("Motion Detection", frame)
    cv2.imshow("Background", fg_mask)
    
    # Wait for a key press, but only for a short time (1 millisecond)
    key = cv2.waitKey(1) & 0xFF
    
    # Check if the 'q'  or 'esc' key was pressed to exit the loop
    if key == ord('q') or key == 27:
        break

    # Check if the 'p' key was pressed to perform a specific action
    elif key == ord('p'):
        print("Key 'p' pressed - Perform an action!")

video_capture.release()
cv2.destroyAllWindows()

print("[INFO] Closing Process...")