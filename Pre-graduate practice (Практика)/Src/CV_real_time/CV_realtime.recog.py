import cv2

# Load the pre-trained HOG + Linear SVM model
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Open a video capture object for the camera
video_capture = cv2.VideoCapture(0)

while True:
    # Read a frame from the video stream
    ret, frame = video_capture.read()

    # Perform human detection on the frame
    boxes, weights = hog.detectMultiScale(frame, winStride=(4, 4), padding=(8, 8), scale=1.05)

    # Draw bounding boxes around the detected humans
    for (x, y, w, h) in boxes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the frame with bounding boxes
    cv2.imshow("Human Detection", frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close the windows
video_capture.release()
cv2.destroyAllWindows()
