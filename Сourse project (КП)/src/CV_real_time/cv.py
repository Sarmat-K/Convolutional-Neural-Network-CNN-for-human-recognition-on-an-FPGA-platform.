import cv2

def detect_person(frame):
    # Load the pre-trained model
    net = cv2.dnn.readNetFromCaffe('models/MobileNetSSD_deploy.prototxt',
                                   'models/MobileNetSSD_deploy.caffemodel')
    
    # Create a blob from the frame
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    
    # Set the input for the model
    net.setInput(blob)
    
    # Perform object detection
    detections = net.forward()
    
    # Loop over the detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        
        # If the confidence is above a certain threshold (e.g., 0.5)
        if confidence > 0.5:
            class_id = int(detections[0, 0, i, 1])
            
            # Check if the detected object is a person
            if class_id == 15:
                return True
    
    return False

# Open the notebook camera
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the camera
    ret, frame = cap.read()
    
    # Perform person detection on the frame
    is_person_present = detect_person(frame)
    
    # Display the result
    if is_person_present:
        cv2.putText(frame, 'Person Detected', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        cv2.putText(frame, 'No Person Detected', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Display the frame
    cv2.imshow('Real-time Person Detection', frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the windows
cap.release()
cv2.destroyAllWindows()
