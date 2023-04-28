import torch
import cv2

# Load the saved model
model = torch.load('human_detection_model.pth')

# Set the device to CPU or GPU depending on availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Initialize the camera
cap = cv2.VideoCapture(0)

# Loop over frames from the video stream
while True:
    # Read the next frame from the camera
    ret, frame = cap.read()
    
    # Convert the frame to a PyTorch tensor
    tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    
    # Move the tensor to the device
    tensor = tensor.to(device)
    
    # Pass the tensor through the model and get the output
    output = model(tensor)
    
    # Convert the output tensor to a numpy array
    output = output.detach().cpu().numpy().squeeze()
    
    # Display the frame with a bounding box around the detected human
    if output > 0.5:
        cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 2)
    cv2.imshow('frame', frame)
    
    # Check if the user has pressed the 'q' key to quit the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
