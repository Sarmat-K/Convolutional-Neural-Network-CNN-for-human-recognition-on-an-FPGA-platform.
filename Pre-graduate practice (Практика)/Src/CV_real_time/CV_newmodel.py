import cv2
import torch 
import torchvision 
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor 
from torchvision.transforms import ToTensor 
 
# Load a pre-trained model   
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True) 
 
# Replace the classifier with a new one that has 1 output channel (person or not person) 
num_classes = 2  # 1 class (person) + background 
in_features = model.roi_heads.box_predictor.cls_score.in_features 
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes) 
 
# Move the model to the GPU if available 
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') 
model.to(device) 
 
# Function to detect people in the camera image 
def detect_people(image):
    # Convert the image to a PyTorch tensor 
    image_t = ToTensor()(image).to(device)
    
    # Run the model on the image 
    model.eval() # Set to evaluation mode
    with torch.no_grad():
        outputs = model([image_t])
 
    # Check if there are any people in the image 
    for i in range(len(outputs)): 
        labels = outputs[i]['labels'] 
        if 1 in labels:
            return 1
 
    return 0
 
# Main program loop to capture camera images and display the results 
cap = cv2.VideoCapture(0) # 0 is for the default camera
while True:
    ret, frame = cap.read()
    
    # Detect people in the camera image 
    has_people = detect_people(frame)
    
    # Display the result on the image 
    if has_people:
        cv2.putText(frame, 'Person detected', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    else:
        cv2.putText(frame, 'No person detected', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    
    # Display the image in a window 
    cv2.imshow('Camera', frame)
    
    # Exit if the user pressed 'q'
    if cv2.waitKey(1) == ord('q'):
        break
 
cap.release()
cv2.destroyAllWindows()