import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms

# Define the neural network architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 2)
    
    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = self.pool(x)
        x = x.view(-1, 32 * 8 * 8)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        return x

# Load the trained model from the .pth file
model = Net()
model.load_state_dict(torch.load('C:/Users/lolol/OneDrive/Документы/Курсовой проект/Сourse project (КП)/src/Train model/person_detection_model.pth'))
model.eval()

# Define the image transformations
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])

# Open the camera
cap = cv2.VideoCapture(0)

# Classify the camera images
while True:
    # Capture the camera image
    ret, frame = cap.read()
    
    # Convert the camera image to a PyTorch tensor
    image = transform(frame).unsqueeze(0)
    
    # Classify the image using the neural network
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output.data, 1)
    
    # Display the classification result on the camera image
    label = 'person' if predicted.item() == 1 else 'not a person'
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Person Detection', frame)
    
    # Exit on ESC
    if cv2.waitKey(1) == 27:
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()
