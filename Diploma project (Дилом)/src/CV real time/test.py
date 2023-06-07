import cv2
import torch
import torchvision.transforms as transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn

# # Load the pre-trained model
# model = fasterrcnn_resnet50_fpn(pretrained=True)
# model.eval()

# # Set device to GPU if available, otherwise use CPU
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = model.to(device)

# # Load the COCO class labels
# LABELS = [
#     'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
#     'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
#     'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
#     'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella',
#     'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
#     'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
#     'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass', 'cup', 'fork',
#     'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli',
#     'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
#     'bed', 'N/A', 'dining table', 'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop',
#     'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
#     'sink', 'refrigerator', 'N/A', 'book', 'clock', 'vase', 'scissors',
#     'teddy bear', 'hair drier', 'toothbrush'
# ]

# # Transformations to apply to input image
# transform = transforms.Compose([
#     transforms.ToTensor(),
# ])

# # Function to perform object detection on the input image
# def detect_objects(image):
#     image = transform(image).unsqueeze(0).to(device)
#     outputs = model(image)
    
#     # Get the predicted bounding boxes, labels, and scores
#     boxes = outputs[0]['boxes'].detach().cpu().numpy()
#     labels = outputs[0]['labels'].detach().cpu().numpy()
#     scores = outputs[0]['scores'].detach().cpu().numpy()

#     # Filter out detections with low confidence
#     threshold = 0.5
#     filtered_boxes = boxes[scores > threshold]
#     filtered_labels = labels[scores > threshold]
    
#     # Draw bounding boxes on the image
#     for box, label in zip(filtered_boxes, filtered_labels):
#         x1, y1, x2, y2 = box.astype(int)
#         cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
#         cv2.putText(image, LABELS[label], (x1, y1 - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    
#     return image

# # Function to read frames from the camera and perform object detection
# def process_camera_feed():
#     capture = cv2.VideoCapture(0)  # Use the default camera (change index if needed)
#     while True:
#         ret, frame = capture.read()
#         if not ret:
#             break
        
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         frame = detect_objects(frame)
#         frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
#         cv2.imshow('Object Detection', frame)
        
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
    
#     capture.release()
#     cv2.destroyAllWindows()
    
    

# # Start processing the camera feed
# # process_camera_feed()

capture = cv2.VideoCapture(0)  # Use the default camera (change index if needed)
while True:
        ret, frame = capture.read()
    

        
        cv2.imshow('Object Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
