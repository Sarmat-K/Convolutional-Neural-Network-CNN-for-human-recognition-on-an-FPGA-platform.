import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone


# Define the neural network architecture
class FasterRCNNNet(nn.Module):
    def __init__(self):
        super(FasterRCNNNet, self).__init__()
        self.backbone = resnet_fpn_backbone('resnet50', pretrained=True)
        self.rpn = nn.Conv2d(in_channels=self.backbone.out_channels, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.head = FastRCNNPredictor(self.backbone.out_channels, num_classes=2)

    def forward(self, x):
        features = self.backbone(x)
        proposals, _ = self.rpn(features)
        detections = self.head(features, proposals)
        return detections, {}


# Define the dataset and data loader
def collate_fn(batch):
    return tuple(zip(*batch))[0]  # Return only the image data


if __name__ == '__main__':
    # Define the paths to the COCO dataset
    train_data = datasets.CocoDetection(root='C:/Users/lolol/OneDrive/Документы/СOCO Dataset/train2017', annFile='C:/Users/lolol/OneDrive/Документы/СOCO Dataset/annotations/instances_train2017.json', transform=transforms.ToTensor())
    test_data = datasets.CocoDetection(root='C:/Users/lolol/OneDrive/Документы/СOCO Dataset/test2017', annFile='C:/Users/lolol/OneDrive/Документы/СOCO Dataset/annotations/instances_val2017.json', transform=transforms.ToTensor())

    # Define the data loader
    train_loader = DataLoader(train_data, batch_size=8, shuffle=True, num_workers=4, collate_fn=collate_fn)
    test_loader = DataLoader(test_data, batch_size=8, shuffle=False, num_workers=4, collate_fn=collate_fn)

    # Instantiate the neural network and define the loss function and optimizer
    model = FasterRCNNNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train the neural network
    for epoch in range(10):
        print(1)
        for batch_idx, images in enumerate(train_loader):  # images only, not targets
            print(2)
            # Forward pass
            detections, _ = model(images)
            loss = sum(loss for loss in detections.values())

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f'Epoch [{epoch+1}/{10}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}')
            # Print training progress
            if batch_idx % 100 == 0:

                print(f'Epoch [{epoch+1}/{10}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}')

        # Evaluate the neural network on the test set
        with torch.no_grad():
            correct = 0
            total = 0
            for images, targets in test_loader:
                detections, _ = model(images, targets)
                for detection in detections:
                    if detection['labels'].item() == 1:  # person class
                        correct += 1
                total += len(targets)
            accuracy = 100 * correct / total
            print(f'Test Accuracy: {accuracy:.2f}%')

    # Save the trained model
    torch.save(model.state_dict(), 'FasterRCNNNet_model.pth')
    print('Model saved!')
