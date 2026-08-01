import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class ResidualDepthwiseBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()

        self.depthwise = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=channels,
            bias=False,
        )

        self.bn1 = nn.BatchNorm2d(channels)
        self.activation = nn.GELU()

        self.pointwise = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=1,
            bias=False,
        )

        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = self.depthwise(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.pointwise(out)
        out = self.bn2(out)

        out = out + residual
        out = self.activation(out)

        return out

class ModernLightCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        
        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(32),
            nn.GELU(),
            )

        self.stage1 = nn.Sequential(
            ResidualDepthwiseBlock(32),
            ResidualDepthwiseBlock(32),
        )

        self.downsample1 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(64),
            nn.GELU(),
        )

        self.stage2 = nn.Sequential(
            ResidualDepthwiseBlock(64),
            ResidualDepthwiseBlock(64)
        )

        self.downsample2 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(128),
            nn.GELU()
        )

        self.stage3 = nn.Sequential(
            ResidualDepthwiseBlock(128),
            ResidualDepthwiseBlock(128)
        )

        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.flatten = nn.Flatten(start_dim=1)
        self.classifier = nn.Linear(
            in_features=128,
            out_features=num_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)

        x = self.downsample1(x)
        x = self.stage2(x)

        x = self.downsample2(x)
        x = self.stage3(x)

        x = self.pool(x)
        x = self.flatten(x)
        logits = self.classifier(x)

        return logits

model = ModernLightCNN(num_classes=10)

x = torch.randn(8, 3, 32, 32)
logits = model(x)

print(logits.shape)

#DATA
mean = (0.4914, 0.4822, 0.4465)
std = (0.2470, 0.2435, 0.2616)

train_transforms = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])

test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])

train_dataset = datasets.CIFAR10(
    root='/kaggle/working/',   
    train=True,           
    download=True,         
    transform=train_transforms
)

test_dataset = datasets.CIFAR10(
    root='/kaggle/working/',   
    train=False,           
    download=True,         
    transform=test_transforms
)

train_loader = DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True,
    num_workers=0,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=128,
    shuffle=False,
    num_workers=0,
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = model.to(device)

#model

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4,
    weight_decay=1e-4,
)

#train

model.train()
num_epochs = 10
running_loss = 0.0
correct,total=0,0

for epoch in range(num_epochs):
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
    
        optimizer.zero_grad()
    
        logits = model(images)
        loss = criterion(logits, labels)
    
        loss.backward()
        optimizer.step()
        if total%100==0:
            print(loss.item())
        running_loss += loss.item()
        predictions = logits.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

accuracy = correct / total
train_loss = running_loss / len(train_loader)

model.eval()

test_loss_sum = 0.0
test_correct = 0
test_total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        test_loss_sum += loss.item()
        predictions = logits.argmax(dim=1)
        test_correct += (predictions == labels).sum().item()
        test_total += labels.size(0)

test_accuracy = test_correct / test_total
test_loss = test_loss_sum / len(test_loader)

