import os
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset


# ============================================================
# CONFIG
# ============================================================

SEED = 42

DATA_ROOT = "/kaggle/input/datasets/pankrzysiu/cifar10-python"
CHECKPOINT_PATH = "/kaggle/working/best_model.pth"

NUM_CLASSES = 10
BATCH_SIZE = 256
NUM_EPOCHS = 50

LEARNING_RATE = 0.05
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4

VAL_SIZE = 5_000

PATIENCE = 10
MIN_DELTA = 1e-4

NUM_WORKERS = min(os.cpu_count() or 2, 4)


# ============================================================
# REPRODUCIBILITY
# ============================================================

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

pin_memory = device.type == "cuda"

print(f"Device: {device}")
print(f"Available GPUs: {torch.cuda.device_count()}")
print(f"DataLoader workers: {NUM_WORKERS}")


# ============================================================
# MODEL
# ============================================================

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
            ResidualDepthwiseBlock(64),
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
            nn.GELU(),
        )

        self.stage3 = nn.Sequential(
            ResidualDepthwiseBlock(128),
            ResidualDepthwiseBlock(128),
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


model = ModernLightCNN(num_classes=NUM_CLASSES)
model = model.to(device)

# Проверяем shape ДО DataParallel.
# Для проверки модель переводим в eval, чтобы BatchNorm
# не обновлял running statistics.
model.eval()

dummy_input = torch.randn(
    8, 3, 32, 32,
    device=device,
)

with torch.no_grad():
    dummy_logits = model(dummy_input)

print(f"Model output shape: {dummy_logits.shape}")

# Возвращаем режим обучения.
model.train()

# Только после dummy-проверки оборачиваем модель.
if torch.cuda.device_count() > 1:
    print(f"Используем {torch.cuda.device_count()} GPU")
    model = nn.DataParallel(model)

trainable_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)

print(f"Trainable parameters: {trainable_parameters:,}")


# ============================================================
# DATA
# ============================================================

mean = (0.4914, 0.4822, 0.4465)
std = (0.2470, 0.2435, 0.2616)

train_transforms = transforms.Compose([
    transforms.RandomCrop(
        size=32,
        padding=4,
    ),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=mean,
        std=std,
    ),
])

eval_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=mean,
        std=std,
    ),
])


# Создаём две версии train-части:
#
# 1. с аугментациями для обучения;
# 2. без аугментаций для validation.
#
# Индексы у них одинаковые.

train_base_augmented = datasets.CIFAR10(
    root=DATA_ROOT,
    train=True,
    download=False,
    transform=train_transforms,
)

train_base_evaluation = datasets.CIFAR10(
    root=DATA_ROOT,
    train=True,
    download=False,
    transform=eval_transforms,
)

test_dataset = datasets.CIFAR10(
    root=DATA_ROOT,
    train=False,
    download=False,
    transform=eval_transforms,
)


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

split_generator = torch.Generator()
split_generator.manual_seed(SEED)

indices = torch.randperm(
    len(train_base_augmented),
    generator=split_generator,
).tolist()

val_indices = indices[:VAL_SIZE]
train_indices = indices[VAL_SIZE:]

train_dataset = Subset(
    train_base_augmented,
    train_indices,
)

val_dataset = Subset(
    train_base_evaluation,
    val_indices,
)

print(f"Train samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Test samples: {len(test_dataset)}")


# ============================================================
# DATALOADERS
# ============================================================

loader_generator = torch.Generator()
loader_generator.manual_seed(SEED)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=pin_memory,
    persistent_workers=NUM_WORKERS > 0,
    generator=loader_generator,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=pin_memory,
    persistent_workers=NUM_WORKERS > 0,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=pin_memory,
    persistent_workers=NUM_WORKERS > 0,
)


# Проверяем реальный batch
images, labels = next(iter(train_loader))

print(f"Images shape: {images.shape}")
print(f"Labels shape: {labels.shape}")
print(f"Images dtype: {images.dtype}")
print(f"Labels dtype: {labels.dtype}")


# ============================================================
# LOSS, OPTIMIZER, SCHEDULER
# ============================================================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=LEARNING_RATE,
    momentum=MOMENTUM,
    weight_decay=WEIGHT_DECAY,
    nesterov=True,
)

scheduler = lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=NUM_EPOCHS,
    eta_min=1e-5,
)


# ============================================================
# TRAIN FUNCTION
# ============================================================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> tuple[float, float]:
    model.train()

    loss_sum = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(
            device,
            non_blocking=pin_memory,
        )

        labels = labels.to(
            device,
            non_blocking=pin_memory,
        )

        optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)

        loss_sum += loss.item() * batch_size
        total += batch_size

        predictions = logits.argmax(dim=1)
        correct += (predictions == labels).sum().item()

        if batch_idx % 50 == 0:
            current_accuracy = correct / total

            print(
                f"Epoch {epoch:02d} | "
                f"batch {batch_idx:03d}/{len(loader):03d} | "
                f"loss {loss.item():.4f} | "
                f"running acc {current_accuracy:.2%}"
            )

    epoch_loss = loss_sum / total
    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()

    loss_sum = 0.0
    correct = 0
    total = 0

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(
                device,
                non_blocking=pin_memory,
            )

            labels = labels.to(
                device,
                non_blocking=pin_memory,
            )

            logits = model(images)
            loss = criterion(logits, labels)

            batch_size = labels.size(0)

            loss_sum += loss.item() * batch_size
            total += batch_size

            predictions = logits.argmax(dim=1)
            correct += (predictions == labels).sum().item()

    epoch_loss = loss_sum / total
    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy


# ============================================================
# CHECKPOINT HELPERS
# ============================================================

def get_model_state_dict(model: nn.Module) -> dict:
    if isinstance(model, nn.DataParallel):
        return model.module.state_dict()

    return model.state_dict()


def load_model_state_dict(
    model: nn.Module,
    state_dict: dict,
) -> None:
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict)


# ============================================================
# TRAINING
# ============================================================

best_val_accuracy = 0.0
best_epoch = 0
patience_counter = 0

for epoch in range(1, NUM_EPOCHS + 1):
    current_lr = optimizer.param_groups[0]["lr"]

    print()
    print("=" * 80)
    print(
        f"Epoch {epoch}/{NUM_EPOCHS} | "
        f"learning rate: {current_lr:.6f}"
    )
    print("=" * 80)

    train_loss, train_accuracy = train_one_epoch(
        model=model,
        loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epoch=epoch,
    )

    val_loss, val_accuracy = evaluate(
        model=model,
        loader=val_loader,
        criterion=criterion,
        device=device,
    )

    print(
        f"Epoch {epoch:02d} summary | "
        f"train loss: {train_loss:.4f} | "
        f"train acc: {train_accuracy:.2%} | "
        f"val loss: {val_loss:.4f} | "
        f"val acc: {val_accuracy:.2%}"
    )

    # Scheduler шагает один раз после завершения эпохи.
    scheduler.step()

    improved = val_accuracy > best_val_accuracy + MIN_DELTA

    if improved:
        best_val_accuracy = val_accuracy
        best_epoch = epoch
        patience_counter = 0

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": get_model_state_dict(model),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "class_names": train_base_augmented.classes,
            "config": {
                "num_classes": NUM_CLASSES,
                "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "momentum": MOMENTUM,
                "weight_decay": WEIGHT_DECAY,
            },
        }

        torch.save(
            checkpoint,
            CHECKPOINT_PATH,
        )

        print(
            f"Best model saved: "
            f"val accuracy {best_val_accuracy:.2%}"
        )

    else:
        patience_counter += 1

        print(
            f"No improvement: "
            f"{patience_counter}/{PATIENCE}"
        )

    if patience_counter >= PATIENCE:
        print()
        print(
            f"Early stopping. "
            f"Best epoch: {best_epoch}, "
            f"best validation accuracy: {best_val_accuracy:.2%}"
        )
        break


# ============================================================
# LOAD BEST MODEL
# ============================================================

print()
print(f"Loading best checkpoint: {CHECKPOINT_PATH}")

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device,
)

load_model_state_dict(
    model,
    checkpoint["model_state_dict"],
)

print(
    f"Loaded epoch {checkpoint['epoch']} | "
    f"validation accuracy: {checkpoint['val_accuracy']:.2%}"
)


# ============================================================
# FINAL TEST
# ============================================================

test_loss, test_accuracy = evaluate(
    model=model,
    loader=test_loader,
    criterion=criterion,
    device=device,
)

print()
print("=" * 80)
print(f"Final test loss: {test_loss:.4f}")
print(f"Final test accuracy: {test_accuracy:.2%}")
print("=" * 80)