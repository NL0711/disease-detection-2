import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models

import mlflow
import mlflow.pytorch

def get_model(num_classes: int, device: str):
    device = torch.device(device)
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    # Replace final fully connected layer
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    # Freeze all layers except the final layer
    for param in model.parameters():
        param.requires_grad = False
        
    model.fc.requires_grad_(True)
    return model.to(device)

def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion,
    device: torch.device
):
    model.train()
    running_loss = 0
    running_correct = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad() # Clear gradients
        outputs = model(images) # Forward pass 

        loss = criterion(outputs, labels)  # Calculate loss / get predicted class
        loss.backward()
        optimizer.step()  # Update weights

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        running_correct += (preds == labels).sum().item()

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc  = running_correct / len(train_loader.dataset)

    return epoch_loss, epoch_acc

def evaluate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion, device:
    torch.device
):
    model.eval()
    val_loss = 0
    val_correct = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            loss = criterion(outputs, labels) 
            val_loss += loss.item() * images.size(0)

            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()
        epoch_loss = val_loss / len(val_loader.dataset)
        epoch_acc = val_correct / len(val_loader.dataset)

    return epoch_loss, epoch_acc

def train_resnet(model_transformed_ds, training):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_folds, val_folds = model_transformed_ds

    mlflow.log_params({
        "model": "resnet18",
        "batch_size": training["batch_size"],
        "learning_rate": training["learning_rate"],
        "num_epochs": training["num_epochs"],
    })

    final_model = None

    for fold_idx, ((X_train, y_train), (X_val, y_val)) in enumerate(
        zip(train_folds, val_folds)
    ):
        model = get_model(training["num_classes"], device)

        train_loader = DataLoader(
            torch.utils.data.TensorDataset(X_train, y_train),
            batch_size=training["batch_size"],
            shuffle=True,
        )

        val_loader = DataLoader(
            torch.utils.data.TensorDataset(X_val, y_val),
            batch_size=training["batch_size"],
            shuffle=False,
        )

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=training["learning_rate"],
        )

        for epoch in range(training["num_epochs"]):
            train_epoch(model, train_loader, optimizer, criterion, device)
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        mlflow.log_metric(f"fold_{fold_idx}_val_acc", val_acc)
        mlflow.log_metric(f"fold_{fold_idx}_val_loss", val_loss)
        final_model = model

    mlflow.pytorch.log_model(final_model, artifact_path="model")
    return final_model