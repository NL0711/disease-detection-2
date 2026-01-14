import torch
import mlflow
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import torch
from kedro.io import DataCatalog

def evaluate_model_node(
    trained_resnet_model,
    model_transformed_ds,
    val_folds
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = trained_resnet_model.to(device)
    model.eval()

    all_y_true = []
    all_y_pred = []

    train_folds, val_folds = model_transformed_ds
    x_val_list, y_val_list = [],[]

    for fold_idx, ((X_train, y_train), (X_val, y_val)) in enumerate(zip(train_folds, val_folds)):
      x_val_list.append(X_val)
      y_val_list.append(y_val)

    with torch.no_grad():
      outputs = model(X_val)
      preds = outputs.argmax(dim=1)

      all_y_true.append(y_val.cpu())
      all_y_pred.append(preds.cpu())

    return torch.cat(all_y_true), torch.cat(all_y_pred)


def log_confusion_matrix(
    y_true,
    y_pred,
    class_names,
    artifact_name="confusion_matrix.png",
):
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.numpy()

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(cm)

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center")

    mlflow.log_figure(fig, artifact_name)
    plt.close(fig)

    return fig
    