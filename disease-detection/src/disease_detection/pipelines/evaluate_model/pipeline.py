from kedro.pipeline import Pipeline, node
from .nodes import evaluate_model_node, log_confusion_matrix


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(
            func=evaluate_model_node,
            inputs=["trained_resnet_model", "model_transformed_ds", "val_folds"],
            outputs=["y_true", "y_pred"],
            name="evaluate_model_node",
        ),
        node(
            func=log_confusion_matrix,
            inputs=["y_true", "y_pred", "class_names"],
            outputs="visualize",
            name="log_confusion_matrix_node",
        ),
    ])