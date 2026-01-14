from kedro.pipeline import Pipeline, node
from .nodes import train_resnet

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(
            func=train_resnet,
            inputs=["model_transformed_ds", "params:training"],
            outputs="trained_resnet_model",
            name="resnet_training_node"
        )
    ])