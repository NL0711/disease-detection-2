from kedro.pipeline import Pipeline, node
from .nodes import predict_on_test_images

def create_userapp_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=predict_on_test_images,
                inputs=[
                    "trained_resnet_model",
                    "test_images",
                    "class_names"
                ],
                outputs="predictions",
                name="predict_test_images",
            )
        ]
    )