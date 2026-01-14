from kedro.pipeline import Node, Pipeline  
from .nodes.split_data import split_data_node
from .nodes.preprocessing import preprocessing_node
from .nodes.preprocess_train import preprocess_train_node
from .nodes.augmentation import augmentation_node
from .nodes.model_transform import model_transform_node
# from .nodes.remove_duplicates import fiftyone_similarity_node

from torchvision.models import mobilenet_v2
import torch.nn as nn

def load_embedding_model():
    model = mobilenet_v2(weights="IMAGENET1K_V1")
    model.classifier = nn.Identity()
    model.eval()
    return model


def create_pipeline(**kwargs) -> Pipeline:

    pipeline_preprocess = Pipeline(
        [
            Node(
                func=preprocessing_node,
                inputs="params:data",
                outputs="preprocessed_ds",
                name="preprocessing_node",
                tags=["preprocessing"],
            ),
        ]
    )

    pipeline_augment = Pipeline(

        [
            #returns indices of train and val folds
            Node(
                func=split_data_node,
                inputs=["preprocessed_ds", "params:data", "params:k_fold"],
                outputs=["train_folds", "val_folds", "class_names"],
                name="split_data_node",
                tags=["augment"],
            ), 
            Node(
                func=preprocess_train_node,
                inputs=["preprocessed_ds", "train_folds", "params:preprocess"],
                outputs=["train_preprocessed_ds", "rejected_ds"],
                name="train_preprocess_node",
                tags=["augment"],
            ),
            Node(
                func=augmentation_node,
                inputs=["train_preprocessed_ds", "params:data"],
                outputs="train_augmented_ds",
                name="train_augmentation_node",
                tags=["augment"],
            ),
        ]
    )

    # pipeline_similarity = Pipeline(
    #     [
    #         Node(
    #             func=fiftyone_similarity_node,
    #             inputs=["train_augmented_ds" , "params:embeddings"],
    #             outputs="deduplicated_ds",
    #             name="fiftyone_similarity_node",
    #             tags=["deduplication"],
    #         ),
    #     ]
    # )

    pipeline_transform = Pipeline(
        [
            Node(
                func=model_transform_node,
                inputs=["train_augmented_ds", "preprocessed_ds", "train_folds", "val_folds"],
                outputs="model_transformed_ds",
                name="model_transform_node",
                tags=["transform"],
            )
        ]
    )

    return pipeline_preprocess + pipeline_augment + pipeline_transform