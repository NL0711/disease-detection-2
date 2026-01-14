"""Project pipelines."""

from kedro.pipeline import Pipeline

from disease_detection.pipelines.data_preprocessing.pipeline import create_pipeline as create_preprocess_pipeline
from disease_detection.pipelines.evaluate_model.pipeline import create_pipeline as create_eval_pipeline
from disease_detection.pipelines.resnet50_training.pipeline import create_pipeline as create_res50_pipeline
from disease_detection.pipelines.user_app.pipeline import create_pipeline as create_userapp_pipeline


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """

    preprocess_pipeline = create_preprocess_pipeline()
    # deduplication_pipeline = preprocess_pipeline.only_nodes_with_tags("deduplication")
    augment_pipeline = preprocess_pipeline.only_nodes_with_tags("augment")
    transform_pipeline = preprocess_pipeline.only_nodes_with_tags("transform")

    resnet_pipeline = create_res50_pipeline()
    eval_pipeline = create_eval_pipeline()
    userapp_pipeline = create_userapp_pipeline()

    return {
        "preprocess_pipeline": preprocess_pipeline,
        # "deduplication_pipeline": deduplication_pipeline,
        "augment_pipeline": augment_pipeline,
        "transform_pipeline": transform_pipeline,
        "resnet_pipeline": resnet_pipeline,
        "eval_pipeline": eval_pipeline,
        "userapp_pipeline": userapp_pipeline,
        "__default__": preprocess_pipeline
        # + deduplication_pipeline
        + augment_pipeline
        + transform_pipeline
        + resnet_pipeline
        + eval_pipeline
        + userapp_pipeline,
    }
