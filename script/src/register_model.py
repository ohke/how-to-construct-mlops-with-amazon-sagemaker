import click
from sagemaker.estimator import Model
from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.session import Session


@click.command()
@click.option("--role", type=str)
@click.option("--image-uri", type=str)
@click.option("--model-s3-uri", type=str)
@click.option("--evaluation-s3-uri", type=str)
@click.option("--model-package-group-name", type=str, default="mnist")
def main(
    role: str,
    image_uri: str,
    model_s3_uri: str,
    evaluation_s3_uri: str,
    model_package_group_name: str,
):
    """Register the model with its evaluation result."""
    session = Session()

    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri=evaluation_s3_uri,
            content_type="application/json",
        )
    )

    model = Model(
        image_uri=image_uri,
        model_data=model_s3_uri, 
        role=role,
        sagemaker_session=session,
    )

    model.register(
        content_types=["image/jpeg"],
        response_types=["application/json"],
        inference_instances=["ml.t2.medium"],
        transform_instances=["ml.m5.large"],
        model_package_group_name=model_package_group_name,
        model_metrics=model_metrics,
        image_uri=image_uri,
    )

    print("Completed.")


if __name__ == "__main__":
    main()
