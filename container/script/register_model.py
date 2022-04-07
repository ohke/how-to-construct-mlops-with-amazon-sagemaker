import click
from sagemaker.estimator import Model
from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.session import Session


@click.command()
@click.option("--role")
@click.option("--image-uri")
@click.option("--model-s3-uri")
@click.option("--evaluation-s3-uri")
@click.option("--model-package-group-name", default="mnist")
def main(
    role: str,
    image_uri: str,
    model_s3_uri: str,
    evaluation_s3_uri: str,
    model_package_group_name: str,
):
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


if __name__ == "__main__":
    main()
