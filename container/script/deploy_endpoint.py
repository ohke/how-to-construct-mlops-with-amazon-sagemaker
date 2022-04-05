import click
from datetime import datetime
from sagemaker.estimator import Model


@click.command()
@click.option("--model-s3-path")
@click.option("--role")
@click.option("--image-uri")
@click.option("--endpoint-name", default=f"mnist-{datetime.now().strftime('%Y%m%d%H%M%S')}")
def main(model_s3_path: str, role: str, image_uri: str, endpoint_name: str):
    model = Model(
        model_data=model_s3_path,
        role=role,
        image_uri=image_uri,
    )

    model.deploy(
        endpoint_name=endpoint_name,
        initial_instance_count=1,
        instance_type="ml.t2.medium",
        wait=True,
    )


if __name__ == "__main__":
    main()
