import click
from sagemaker.session import Session


@click.command()
@click.option("--endpoint-name", type=str)
@click.option("--endpoint-config-name", type=str)
@click.option("--model-name", type=str)
@click.option("--initial-instance-count", type=int, default=1)
@click.option("--instance-type", type=str, default="ml.t2.medium")
def main(
    endpoint_name: str,
    endpoint_config_name: str,
    model_name: str,
    initial_instance_count: int,
    instance_type: str,
):
    """Create or update SageMaker endpoint."""
    session = Session()

    session.create_endpoint_config(
        name=endpoint_config_name,
        model_name=model_name,
        initial_instance_count=initial_instance_count,
        instance_type=instance_type,
    )

    try:
        session.create_endpoint(
            endpoint_name=endpoint_name,
            config_name=endpoint_config_name,
            wait=True,
        )
    except:
        session.update_endpoint(
            endpoint_name=endpoint_name,
            endpoint_config_name=endpoint_config_name,
            wait=True,
        )

    print("Completed.")


if __name__ == "__main__":
    main()
