import click
from sagemaker import Session


@click.command()
@click.option("--endpoint-name", type=str)
def main(endpoint_name: str):
    """Delete the SageMaker endpoint."""
    session = Session()

    session.delete_endpoint(endpoint_name)

    print("Completed.")


if __name__ == "__main__":
    main()
