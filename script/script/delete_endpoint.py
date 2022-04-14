import click
from sagemaker import Session


@click.command()
@click.option("--endpoint-name")
def main(endpoint_name: str):
    session = Session()

    session.delete_endpoint(endpoint_name)


if __name__ == "__main__":
    main()
