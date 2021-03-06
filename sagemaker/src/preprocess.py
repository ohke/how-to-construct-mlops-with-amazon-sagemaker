from typing import Optional
import click
from sagemaker.processing import ProcessingOutput, Processor
from sagemaker.session import Session

from utility import ExperimentSetting


@click.command()
@click.option("--image-uri", type=str, envvar="IMAGE_URI")
@click.option("--role", type=str, envvar="ROLE")
@click.option("--experiment-name", type=str, envvar="SAGEMAKER_EXPERIMENT_NAME")
@click.option("--component-name", type=str, default="preprocess")
@click.option("--trial-suffix", type=str, default=None)
@click.option("--instance-type", type=str, default="ml.c5.xlarge")
@click.option("--instance-count", type=int, default=1)
def main(
    image_uri: str,
    role: str,
    experiment_name: str,
    component_name: str,
    trial_suffix: Optional[str],
    instance_type: str,
    instance_count: int,
):
    """Preprocess MNIST data."""
    print("Started MNIST data preprocessing.")

    session = Session()

    setting = ExperimentSetting.new(experiment_name, trial_suffix)

    outputs = [
        ProcessingOutput(
            source="/opt/ml/processing/output",
            output_name="preprocess",
        ),
    ]

    processor = Processor(
        image_uri=image_uri,
        entrypoint=["python", "/opt/program/preprocess.py"],
        role=role,
        instance_count=instance_count,
        instance_type=instance_type,
        sagemaker_session=session,
    )
    processor.run(
        inputs=[],
        arguments=["--output-path", "/opt/ml/processing/output"],
        outputs=outputs,
        experiment_config=setting.create_experiment_config(component_name),
        wait=True,
        logs=True,
    )

    print("Completed  MNIST data preprocessing.")


if __name__ == "__main__":
    main()
