from typing import Optional
import click
from sagemaker.processing import ProcessingInput, ProcessingOutput, Processor
from smexperiments.tracker import Tracker

from script import ExperimentSetting


@click.command()
@click.option("--image-uri")
@click.option("--role")
@click.option("--input-s3-uri")
@click.option("--model-s3-uri")
@click.option("--instance-type", default="ml.c5.xlarge")
@click.option("--instance-count", default=1)
@click.option("--experiment-name", default="mnist")
@click.option("--trial-suffix", default=None)
def main(
    image_uri: str,
    role: str,
    input_s3_uri: str,
    model_s3_uri: str,
    instance_type: str,
    instance_count: int,
    experiment_name: Optional[str],
    trial_suffix: Optional[str],
):
    processor = Processor(
        image_uri=image_uri,
        entrypoint=["python", "/opt/program/evaluate.py"],
        role=role,
        instance_count=instance_count,
        instance_type=instance_type,
    )

    inputs = [
        ProcessingInput(
            input_name="input-data",
            source=input_s3_uri,
            destination="/opt/ml/processing/input/data/",
        ),
        ProcessingInput(
            input_name="input-model",
            source=model_s3_uri,
            destination="/opt/ml/processing/input/model/",
        ),
    ]

    outputs = [
        ProcessingOutput(
            output_name="evaluation",
            source="/opt/ml/processing/output/",
        ),
    ]

    experiment_setting = ExperimentSetting.new(
        experiment_name=experiment_name,
        trial_suffix=trial_suffix,
    )

    processor.run(
        inputs=inputs,
        arguments=[
            "--input-path",
            "/opt/ml/processing/input/data",
            "--model-path",
            "/opt/ml/processing/input/model/model.tar.gz",
            "--output-path",
            "/opt/ml/processing/output/evaluation.json",
        ],
        outputs=outputs,
        experiment_config=experiment_setting.create_experiment_config("evaluate"),
        wait=True,
        logs=True,
    )


if __name__ == "__main__":
    main()
