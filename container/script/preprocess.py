from typing import Optional
import click
from datetime import datetime
from sagemaker.processing import ProcessingOutput, Processor
from smexperiments.experiment import Experiment
from smexperiments.trial import Trial
from smexperiments.trial_component import TrialComponent


@click.command()
@click.option("--image-uri")
@click.option("--role")
@click.option("--experiment-name", default="mnist")
@click.option("--trial-suffix", default=None)
@click.option("--instance-type", default="ml.c5.xlarge")
@click.option("--instance-count", default=1)
def main(
    image_uri: str,
    role: str,
    experiment_name: str,
    trial_suffix: Optional[str],
    instance_type: str,
    instance_count: int,
):
    print("Starting preprocess MNIST data.")

    if trial_suffix:
        suffix = trial_suffix
    else:
        suffix = datetime.strftime(datetime.now(), "%Y%m%d%H%M%S")

    try:
        experiment = Experiment.load(experiment_name=experiment_name)
    except:
        experiment = Experiment.create(
            experiment_name=experiment_name, description="MNIST experiment"
        )

    trial_name = f"{experiment.experiment_name}-{suffix}"
    try:
        trial = Trial.load(trial_name=trial_name)
    except:
        trial = Trial.create(
            trial_name=trial_name, experiment_name=experiment.experiment_name
        )

    trial_component_name = (
        f"{trial_name}-preprocess-{datetime.strftime(datetime.now(), '%Y%m%d%H%M%S')}"
    )
    trial_component = TrialComponent.create(trial_component_name=trial_component_name)
    trial.add_trial_component(trial_component)

    trial.add_trial_component(trial_component)

    experiment_config = {
        "ExperimentName": experiment.experiment_name,
        "TrialName": trial.trial_name,
        "TrialComponentDisplayName": trial_component_name,
    }

    outputs = [
        ProcessingOutput(
            source="/opt/ml/processing/output/train",
            output_name="preprocess-train",
        ),
        ProcessingOutput(
            source="/opt/ml/processing/output/test",
            output_name="preprocess-test",
        ),
    ]

    processor = Processor(
        image_uri=image_uri,
        entrypoint=["python", "/opt/program/preprocess.py"],
        role=role,
        instance_count=instance_count,
        instance_type=instance_type,
    )
    processor.run(
        inputs=[],
        arguments=[
            "--output-train-path",
            "/opt/ml/processing/output/train",
            "--output-test-path",
            "/opt/ml/processing/output/test",
        ],
        outputs=outputs,
        experiment_config=experiment_config,
        wait=True,
        logs=True,
    )

    print("Completed preprocess MNIST data.")


if __name__ == "__main__":
    main()
