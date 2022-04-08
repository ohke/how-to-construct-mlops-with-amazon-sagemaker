import click
from typing import Optional
from sagemaker.session import Session
from sagemaker.processing import ProcessingInput, ProcessingOutput, Processor
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_experiment_config import PipelineExperimentConfig
from sagemaker.workflow.steps import ProcessingStep, TrainingStep

from script import ExperimentSetting


@click.command()
@click.option("--image-uri")
@click.option("--role")
@click.option("--experiment-name", default="mnist")
@click.option("--trial-suffix", default=None)
def main(image_uri: str, role: str, experiment_name: str, trial_suffix: Optional[str]):
    session = Session()

    preprocess_outputs = [
        ProcessingOutput(
            source="/opt/ml/processing/output/train",
            output_name="preprocess-train",
        ),
    ]

    preprocess_processor = Processor(
        image_uri=image_uri,
        entrypoint=["python", "/opt/program/preprocess.py"],
        role=role,
        instance_count=1,
        instance_type="ml.c5.xlarge",
        sagemaker_session=session,
    )

    preprocess_step = ProcessingStep(
        name="preprocess",
        display_name="Preprocess",
        processor=preprocess_processor,
        outputs=preprocess_outputs,
        job_arguments=[
            "--output-train-path",
            "/opt/ml/processing/output/train",
            "--output-test-path",
            "/opt/ml/processing/output/test",
        ],
    )

    setting = ExperimentSetting.new(experiment_name, trial_suffix)
    pipeline = Pipeline(
        name=f"{experiment_name}-pipeline",
        steps=[preprocess_step],
        pipeline_experiment_config=PipelineExperimentConfig(
            experiment_name=setting.experiment.experiment_name,
            trial_name=setting.trial.trial_name,
        ),
    )

    pipeline.create(role_arn=role)

    execution = pipeline.start()


if __name__ == "__main__":
    main()
