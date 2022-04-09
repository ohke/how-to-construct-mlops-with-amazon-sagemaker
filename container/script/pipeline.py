import click
from typing import Optional
from sagemaker.debugger import (
    FrameworkProfile,
    ProfilerConfig,
    ProfilerRule,
    rule_configs,
)
from sagemaker.estimator import Estimator
from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.session import Session
from sagemaker.processing import ProcessingInput, ProcessingOutput, Processor
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_experiment_config import PipelineExperimentConfig
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.steps import ProcessingStep, TrainingInput, TrainingStep

from script import ExperimentSetting


@click.command()
@click.option("--image-uri")
@click.option("--role")
@click.option("--experiment-name", default="mnist")
@click.option("--trial-suffix", default=None)
@click.option("--epochs", type=int, default=2)
@click.option("--batch-size", type=int, default=64)
@click.option("--lr", type=float, default=1.0)
@click.option("--model-package-group-name", default="mnist")
def main(
    image_uri: str,
    role: str,
    experiment_name: str,
    trial_suffix: Optional[str],
    epochs: int,
    batch_size: int,
    lr: float,
    model_package_group_name: str,
):
    session = Session()

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
        outputs=[
            ProcessingOutput(
                source="/opt/ml/processing/output/train",
                output_name="preprocess-train",
            ),
        ],
        job_arguments=[
            "--output-train-path",
            "/opt/ml/processing/output/train",
            "--output-test-path",
            "/opt/ml/processing/output/test",
        ],
    )

    train_estimator = Estimator(
        image_uri=image_uri,
        role=role,
        instance_type="ml.c5.xlarge",
        instance_count=1,
        hyperparameters={
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
        },
        metric_definitions=[
            {"Name": "test:loss", "Regex": "test_loss: ([0-9\\.]+)"},
            {"Name": "test:accuracy", "Regex": "test_accuracy: ([0-9\\.]+)"},
        ],
        rules=[
            ProfilerRule.sagemaker(rule_configs.ProfilerReport()),
            ProfilerRule.sagemaker(rule_configs.BatchSize()),
            ProfilerRule.sagemaker(rule_configs.CPUBottleneck()),
            ProfilerRule.sagemaker(rule_configs.GPUMemoryIncrease()),
            ProfilerRule.sagemaker(rule_configs.IOBottleneck()),
            ProfilerRule.sagemaker(rule_configs.LowGPUUtilization()),
            ProfilerRule.sagemaker(rule_configs.OverallSystemUsage()),
            ProfilerRule.sagemaker(rule_configs.StepOutlier()),
        ],
        profiler_config=ProfilerConfig(
            framework_profile_params=FrameworkProfile(start_step=2, num_steps=10)
        ),
        sagemaker_session=session,
    )

    train_step = TrainingStep(
        name="train",
        display_name="Train",
        estimator=train_estimator,
        inputs={
            "train": TrainingInput(
                s3_data=preprocess_step.properties.ProcessingOutputConfig.Outputs[
                    "preprocess-train"
                ].S3Output.S3Uri,
            )
        },
    )

    evaluate_processor = Processor(
        image_uri=image_uri,
        entrypoint=["python", "/opt/program/evaluate.py"],
        role=role,
        instance_count=1,
        instance_type="ml.c5.xlarge",
        sagemaker_session=session,
    )

    evaluate_step = ProcessingStep(
        name="evaluate",
        display_name="Evaluate",
        processor=evaluate_processor,
        inputs=[
            ProcessingInput(
                input_name="input-data",
                source=preprocess_step.properties.ProcessingOutputConfig.Outputs[
                    "preprocess-train"
                ].S3Output.S3Uri,
                destination="/opt/ml/processing/input/data/",
            ),
            ProcessingInput(
                input_name="input-model",
                source=train_step.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/input/model/",
            ),
        ],
        outputs=[
            ProcessingOutput(
                output_name="evaluation",
                source="/opt/ml/processing/output/",
            ),
        ],
        job_arguments=[
            "--input-path",
            "/opt/ml/processing/input/data",
            "--model-path",
            "/opt/ml/processing/input/model/model.tar.gz",
            "--output-path",
            "/opt/ml/processing/output/evaluation.json",
        ],
        property_files=[
            PropertyFile(
                name="EvaluationReport",
                output_name="evaluation",
                path="evaluation.json",
            )
        ],
    )

    register_step = RegisterModel(
        name="register",
        display_name="Register",
        estimator=train_step.estimator,
        model_data=train_step.properties.ModelArtifacts.S3ModelArtifacts,
        content_types=["image/jpeg"],
        response_types=["application/json"],
        inference_instances=["ml.t2.medium"],
        transform_instances=["ml.m5.large"],
        model_package_group_name=model_package_group_name,
        model_metrics=ModelMetrics(
            model_statistics=MetricsSource(
                s3_uri=f"{evaluate_step.arguments['ProcessingOutputConfig']['Outputs'][0]['S3Output']['S3Uri']}/evaluation.json",
                content_type="application/json",
            ),
        ),
        depends_on=[evaluate_step],
    )

    setting = ExperimentSetting.new(experiment_name, trial_suffix)
    pipeline = Pipeline(
        name=f"{experiment_name}-pipeline",
        steps=[preprocess_step, train_step, evaluate_step, register_step],
        pipeline_experiment_config=PipelineExperimentConfig(
            experiment_name=setting.experiment.experiment_name,
            trial_name=setting.trial.trial_name,
        ),
    )

    pipeline.upsert(role_arn=role)

    execution = pipeline.start()


if __name__ == "__main__":
    main()
