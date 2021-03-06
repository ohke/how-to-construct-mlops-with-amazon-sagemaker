import json
import click
from sagemaker.debugger import (
    FrameworkProfile,
    ProfilerConfig,
    ProfilerRule,
    rule_configs,
)
from sagemaker.estimator import Estimator
from sagemaker.model import Model
from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.session import Session
from sagemaker.processing import ProcessingInput, ProcessingOutput, Processor
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.functions import Join, JsonGet
from sagemaker.workflow.parameters import ParameterFloat, ParameterInteger
from sagemaker.workflow.pipeline import ExecutionVariables, Pipeline
from sagemaker.workflow.pipeline_experiment_config import PipelineExperimentConfig
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.steps import (
    CreateModelInput,
    CreateModelStep,
    ProcessingStep,
    TrainingInput,
    TrainingStep,
)


@click.command()
@click.option("--pipeline-name", type=str)
@click.option("--image-uri", type=str, envvar="IMAGE_URI")
@click.option("--role", type=str, envvar="ROLE")
@click.option(
    "--model-package-group-name", type=str, envvar="SAGEMAKER_MODEL_PACKAGE_GROUP_NAME"
)
@click.option("--epochs", type=int, default=2)
@click.option("--batch-size", type=int, default=64)
@click.option("--lr", type=float, default=1.0)
@click.option("--min-accuracy", type=float, default=0.98)
@click.option("--start", is_flag=True, show_default=True, default=False)
def main(
    pipeline_name: str,
    image_uri: str,
    role: str,
    model_package_group_name: str,
    epochs: int,
    batch_size: int,
    lr: float,
    min_accuracy: float,
    start: bool,
):
    """Create/update SageMaker pipeline."""
    print("Started create/update pipeline.")

    session = Session()

    train_epochs = ParameterInteger(name="Epochs", default_value=epochs)
    train_batch_size = ParameterInteger(name="BatchSize", default_value=batch_size)
    train_lr = ParameterFloat(name="LR", default_value=lr)

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
                source="/opt/ml/processing/output",
                output_name="preprocess",
            ),
        ],
        job_arguments=[
            "--output-path",
            "/opt/ml/processing/output",
        ],
    )

    train_estimator = Estimator(
        image_uri=image_uri,
        role=role,
        instance_type="ml.c5.xlarge",
        instance_count=1,
        hyperparameters={
            "epochs": train_epochs,
            "batch_size": train_batch_size,
            "lr": train_lr,
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
                    "preprocess"
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

    evaluation_report_property_file = PropertyFile(
        name="EvaluationReport",
        output_name="evaluation",
        path="evaluation.json",
    )

    evaluate_step = ProcessingStep(
        name="evaluate",
        display_name="Evaluate",
        processor=evaluate_processor,
        inputs=[
            ProcessingInput(
                input_name="input-data",
                source=preprocess_step.properties.ProcessingOutputConfig.Outputs[
                    "preprocess"
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
        property_files=[evaluation_report_property_file],
    )

    model = Model(
        image_uri=image_uri,
        model_data=train_step.properties.ModelArtifacts.S3ModelArtifacts,
        role=role,
        sagemaker_session=session,
    )

    register_step = RegisterModel(
        name="register",
        display_name="Register",
        estimator=train_step.estimator,
        model=model,
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
    )

    create_step = CreateModelStep(
        name="create",
        display_name="Create",
        model=model,
        inputs=CreateModelInput(instance_type="ml.t2.medium"),
    )

    min_accuracy_condition = ConditionGreaterThanOrEqualTo(
        left=JsonGet(
            step_name=evaluate_step.name,
            property_file=evaluation_report_property_file,
            json_path="metrics.accuracy.value",
        ),
        right=min_accuracy,
    )

    min_accuracy_condition_step = ConditionStep(
        name="min_accuracy_condition",
        display_name="MinAccuracyCondition",
        conditions=[min_accuracy_condition],
        if_steps=[register_step, create_step],
        else_steps=[],
    )

    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[train_epochs, train_batch_size, train_lr],
        steps=[preprocess_step, train_step, evaluate_step, min_accuracy_condition_step],
        pipeline_experiment_config=PipelineExperimentConfig(
            experiment_name=ExecutionVariables.PIPELINE_NAME,
            trial_name=Join(
                on="-",
                values=[
                    ExecutionVariables.PIPELINE_NAME,
                    ExecutionVariables.PIPELINE_EXECUTION_ID,
                ],
            ),
        ),
        sagemaker_session=session,
    )

    assert json.loads(pipeline.definition())

    pipeline.upsert(role_arn=role)

    if start:
        print("Running.")
        pipeline.start(
            parameters=dict(
                Epochs=epochs,
                BatchSize=batch_size,
                LR=lr,
            )
        )

    print("Completed.")


if __name__ == "__main__":
    main()
