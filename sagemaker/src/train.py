import click
from typing import Optional
from sagemaker.debugger import (
    FrameworkProfile,
    ProfilerConfig,
    ProfilerRule,
    rule_configs,
)
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput

from utility import ExperimentSetting


@click.command()
@click.option("--image-uri", type=str, required=True)
@click.option("--role", type=str, required=True)
@click.option("--experiment-name", type=str, default="mnist")
@click.option("--component-name", type=str, default="train")
@click.option("--trial-suffix", type=str, default=None)
@click.option("--input-s3-uri", type=str, required=True)
@click.option("--output-s3-uri", type=str, default=None)
@click.option(
    "--instance-type",
    type=click.Choice(["ml.c5.xlarge", "ml.g4dn.xlarge", "local"]),
    default="ml.c5.xlarge",
)
@click.option("--instance-count", type=int, default=1)
@click.option("--use-spot-instances", is_flag=True, show_default=True, default=False)
@click.option("--checkpoint-s3-uri", type=str, default=None)
@click.option("--epochs", type=int, default=2)
@click.option("--batch-size", type=int, default=64)
@click.option("--lr", type=float, default=1.0)
def main(
    image_uri: str,
    role: str,
    experiment_name: str,
    component_name: str,
    trial_suffix: Optional[str],
    input_s3_uri: str,
    output_s3_uri: Optional[str],
    instance_type: str,
    instance_count: int,
    use_spot_instances: bool,
    checkpoint_s3_uri: Optional[str],
    epochs: int,
    batch_size: str,
    lr: float,
):
    """Train model with MNIST dataset."""
    print("Started model training.")

    local_mode = instance_type == "local"

    if local_mode:
        experiment_config = None

        inputs = {"train": input_s3_uri}

        rules = None
        profiler_config = None
    else:
        setting = ExperimentSetting.new(experiment_name, trial_suffix)
        experiment_config = setting.create_experiment_config(component_name)

        inputs = {
            "train": TrainingInput(s3_data=input_s3_uri, distribution="ShardedByS3Key")
        }

        rules = [
            ProfilerRule.sagemaker(rule_configs.ProfilerReport()),
            ProfilerRule.sagemaker(rule_configs.BatchSize()),
            ProfilerRule.sagemaker(rule_configs.CPUBottleneck()),
            ProfilerRule.sagemaker(rule_configs.GPUMemoryIncrease()),
            ProfilerRule.sagemaker(rule_configs.IOBottleneck()),
            ProfilerRule.sagemaker(rule_configs.LowGPUUtilization()),
            ProfilerRule.sagemaker(rule_configs.OverallSystemUsage()),
            ProfilerRule.sagemaker(rule_configs.StepOutlier()),
        ]

        profiler_config = ProfilerConfig(
            framework_profile_params=FrameworkProfile(start_step=2, num_steps=10)
        )

    estimator = Estimator(
        image_uri=image_uri,
        role=role,
        output_path=output_s3_uri,
        instance_type=instance_type,
        instance_count=instance_count,
        use_spot_instances=use_spot_instances,
        checkpoint_s3_uri=checkpoint_s3_uri,
        hyperparameters={
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
        },
        metric_definitions=[
            {"Name": "test:loss", "Regex": "test_loss: ([0-9\\.]+)"},
            {"Name": "test:accuracy", "Regex": "test_accuracy: ([0-9\\.]+)"},
        ],
        rules=rules,
        profiler_config=profiler_config,
    )
    estimator.fit(inputs, experiment_config=experiment_config, logs=True)

    print("Completed model training.")


if __name__ == "__main__":
    main()