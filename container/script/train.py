import argparse
from datetime import datetime
from sagemaker.debugger import (
    FrameworkProfile,
    ProfilerConfig,
    ProfilerRule,
    Rule,
    rule_configs,
)
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from smexperiments.experiment import Experiment
from smexperiments.trial import Trial
from smexperiments.trial_component import TrialComponent


def main():
    print("Starting model training.")

    parser = argparse.ArgumentParser()
    parser.add_argument("--image-uri", type=str, required=True)
    parser.add_argument("--role", type=str, required=True)
    parser.add_argument("--experiment-name", type=str, default="mnist")
    parser.add_argument("--trial-suffix", type=str, default=None)
    parser.add_argument("--input-s3-uri", type=str, required=True)
    parser.add_argument("--output-s3-uri", type=str, default=None)
    parser.add_argument(
        "--instance-type",
        type=str,
        choices=["ml.c5.xlarge", "ml.g4dn.xlarge"],
        default="ml.c5.xlarge",
    )
    parser.add_argument("--instance-count", type=int, default=1)
    parser.add_argument("--use-spot-instances", action="store_true", default=False)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1.0)
    args = parser.parse_args()

    if args.trial_suffix:
        suffix = args.trial_suffix
    else:
        suffix = datetime.strftime(datetime.now(), "%Y%m%d%H%M%S")

    try:
        experiment = Experiment.load(experiment_name=args.experiment_name)
    except:
        experiment = Experiment.create(
            experiment_name=args.experiment_name, description="MNIST experiment"
        )

    trial_name = f"{experiment.experiment_name}-{suffix}"
    try:
        trial = Trial.load(trial_name=trial_name)
    except:
        trial = Trial.create(
            trial_name=trial_name, experiment_name=experiment.experiment_name
        )

    trial_component_name = (
        f"{trial_name}-train-{datetime.strftime(datetime.now(), '%Y%m%d%H%M%S')}"
    )
    trial_component = TrialComponent.create(trial_component_name=trial_component_name)
    trial.add_trial_component(trial_component)

    experiment_config = {
        "ExperimentName": experiment.experiment_name,
        "TrialName": trial.trial_name,
        "TrialComponentDisplayName": trial_component_name,
    }

    inputs = {
        "train": TrainingInput(
            s3_data=args.input_s3_uri, distribution="ShardedByS3Key"
        ),
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
        image_uri=args.image_uri,
        role=args.role,
        output_path=args.output_s3_uri,
        instance_type=args.instance_type,
        instance_count=args.instance_count,
        use_spot_instances=args.use_spot_instances,
        hyperparameters={
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
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
