import click
from sagemaker.analytics import HyperparameterTuningJobAnalytics
from sagemaker.estimator import Estimator, TrainingInput
from sagemaker.tuner import ContinuousParameter, HyperparameterTuner
from smexperiments.tracker import Tracker
from typing import Optional

from script import ExperimentSetting


@click.command()
@click.option("--image-uri")
@click.option("--role")
@click.option("--job-name")
@click.option("--max-jobs")
@click.option("--max-parallel-jobs")
@click.option("--experiment-name", default="mnist")
@click.option("--trial-suffix", default=None)
@click.option("--input-s3-uri")
@click.option("--output-s3-uri", default=None)
@click.option(
    "--instance-type",
    type=click.Choice(["ml.c5.xlarge", "ml.g4dn.xlarge"]),
    default="ml.c5.xlarge",
)
@click.option("--use-spot-instances", is_flag=True, default=False)
def main(
    image_uri: str,
    role: str,
    job_name: str,
    max_jobs: int,
    max_parallel_jobs: int,
    experiment_name: Optional[str],
    trial_suffix: Optional[str],
    input_s3_uri: str,
    output_s3_uri: Optional[str],
    instance_type: str,
    use_spot_instances: bool,
    epochs: int,
    batch_size: int,
):
    objective_metric_name = "test:loss"
    metric_definitions = [
        {"Name": objective_metric_name, "Regex": "test_loss: ([0-9\\.]+)"},
        {"Name": "test:accuracy", "Regex": "test_accuracy: ([0-9\\.]+)"},
    ]

    estimator = Estimator(
        image_uri=image_uri,
        role=role,
        output_path=output_s3_uri,
        instance_type=instance_type,
        instance_count=1,
        use_spot_instances=use_spot_instances,
        hyperparameters={
            "epochs": 2,
            "batch_size": 64,
        },
        metric_definitions=metric_definitions,
    )

    hyperparameter_ranges = {
        "lr": ContinuousParameter(0.01, 2.0, scaling_type="Auto"),
    }

    tuner = HyperparameterTuner(
        estimator=estimator,
        objective_type="Minimize",
        objective_metric_name=objective_metric_name,
        hyperparameter_ranges=hyperparameter_ranges,
        metric_definitions=metric_definitions,
        max_jobs=max_jobs,
        max_parallel_jobs=max_parallel_jobs,
        strategy="Bayesian",
        early_stopping_type="Auto",
    )

    inputs = {
        "train": TrainingInput(s3_data=input_s3_uri),
    }

    experiment_setting = ExperimentSetting.new(
        experiment_name=experiment_name,
        trial_suffix=trial_suffix,
    )

    tuner.fit(
        job_name=job_name,
        inputs=inputs,
        experiment_config=experiment_setting.create_experiment_config(
            "hyperparameter_tuning"
        ),
    )

    result = HyperparameterTuningJobAnalytics(job_name)

    print(result.dataframe())


if __name__ == "__main__":
    main()
