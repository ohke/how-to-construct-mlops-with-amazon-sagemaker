import click
from sagemaker.analytics import HyperparameterTuningJobAnalytics
from sagemaker.estimator import Estimator, TrainingInput
from sagemaker.session import Session
from sagemaker.tuner import (
    CategoricalParameter,
    ContinuousParameter,
    HyperparameterTuner,
)
from typing import Optional

from utility import ExperimentSetting


@click.command()
@click.option("--image-uri", type=str, envvar="IMAGE_URI")
@click.option("--role", type=str, envvar="ROLE")
@click.option("--job-name", type=str)
@click.option("--max-jobs", type=int)
@click.option("--max-parallel-jobs", type=int)
@click.option("--experiment-name", type=str, envvar="SAGEMAKER_EXPERIMENT_NAME")
@click.option("--component-name", type=str, default="hyperparameter_tuning")
@click.option("--trial-suffix", type=str, default=None)
@click.option("--input-s3-uri", type=str)
@click.option("--output-s3-uri", type=str, default=None)
@click.option(
    "--instance-type",
    type=click.Choice(["ml.c5.xlarge", "ml.g4dn.xlarge"]),
    default="ml.c5.xlarge",
)
@click.option("--use-spot-instances", is_flag=True, default=False)
@click.option("--epochs", type=int, default=2)
def main(
    image_uri: str,
    role: str,
    job_name: str,
    max_jobs: int,
    max_parallel_jobs: int,
    experiment_name: str,
    component_name: str,
    trial_suffix: Optional[str],
    input_s3_uri: str,
    output_s3_uri: Optional[str],
    instance_type: str,
    use_spot_instances: bool,
    epochs: int,
):
    """Select optimal batch size and LR."""
    print("Started hyperparameter tuning.")

    session = Session()

    # 最小化するメトリクス (ここではtest:loss) を学習時に収集する
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
            "epochs": epochs,
            "batch_size": 64,
        },
        metric_definitions=metric_definitions,
        sagemaker_session=session,
    )

    # チューニング対象のハイパパラメータ (ここでは2変数)
    hyperparameter_ranges = {
        "batch_size": CategoricalParameter([32, 64]),
        "lr": ContinuousParameter(0.01, 2.0, scaling_type="Auto"),
    }

    tuner = HyperparameterTuner(
        estimator=estimator,
        objective_type="Minimize",  # or "Maximize"
        objective_metric_name=objective_metric_name,
        hyperparameter_ranges=hyperparameter_ranges,
        metric_definitions=metric_definitions,
        max_jobs=max_jobs,  # 合計試行数
        max_parallel_jobs=max_parallel_jobs,  # 同時実行する試行数
        strategy="Bayesian",  # or "Random"
        early_stopping_type="Auto",  # or "Off"
    )

    inputs = {
        "train": TrainingInput(s3_data=input_s3_uri),
    }

    experiment_setting = ExperimentSetting.new(
        experiment_name=experiment_name,
        trial_suffix=trial_suffix,
    )

    # チューニング開始
    tuner.fit(
        job_name=job_name,
        inputs=inputs,
        experiment_config=experiment_setting.create_experiment_config(component_name),
    )

    # 結果をDataFrameで取得
    result = HyperparameterTuningJobAnalytics(job_name, sagemaker_session=session)
    print(result.dataframe())

    print("Completed.")


if __name__ == "__main__":
    main()
