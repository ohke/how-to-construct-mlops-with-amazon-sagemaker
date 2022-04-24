from typing import Optional
import click
from sagemaker.estimator import Model
from sagemaker.session import Session

from utility import ExperimentSetting


@click.command()
@click.option("--model-s3-uri", type=str)
@click.option("--role", type=str, envvar="ROLE")
@click.option("--image-uri", type=str, envvar="IMAGE_URI")
@click.option("--input-s3-uri", type=str)
@click.option("--job-name", type=str, default=None)
@click.option("--instance-count", type=int, default=1)
@click.option("--instance-type", type=str, default="ml.m5.large")
@click.option("--experiment-name", type=str, envvar="SAGEMAKER_EXPERIMENT_NAME")
@click.option("--component-name", type=str, default="batch-transform")
@click.option("--trial-suffix", type=str, default=None)
def main(
    model_s3_uri: str,
    role: str,
    image_uri: str,
    input_s3_uri: str,
    job_name: Optional[str],
    instance_count: int,
    instance_type: str,
    experiment_name: str,
    component_name: str,
    trial_suffix: Optional[str],
):
    """Run SageMaker batch transform with the model."""
    session = Session()

    model = Model(
        model_data=model_s3_uri,
        role=role,
        image_uri=image_uri,
        sagemaker_session=session,
    )

    transformer = model.transformer(
        instance_count=instance_count,
        instance_type=instance_type,
        strategy="SingleRecord",  # 1リクエストに含むレコード数 (複数の場合は "MultiRecord")
        max_concurrent_transforms=1,  # 同時HTTPリクエスト数
        max_payload=1,  # 1リクエストのペイロードサイズ (MB単位)
    )

    setting = ExperimentSetting.new(experiment_name, trial_suffix)

    transformer.transform(
        data=input_s3_uri,
        data_type="S3Prefix",
        content_type="image/jpeg",
        job_name=job_name,
        experiment_config=setting.create_experiment_config(component_name),
        wait=True,
        logs=True,
    )

    print("Completed.")


if __name__ == "__main__":
    main()
