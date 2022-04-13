from typing import Optional
import click
from sagemaker.estimator import Model

from script import ExperimentSetting


@click.command()
@click.option("--model-s3-uri")
@click.option("--role")
@click.option("--image-uri")
@click.option("--input-s3-uri")
@click.option("--job-name", type=str, default=None)
@click.option("--experiment-name", default="mnist")
@click.option("--trial-suffix", type=str, default=None)
def main(
    model_s3_uri: str,
    role: str,
    image_uri: str,
    input_s3_uri: str,
    job_name: Optional[str],
    experiment_name: str,
    trial_suffix: Optional[str],
):
    model = Model(
        model_data=model_s3_uri,
        role=role,
        image_uri=image_uri,
    )

    transformer = model.transformer(
        instance_count=1,
        instance_type="ml.m5.large",
        strategy="SingleRecord",
        max_concurrent_transforms=1,
        max_payload=1,
    )

    setting = ExperimentSetting.new(experiment_name, trial_suffix)

    transformer.transform(
        data=input_s3_uri,
        data_type="S3Prefix",
        content_type="image/jpeg",
        job_name=job_name,
        experiment_config=setting.create_experiment_config("batch-transform"),
        wait=True,
        logs=True,
    )


if __name__ == "__main__":
    main()
