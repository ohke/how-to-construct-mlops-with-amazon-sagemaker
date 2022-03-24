from sagemaker.estimator import Estimator


def main():
    print("Starting model training.")

    image = "amazon-sagemaker-sandbox-model"
    dummy_role = "arn:aws:iam::111111111111:role/service-role/AmazonSageMaker-ExecutionRole-20200101T000001"

    inputs = {"train": "file://./data/input"}
    output_path = "file://./data/output"

    estimator = Estimator(
        image_uri=image,
        role=dummy_role,
        instance_count=1,
        instance_type="local",
        output_path=output_path,
    )
    estimator.fit(inputs, logs=True)

    print("Completed model training.")


if __name__ == "__main__":
    main()
