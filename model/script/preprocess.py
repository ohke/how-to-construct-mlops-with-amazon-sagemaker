from sagemaker.processing import ProcessingOutput, Processor
from sagemaker.local import LocalSession


def main():
    print("Starting preprocess MNIST data.")

    image = "amazon-sagemaker-sandbox-model"
    dummy_role = "arn:aws:iam::111111111111:role/service-role/AmazonSageMaker-ExecutionRole-20200101T000001"

    outputs = [
        ProcessingOutput(
            source="/opt/ml/processing/original",
            output_name="preprocess-output",
        )
    ]

    processor = Processor(
        image_uri=image,
        entrypoint=["python", "/opt/program/preprocess.py"],
        role=dummy_role,
        instance_count=1,
        instance_type="local",
    )
    processor.run(
        inputs=[],
        arguments=["--output-path", "/opt/ml/processing/original"],
        outputs=outputs,
        wait=True,
        logs=True,
    )

    print("Completed preprocess MNIST data.")


if __name__ == "__main__":
    main()
