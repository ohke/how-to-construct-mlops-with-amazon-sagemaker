import click
import os
import shutil
from glob import glob
from torchvision import datasets


@click.command()
@click.option("--output-train-path", default="/opt/ml/processing/output/train")
@click.option("--output-test-path", default="/opt/ml/processing/output/test")
def main(output_train_path: str, output_test_path: str):
    os.makedirs(output_train_path, exist_ok=True)
    datasets.MNIST(output_train_path, download=True)
    for p in glob(os.path.join(output_train_path, "**/*.gz"), recursive=True):
        os.remove(p)

    src = os.path.join(output_train_path, "MNIST/raw/")
    dst = os.path.join(output_test_path, "MNIST/raw/")
    os.makedirs(dst, exist_ok=True)
    shutil.copyfile(
        os.path.join(src, "t10k-images-idx3-ubyte"),
        os.path.join(dst, "t10k-images-idx3-ubyte"),
    )
    shutil.copyfile(
        os.path.join(src, "t10k-labels-idx1-ubyte"),
        os.path.join(dst, "t10k-labels-idx1-ubyte"),
    )


if __name__ == "__main__":
    main()
