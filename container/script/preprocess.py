import argparse
from datetime import datetime
from sagemaker.processing import ProcessingOutput, Processor
from smexperiments.experiment import Experiment
from smexperiments.trial import Trial
from smexperiments.trial_component import TrialComponent


def main():
    print("Starting preprocess MNIST data.")

    parser = argparse.ArgumentParser()
    parser.add_argument("--image-uri", type=str, required=True)
    parser.add_argument("--role", type=str, required=True)
    parser.add_argument("--experiment-name", type=str, default="mnist")
    parser.add_argument("--trial-suffix", type=str, default=None)
    parser.add_argument("--output-s3-uri", type=str, default=None)
    parser.add_argument("--instance-type", type=str, default="ml.c5.xlarge")
    parser.add_argument("--instance-count", type=int, default=1)
    parser.add_argument("--use-spot-instances", action="store_true", default=False)
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

    trial_component_name = f"{trial_name}-preprocess-{datetime.strftime(datetime.now(), '%Y%m%d%H%M%S')}"
    trial_component = TrialComponent.create(trial_component_name=trial_component_name)
    trial.add_trial_component(trial_component)

    trial.add_trial_component(trial_component)

    experiment_config = {
        "ExperimentName": experiment.experiment_name,
        "TrialName": trial.trial_name,
        "TrialComponentDisplayName": trial_component_name,
    }

    outputs = [
        ProcessingOutput(
            source="/opt/ml/processing/output/original",
            output_name="preprocess-output",
        )
    ]

    processor = Processor(
        image_uri=args.image_uri,
        entrypoint=["python", "/opt/program/preprocess.py"],
        role=args.role,
        instance_count=1,
        instance_type=args.instance_type,
    )
    processor.run(
        inputs=[],
        arguments=["--output-path", "/opt/ml/processing/output/original"],
        outputs=outputs,
        experiment_config=experiment_config,
        wait=True,
        logs=True,
    )

    print("Completed preprocess MNIST data.")


if __name__ == "__main__":
    main()
