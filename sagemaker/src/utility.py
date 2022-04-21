from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from smexperiments.experiment import Experiment
from smexperiments.trial import Trial
from typing import Optional


@dataclass
class ExperimentSetting:
    experiment: Experiment
    trial: Trial

    def new(
        experiment_name: str, trial_suffix: Optional[str] = None
    ) -> ExperimentSetting:
        if trial_suffix:
            suffix = trial_suffix
        else:
            suffix = datetime.strftime(datetime.now(), "%Y%m%d%H%M%S")

        try:
            experiment = Experiment.load(experiment_name=experiment_name)
        except Exception:
            experiment = Experiment.create(
                experiment_name=experiment_name, description="MNIST experiment"
            )

        trial_name = f"{experiment.experiment_name}-{suffix}"
        try:
            trial = Trial.load(trial_name=trial_name)
        except Exception:
            trial = Trial.create(
                trial_name=trial_name, experiment_name=experiment.experiment_name
            )

        return ExperimentSetting(experiment, trial)

    def create_experiment_config(self, component_name: str) -> dict:
        return {
            "ExperimentName": self.experiment.experiment_name,
            "TrialName": self.trial.trial_name,
            "TrialComponentDisplayName": component_name,
        }
