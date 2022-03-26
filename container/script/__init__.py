from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from smexperiments.experiment import Experiment
from smexperiments.trial import Trial
from smexperiments.trial_component import TrialComponent
from typing import Optional


@dataclass
class ExperimentSetting:
    experiment: Experiment
    trial: Trial
    trial_component: TrialComponent

    def new(
        experiment_name: str, trial_suffix: Optional[str] = None
    ) -> ExperimentSetting:
        if trial_suffix:
            suffix = trial_suffix
        else:
            suffix = datetime.strftime(datetime.now(), "%Y%m%d%H%M%S")

        try:
            experiment = Experiment.load(experiment_name=experiment_name)
        except:
            experiment = Experiment.create(
                experiment_name=experiment_name, description="MNIST experiment"
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
        trial_component = TrialComponent.create(
            trial_component_name=trial_component_name
        )
        trial.add_trial_component(trial_component)

        return ExperimentSetting(experiment, trial, trial_component)

    def create_experiment_config(self) -> dict:
        return {
            "ExperimentName": self.experiment.experiment_name,
            "TrialName": self.trial.trial_name,
            "TrialComponentDisplayName": self.trial_component.trial_component_name,
        }
