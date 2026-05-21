"""Base class for train-and-compare experiments on HybridMTLNet."""
import copy
import json
from abc import ABC, abstractmethod
from pathlib import Path

from src.evaluation.evaluator import Evaluator
from src.training.trainer import Trainer


class Experiment(ABC):
    """Trains the same network under several configurations and compares them.

    Subclasses define `configs()` (the list of runs) and `make_cfg()` (how each
    run maps to a TrainConfig). Shared train/eval/save/print logic lives here.
    """

    results_filename: str = "experiment_results.json"

    def __init__(self, data_cfg, model_cfg, train_cfg):
        self.dcfg = data_cfg
        self.mcfg = model_cfg
        self.tcfg = train_cfg
        self.evaluator = Evaluator(data_cfg, model_cfg, train_cfg)

    # ---- subclass hooks ----------------------------------------------

    @abstractmethod
    def configs(self) -> list[dict]:
        """Return the list of run definitions."""

    @abstractmethod
    def make_cfg(self, run: dict):
        """Map a run definition to a TrainConfig (copy of self.tcfg)."""

    @abstractmethod
    def record(self, run: dict, miou: float, pck: float) -> dict:
        """Build the result entry stored for a run."""

    @abstractmethod
    def print_table(self, results: dict) -> None:
        """Pretty-print the comparison."""

    def task_of(self, run: dict) -> str:
        return run.get("task", "joint")

    # ---- shared machinery --------------------------------------------

    def _base_cfg(self, run: dict):
        return copy.copy(self.tcfg)

    @property
    def results_path(self) -> Path:
        return Path("models") / self.results_filename

    def run_all(self) -> dict:
        """Train (if needed) and evaluate every configuration.

        Skips runs already recorded in the results file, and reuses an existing
        checkpoint when present (evaluates without retraining).
        """
        results = self._load_results()
        for run in self.configs():
            if run["name"] in results:
                print(f"\nSkipping {run['name']} — already done. Use eval-only to re-evaluate.")
                continue
            print(f"\n{'='*65}\n  Experiment: {run.get('label', run['name'])}\n{'='*65}")
            cfg = self.make_cfg(run)
            weights = Evaluator.resolve_weights(cfg.output_dir)
            if weights is None:
                Trainer(self.dcfg, self.mcfg, cfg).run()
                weights = Evaluator.resolve_weights(cfg.output_dir)
            else:
                print(f"Reusing existing checkpoint: {weights}")
            miou, pck = self.evaluator.evaluate(str(weights), self.task_of(run))
            results[run["name"]] = self.record(run, miou, pck)
            self._save(results)
        return results

    def eval_only(self) -> dict:
        results = {}
        for run in self.configs():
            cfg = self.make_cfg(run)
            weights = Evaluator.resolve_weights(cfg.output_dir)
            if weights is None:
                print(f"No weights for {run['name']} — skipping.")
                continue
            miou, pck = self.evaluator.evaluate(str(weights), self.task_of(run))
            results[run["name"]] = self.record(run, miou, pck)
        self._save(results)
        return results

    def _eval(self, cfg, run) -> tuple[float, float]:
        weights = Evaluator.resolve_weights(cfg.output_dir)
        return self.evaluator.evaluate(str(weights), self.task_of(run))

    def _load_results(self) -> dict:
        if self.results_path.exists():
            with open(self.results_path) as f:
                return json.load(f)
        return {}

    def _save(self, results: dict) -> None:
        self.results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.results_path, "w") as f:
            json.dump(results, f, indent=4)
        print(f"Results saved -> {self.results_path}")
