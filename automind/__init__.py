from dataclasses import dataclass

from .backend import compile_prompt_to_md

from .agent import Agent
from .interpreter import Interpreter
from .trajectory import Trajectory, Node
from omegaconf import OmegaConf
from rich.status import Status
from .utils.config import (
    load_task_desc,
    prep_agent_workspace,
    save_run,
    _load_cfg,
    prep_cfg,
)
from pathlib import Path


@dataclass
class Solution:
    code: str
    valid_metric: float


class Experiment:
    def __init__(self, data_dir: str, goal: str, eval: str | None = None):
        """Initialize a new experiment run.

        Args:
            data_dir (str): Path to the directory containing the data files.
            goal (str): Description of the goal of the task.
            eval (str | None, optional): Optional description of the preferred way for the agent to evaluate its solutions.
        """

        _cfg = _load_cfg(use_cli_args=False)
        _cfg.data_dir = data_dir
        _cfg.goal = goal
        _cfg.eval = eval
        self.cfg = prep_cfg(_cfg)

        self.task_desc = load_task_desc(self.cfg)

        with Status("Preparing agent workspace (copying and extracting files) ..."):
            prep_agent_workspace(self.cfg)

        self.traj = Trajectory()
        self.agent = Agent(
            task_desc=self.task_desc,
            cfg=self.cfg,
            traj=self.traj,
        )
        self.interpreter = Interpreter(
            self.cfg.workspace_dir, **OmegaConf.to_container(self.cfg.exec)  # type: ignore
        )

    def run(self, steps: int) -> Solution:
        for _i in range(steps):
            self.agent.step(exec_callback=self.interpreter.run)
            save_run(self.cfg, self.traj)
        self.interpreter.cleanup_session()

        best_node = self.traj.get_best_node()
        return Solution(code=best_node.code, valid_metric=best_node.metric.value)
