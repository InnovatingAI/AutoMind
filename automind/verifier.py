import logging
from abc import ABC, abstractmethod
from typing import Optional, cast

from .backend import query
from .backend.utils import FuncSpec

logger = logging.getLogger("automind")


class Verifier(ABC):
    def __init__(
        self,
        model: str,
        task_desc: str,
        human_guideline: Optional[str] = None,
    ):
        self.model = model
        self.task_desc = task_desc
        self.human_guideline = human_guideline

    @abstractmethod
    def prompt_integration(self):
        pass

    @abstractmethod
    def step(self):
        pass


class SubmissionVerifier(Verifier):
    def __init__(
        self,
        model: str,
        task_desc: str,
        human_guideline: Optional[str] = None,
    ):
        super().__init__(model, task_desc, human_guideline)

    @property
    def _prompt_introduction(self):
        return {
            "Introduction": (
                "You are an expert machine learning engineer attempting a task. "
                "You have written code to solve this task and now need to evaluate the output of the code execution. "
                "You should determine if there were any bugs as well as report the empirical findings."
            )
        }

    @property
    def _verifier_func_spec(self) -> FuncSpec:
        return FuncSpec(
            name="submission_verify",
            json_schema={
                "type": "object",
                "properties": {
                    "is_bug": {
                        "type": "boolean",
                        "description": "true if the output log shows that the execution failed or has some bug, otherwise false.",
                    },
                    "is_overfitting": {
                        "type": "boolean",
                        "description": "true if the output log shows that the model is overfitting or validation metric is much worse than the training metric or validation loss is increasing, otherwise false. ",
                    },
                    "has_csv_submission": {
                        "type": "boolean",
                        "description": "true if the code saves the predictions on the test data in a `submission.csv` file in the `./submission/` directory, otherwise false. "
                        "Note that the file MUST be saved in the ./submission/ directory for this to be evaluated as true, otherwise it should be evaluated as false. "
                        "You can assume the ./submission/ directory exists and is writable.",
                    },
                    "summary": {
                        "type": "string",
                        "description": "write a short summary (2-3 sentences) describing the empirical findings. "
                        "Alternatively mention if there is a bug or the submission.csv was not properly produced. "
                        "You do not need to suggest fixes or improvements.",
                    },
                    "metric": {
                        "type": "number",
                        "description": "If the code ran successfully, report the value of the validation metric. Otherwise, leave it null.",
                    },
                    "lower_is_better": {
                        "type": "boolean",
                        "description": "true if the metric should be minimized (i.e. a lower metric value is better, such as with MSE), false if the metric should be maximized (i.e. a higher metric value is better, such as with accuracy).",
                    },
                },
                "required": [
                    "is_bug",
                    "is_overfitting",
                    "has_csv_submission",
                    "summary",
                    "metric",
                    "lower_is_better",
                ],
            },
            description="Verify the execution output of the written code.",
        )

    def prompt_integration(
        self,
        code: str,
        exec_output: str,
    ):
        prompt = self._prompt_introduction
        prompt["Task Description"] = self.task_desc
        prompt["Code"] = code
        prompt["Execution Output"] = exec_output
        return prompt

    def step(
        self,
        code: str,
        exec_output: str,
    ):
        prompt = self.prompt_integration(code, exec_output)

        response = cast(
            dict,
            query(
                system_message=prompt,
                user_message=None,
                func_spec=self._verifier_func_spec,
                model=self.model,
                temperature=0,
            ),
        )

        # if the metric isn't a float then fill the metric with the worst metric
        if not isinstance(response["metric"], float):
            response["metric"] = None

        logger.info(f"Verifier response: {response}")
        return response
