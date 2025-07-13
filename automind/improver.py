import random
import logging
from abc import ABC, abstractmethod
from typing import Optional
import re

from .backend import query
from .utils.response import extract_xml, clean_string

logger = logging.getLogger("automind")


def format_time(time_in_sec: int):
    return f"{time_in_sec // 3600}hrs {(time_in_sec % 3600) // 60}mins {time_in_sec % 60}secs"


class Improver(ABC):
    def __init__(
        self,
        model: str,
        task_desc: str,
        memory: Optional[str] = None,
        human_guideline: Optional[str] = None,
    ):
        self.model = model
        self.task_desc = task_desc
        self.memory = memory
        self.human_guideline = human_guideline

    @property
    def _prompt_environment(self):
        pkgs = [
            "bayesian-optimization==1.5.1",
            "lightgbm==4.5.0",
            "matplotlib==3.9.2",
            "numpy==1.26.2",
            "optuna==4.0.0",
            "pandas==2.1.4",
            "scikit-learn==1.2.2",
            "scipy==1.11.4",
            "seaborn==0.13.2",
            "statsmodels==0.14.4",
            "timm==0.9.7",
            "torch==2.2.0",
            "torchvision==0.17.0",
            "torch-geometric==2.6.1",
            "transformers==4.44.2",
            "xgboost==2.1.3",
        ]
        random.shuffle(pkgs)
        pkg_str = ", ".join([f"`{p}`" for p in pkgs])

        env_prompt = {
            "Installed Packages": (
                f"Your solution can use any relevant machine learning packages such as: {pkg_str}. "
                "Feel free to use any other packages too (all packages are already installed!). "
                "For neural networks please use PyTorch because of the unavailability of TensorFlow in the environment."
            )
        }
        return env_prompt

    @property
    def _prompt_resp_fmt(self):
        return {
            "Response format": (
                "First, provide a brief explanation of your reasoning for the proposed improvement to the previous plan (wrapped in <think></think>). "
                "Then, provide a detailed outline/sketch of your improved solution in natutal language based on the previous plan and your proposed improvement (wrapped in <plan></plan>). "
                "You do not need to implement the solution but you should provide enough detail for another engineer to implement it in Python code."
            )
        }

    @abstractmethod
    def step(self):
        pass


class OneShotImprover(Improver):
    def __init__(
        self,
        model: str,
        task_desc: str,
        memory: Optional[str] = None,
        human_guideline: Optional[str] = None,
    ):
        super().__init__(model, task_desc, memory, human_guideline)

    @property
    def _prompt_introduction(self):
        if self.is_buggy:
            return {
                "Introduction": (
                    "You are an expert machine learning engineer attempting a task. "
                    "You are provided with the plan, code and execution output of a previous solution below that had a bug and/or did not produce a submission.csv, and should improve it in order to fix the bug. "
                    "For this you should first propose an reasonanle improvement and accordingly outline a detailed improved plan in natural language, which will be implemented by another engineer. "
                    "We will now provide a description of the task."
                )
            }
        elif self.is_hpo:
            return {
                "Introduction": (
                    "You are an expert machine learning engineer attempting a task. "
                    "You are provided with the plan, code and execution output of a previous solution below and should optimize the hyperparameters in order to further increase the test time performance. "
                    "For this you should first propose a reasonable and efficient hyperparameter search space and accordingly outline a detailed improved plan in natural language, which will be implemented by another engineer. "
                    "We will now provide a description of the task."
                )
            }
        elif self.knowledge:
            return {
                "Introduction": (
                    "You are an expert machine learning engineer attempting a task. "
                    "You are provided with the plan, code and execution output of a previous solution below and should improve it in order to further increase the test time performance. "
                    "For this you should integrate integrate several useful tricks provided and accordingly outline a detailed improved plan in natural language, which will be implemented by another engineer. "
                    "We will now provide a description of the task."
                )
            }
        else:
            return {
                "Introduction": (
                    "You are an expert machine learning engineer attempting a task. "
                    "You are provided with the plan, code and execution output of a previous solution below and should improve it in order to further increase the test time performance. "
                    "For this you should first propose a reasonable improvement and accordingly outline a detailed improved plan in natural language, which will be implemented by another engineer. "
                    "We will now provide a description of the task."
                )
            }

    @property
    def _prompt_improve_guideline(self):
        if self.is_buggy:
            improve_guideline = [
                "You should pay attention to the execution output of the previous solution, and propose an improvement that will fix the bug.",
                "The improved plan should be derived by adapting the previous plan only based on the proposed improvement, while retaining other details of the previous plan."
                "Don't suggest to do Exploratory Data Analysis.",
                "Don't suggest to do hyperparameter optimization, you should use the best hyperparameters from the previous solution.",
                "If a `sample_submission.csv` file existes, directly load it and use it as a template for the `submission.csv` file. The solution should save predictions on the provide unlabeled test data in the `submission.csv` file in the ./submission/ directory.",
                "When describing your improved plan, do not use phrases like 'the same as before' or 'as in the previous plan'. Instead, fully restate all details from the previous plan that you want to retain, as subsequent implementation will not have access to the previous plan.",
            ]
        elif self.is_hpo:
            # HPO-specific guidance
            improve_guideline = [
                "You should focus ONLY on hyperparameter optimization (HPO) for this improvement.",
                "Identify the key hyperparameters in the model that could be tuned to improve performance.",
                "Suggest specific hyperparameter ranges or values to try, based on the model type and task.",
                "For the efficiency of the hyperparameter search, you should concentrate on the most important hyperparameters and their interactions.",
                "The improved plan should be derived by adapting the previous plan only based on the hyperparameter search, while retaining other details of the previous plan.",
                "Don't change the core algorithm or feature engineering - ONLY tune the hyperparameters.",
                # "Provide clear justification for why these hyperparameter changes might improve performance.",
                "Consider using commonly used HPO libraries such as Optuna.",
                "If a `sample_submission.csv` file existes, directly load it and use it as a template for the `submission.csv` file. The solution should save predictions on the provide unlabeled test data in the `submission.csv` file in the ./submission/ directory.",
                "When describing your improved plan, do not use phrases like 'the same as before' or 'as in the previous plan'. Instead, fully restate all details from the previous plan that you want to retain, as subsequent implementation will not have access to the previous plan.",
            ]
        elif self.knowledge:
            # Knowledge-specific guidance
            improve_guideline = [
                "You should focus ONLY on integrating the provided tricks in the knowledge section into the previous solution to fully leverage their potentials.",
                "Make sure to fully integrate these tricks into your plan while preserving as much details as possible.",
                "Ensure that your plan clearly demonstrates the functions and specifics of the tricks. ",
                "Identify the key areas in the previous solution where the knowledge can be applied.",
                "Suggest specific changes or additions to the code or plan based on the knowledge provided, and avoid unnecessary modifications irrelevant to the tricks.",
                "If a `sample_submission.csv` file existes, directly load it and use it as a template for the `submission.csv` file. The solution should save predictions on the provide unlabeled test data in the `submission.csv` file in the ./submission/ directory.",
                "When describing your improved plan, do not use phrases like 'the same as before' or 'as in the previous plan'. Instead, fully restate all details from the previous plan that you want to retain, as subsequent implementation will not have access to the previous plan.",
            ]
        else:
            improve_guideline = [
                "You should conduct only one expert-level actionable improvement to the previous solution.",
                "This improvement should be atomic so that the effect of the improved solution can be experimentally evaluated.",
                "The improved plan should be derived by adapting the previous plan only based on the proposed improvement, while retaining other details of the previous plan.",
                "Don't suggest to do Exploratory Data Analysis.",
                "Don't suggest to do hyperparameter optimization, you should use the best hyperparameters from the previous solution.",
                "If a `sample_submission.csv` file existes, directly load it and use it as a template for the `submission.csv` file. The solution should save predictions on the provide unlabeled test data in the `submission.csv` file in the ./submission/ directory.",
                "When describing your improved plan, do not use phrases like 'the same as before' or 'as in the previous plan'. Instead, fully restate all details from the previous plan that you want to retain, as subsequent implementation will not have access to the previous plan.",
            ]
        return {"Improve guideline": improve_guideline}

    def filter_knowledge(self, task: str, plan: str, knowledge: Optional[str] = None):
        if knowledge is None:
            return knowledge
        prompt = f"""
            # Task
            Given a machine learning task and the corresponding solution plan, determine whether the provided trick can be applied to the current plan and improve its performance.
            Here is the task description:
            {task}
            Here is the solution plan:
            {plan}
            Here is the trick distilled from relevant task solutions:
            {knowledge}
            
            # Note
            Ensure that:
            1. The trick should be compatible with the current task type and dataset format, and be technically applicable to the current solution.
            2. The trick should not include external inaccessible resources, such as external datasets, public leaderboard score, paid APIs, while allowing use of public Hugging Face models and datasets.
            3. The trick should provide improvement guidance that can be implemented at code level, not vague or broad concepts.
            4. The trick should not have been included in the current solution yet.
            5. The trick should theoretically improve the experimental results. For example, trick involving obviously outdated technologies should be discarded.
            
            # Response Format
            Conduct a careful and thorough comparative analysis of the aforementioned task, plan, and trick. 
            At the end of your response, provide your final conclusion wrapped in backticks (```), with 'True' indicating that the trick should be applied, and 'False' indicating otherwise.
        """
        response = query(
            system_message=prompt,
            user_message=None,
            model=self.model,
            temperature=0.5,
        )
        try:
            result = re.findall(r"```(.*?)```", response)[-1]
            if clean_string("False") in clean_string(result):
                logger.info(
                    f"The provided trick {knowledge} is not applicable to the current task and plan. Omitting it from the knowledge section."
                )
                return None
            else:
                return knowledge
        except IndexError:
            return knowledge

    def prompt_integration(
        self,
        plan: str,
        code: str,
        exec_output: str,
        knowledge: Optional[str] = None,
        data_analysis: Optional[str] = None,
    ):
        prompt = self._prompt_introduction
        prompt["Task description"] = self.task_desc

        if self.memory:
            prompt["Memory"] = (
                "Take the Memory section into consideration when proposing the solution plan, don't propose the similar solution but keep the evaluation metric exactly the same."
            )
            if self.is_hpo:
                prompt[
                    "Memory"
                ] += " You should also consider the memory section when proposing the hyperparameter search space, try to come up with a more reasonable and efficient search space."
            prompt["Memory"] += f"\n\n{self.memory}"

        prompt["Previous Solution"] = {
            "Previous Plan": plan,
            "Previous Code": code,
            "Previous Execution Output": exec_output,
        }

        if knowledge:
            prompt["Knowledge"] = (
                "Here are some tricks that have proved useful for the task: "
                f"\n\n{knowledge}\n\n"
                "You should carefully consider these tricks when designing your solution."
            )
        if data_analysis:
            prompt["Data Analysis"] = data_analysis

        prompt["Instructions"] = {}
        prompt["Instructions"] |= self._prompt_resp_fmt
        prompt["Instructions"] |= self._prompt_environment
        prompt["Instructions"] |= self._prompt_improve_guideline
        if self.human_guideline:
            prompt["Instructions"] |= {"Human Guideline": self.human_guideline}

        return prompt

    def step(
        self,
        plan: str,
        code: str,
        exec_output: str,
        is_buggy: bool = False,
        is_hpo: Optional[bool] = False,
        memory: Optional[str] = None,
        knowledge: Optional[str] = None,
        data_analysis: Optional[str] = None,
        retry: int = 3,
    ):
        self.is_buggy = is_buggy
        self.is_hpo = is_hpo
        self.memory = memory
        self.knowledge = self.filter_knowledge(
            self.task_desc,
            plan,
            knowledge,
        )
        prompt = self.prompt_integration(
            plan, code, exec_output, self.knowledge, data_analysis
        )

        for _ in range(retry):
            response = query(
                system_message=prompt,
                user_message=None,
                model=self.model,
                temperature=0.5,
            )

            think_text = extract_xml(response, "think")
            plan_text = extract_xml(response, "plan")

            if think_text and plan_text:
                logger.info(
                    f"Plan generation successed. Improved plan: \n\n{plan_text}\n\nProposed improvement: \n\n{think_text}\n\n"
                )
                return plan_text, self.knowledge
            else:
                logger.info(f"Plan generation failed, retrying...")

        logger.error(f"Plan generation failed after {retry} retries.")
        return None, None
