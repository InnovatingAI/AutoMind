import random
import logging
from abc import ABC, abstractmethod
from typing import Optional
import re
import ast

from .backend import query
from .utils.response import (
    extract_code,
    extract_xml,
    extract_json,
    wrap_code,
    trim_long_string,
    delete_debug_inform,
)
from .interpreter import Interpreter

logger = logging.getLogger("automind")


class Coder(ABC):
    def __init__(
        self,
        model: str,
        task_desc: str,
        human_guideline: Optional[str] = None,
    ):
        self.model = model
        self.task_desc = task_desc
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

    @abstractmethod
    def step(self):
        pass


class OneShotCoder(Coder):
    def __init__(
        self,
        model: str,
        task_desc: str,
        interpreter: Optional[Interpreter] = None,
        human_guideline: Optional[str] = None,
    ):
        super().__init__(model, task_desc, human_guideline)
        self.interpreter = interpreter

    @property
    def _prompt_introduction(self):
        return {
            "Introduction": (
                "You are an expert machine learning engineer attempting a task. "
                "In order to complete this task, you need to implement a natural language solution plan proposed by another engineer in Python code. "
                "We will now provide a description of the task."
            )
        }

    @property
    def _prompt_code_guideline(self):
        code_guideline = [
            "The code should **implement the proposed solution** and **print the value of the evaluation metric computed on a hold-out validation set**,",
            "**AND MOST IMPORTANTLY SAVE PREDICTIONS ON THE PROVIDED UNLABELED TEST DATA IN A `submission.csv` FILE IN THE ./submission/ DIRECTORY.**",
            "The code should save the evaluation metric computed on the hold-out validation set in a `eval_metric.txt` file in the ./submission/ directory.",
            "The code should be a single-file python program that is self-contained and can be executed as-is.",
            "No parts of the code should be skipped, don't terminate the code before finishing the script.",
            "DO NOT WRAP THE CODE IN A MAIN FUNCTION, BUT WRAP ALL CODE in the '__main__' module, or it cannot be executed successfully.",
            'All class initializations and computational routines MUST BE WRAPPED in `if __name__ == "__main__":`.',
            "DO NOT USE MULTIPROCESSING OR SET 'num_workers' IN DATA LOADER, as it may cause the container to crash.",
            "Your response should only contain a single code block.",
            "All input data is already prepared and available in the `./input` directory. There is no need to unzip any files.",
            'DO NOT load data from "./data" directory, it is not available in the environment.',
            # '**If there is test data provided for this task, please save the test predictions in a `submission.csv` file in the "./working" directory as described in the task description** This is extremely important since this file is used for grading/evaluation. DO NOT FORGET THE submission.csv file!',
            # 'You can also use the "./working" directory to store any temporary files that your code needs to create.',
            "Do not save any intermediate or temporary files through `torch.save` or `pickle.dump`.",
            "Try to accelerate the model training process if any GPU is available.",
            "DO NOT display progress bars. If you have to use function intergrated with progress bars, disable progress bars or using the appropriate parameter to silence them.",
            "Don't do Exploratory Data Analysis.",
        ]
        return {"Code guideline": code_guideline}

    @property
    def _prompt_resp_fmt(self):
        return {
            "Response format": (
                "Your response should be a single markdown code block (wrapped in ```) which implements this solution plan and prints out and save the evaluation metric. "
            )
        }

    def prompt_integration(
        self,
        plan: str,
        data_analysis: Optional[str] = None,
        based_code: Optional[str] = None,
    ):
        prompt = self._prompt_introduction
        prompt["Task Description"] = self.task_desc
        prompt["Proposed Solution"] = plan

        if based_code:
            prompt["Based Code"] = (
                "You should modify the following based code to implement the proposed solution: "
                f"\n\n{based_code}\n\n"
            )

        if data_analysis:
            prompt["Data Analysis"] = data_analysis

        prompt["Instructions"] = {}
        prompt["Instructions"] |= self._prompt_resp_fmt
        prompt["Instructions"] |= self._prompt_environment
        prompt["Instructions"] |= self._prompt_code_guideline
        if self.human_guideline:
            prompt["Instructions"] |= {"Human Guideline": self.human_guideline}

        return prompt

    def step(
        self,
        plan: str,
        data_analysis: Optional[str] = None,
        based_code: Optional[str] = None,
        retry: int = 3,
    ):
        prompt = self.prompt_integration(plan, data_analysis, based_code)
        for _ in range(retry):
            response = query(
                system_message=prompt,
                user_message=None,
                model=self.model,
                temperature=0.5,
            )

            code = extract_code(response)
            if code:
                try:
                    ast.parse(code)
                    logger.info(
                        f"Code generation successed. Implemented code: \n\n{code}\n\n"
                    )
                    logger.info(f"Agent is executing the code")
                    exec_result = self.interpreter.run(code)
                    return code, exec_result
                except Exception as e:
                    logger.info("Code generation failed, retrying ...")
                    continue
            else:
                logger.info("Code generation failed, retrying ...")
        logger.error(f"Code generation failed after {retry} retries")
        return None, None


class StepByStepCoder(Coder):
    def __init__(
        self,
        model: str,
        task_desc: str,
        interpreter: Optional[Interpreter] = None,
        human_guideline: Optional[str] = None,
    ):
        super().__init__(model, task_desc, human_guideline)
        self.interpreter = interpreter
        self._prev_steps = (
            []
        )  # List[dict]: a list of steps. Each step is in JSON format.
        self._prev_code = []  # List[str]: the corresponding code string for each step.
        self._plan_list = (
            []
        )  # List[dict]: a list of steps for code generation. Each step is in JSON format.

    def reset(self):
        self._prev_steps = []
        self._prev_code = []
        self._plan_list = []

    @property
    def _prompt_introduction(self):
        return {
            "Introduction": (
                "You are an expert machine learning engineer attempting a task. "
                "In order to complete this task, you are given the code for previous steps and need to implement the current step of a natural language solution plan proposed by another engineer in Python code. "
                "We will now provide a description of the task."
            )
        }

    @property
    def _prompt_code_guideline(self):
        code_guideline = [
            "You should first provide suggestions for the current step based on the previous steps and the failed last try step if provided, and then implement the current step of the solution plan.",
            "**You should ONLY implement the code for the current step of the solution plan, rather than the entire solution plan.**",
            "DO NOT MODIFY THE CURRENT STEP. You should implement the current step exactly as it is.",
            "You should **print the value of the evaluation metric computed on a hold-out validation set** if it is calculated in the current step.",
            "You should save the evaluation metric computed on the hold-out validation set in a `eval_metric.txt` file in the ./submission/ directory if it is calculated in the current step.",
            "DO NOT PRINT ANYTHING ELSE IN THE CODE, except for the evaluation metric and completion message for the current step.",
            "The code should be a single-file python program that is self-contained and can be executed as-is.",
            "DO NOT WRAP THE CODE IN A MAIN FUNCTION, BUT WRAP ALL CODE in the '__main__' module, or it cannot be executed successfully.",
            'All class initializations and computational routines MUST BE WRAPPED in `if __name__ == "__main__":`.',
            "DO NOT USE MULTIPROCESSING OR SET 'num_workers' IN DATA LOADER, as it may cause the container to crash.",
            "No parts of the code should be skipped, don't terminate the code before finishing the script.",
            "**DO NOT REPEAT the code for previous steps, you should only import them from prev_steps.py.**",
            "DO NOT REPETITIVELY IMPORT THE SAME MODULES IN PREVIOUS STEPS, but you can import other modules if needed.",
            "**AND MOST IMPORTANTLY SAVE PREDICTIONS ON THE PROVIDED UNLABELED TEST DATA IN A `submission.csv` FILE IN THE ./submission/ DIRECTORY.** if predictions are involved in the current step.",
            "All input data is already prepared and available in the `./input` directory. There is no need to unzip any files.",
            'DO NOT load data from "./data" directory, it is not available in the environment.',
            # '**If there is test data provided for this task, please save the test predictions in a `submission.csv` file in the "./working" directory as described in the task description** This is extremely important since this file is used for grading/evaluation. DO NOT FORGET THE submission.csv file!',
            # 'You can also use the "./working" directory to store any temporary files that your code needs to create.',
            "Do not save any intermediate or temporary files through `torch.save` or `pickle.dump`.",
            "You can reference to the based code to implement the current step, but do not completely repeat it.",
            "Try to accelerate the model training process if any GPU is available.",
            "DO NOT display progress bars. If you have to use function intergrated with progress bars, disable progress bars or using the appropriate parameter to silence them.",
            # "Always add 'import os\nos.environ[\"TQDM_DISABLE\"] = \"1\"' at the beginning of your code to disable progress bars. Always deactivate pytorch internal progress bars by setting, for example, 'prefetch_factor=None'",
            "Don't do Exploratory Data Analysis.",
        ]
        return {"Code guideline": code_guideline}

    @property
    def _prompt_resp_fmt(self):
        return {
            "Response format": (
                "First, provide suggestions for the current step based on the previous steps and the failed last try step if provided (wrapped in <think></think>). "
                "Then, provide a single markdown code block (wrapped in ```) which implements the current step of a solution plan. "
            )
        }

    @property
    def _has_next_step(self):
        # Check if there is a next step for code generation
        return bool(len(self._prev_steps) < len(self._plan_list))

    def plan_decomposition(self, plan: str, retry: int = 3):
        """
        Decompose the plan into multiple steps.

        """
        prompt = {
            "Introduction": (
                "You are an expert machine learning engineer attempting a task. "
                "In order to complete this task, you are given the proposed solution and supposed to decompose it into multiple steps. "
                "We will now provide a description of the task."
            )
        }
        prompt["Task Description"] = self.task_desc
        prompt["Proposed Solution"] = plan
        prompt["Instructions"] = {}
        prompt["Instructions"] |= {
            "Solution decomposition guideline": [
                "You should decompose the proposed solution into multiple steps, and provide detailed descriptions of each step.",
                "DO NOT MODIFY THE PROPOSED SOLUTION. In the description of each step, you should keep as many details of the proposed solution as possible, especially the exact hyperparameters and sample code.",
                "DO NOT CHEAT ON EVALUATION. The solution should calculate the evaluation metric described in the task description on a hold-out validation set.",
                "If the evaluation metric is not provided, you should propose a reasonable evaluation metric for the task and calculate it.",
                "The solution should save the evaluation metric computed on the hold-out validation set in a `eval_metric.txt` file in the ./submission/ directory.",
                "The solution should use os.walk to get the paths of all available files in the `. /input` directory for data loading.",
                "If a sample_submission.csv file existes, directly load it and use it as a template for the `submission.csv` file. The solution should save predictions on the provide unlabeled test data in the `submission.csv` file in the ./submission/ directory.",
                "You should **print the value of the evaluation metric computed on a hold-out validation set** in the last step of the decomposed steps.",
                "Don't do Exploratory Data Analysis in the decomposition steps.",
                "If you find improvements suggestions in the plan, you should take them in serious consideration and include them in the decomposition steps.",
                "You do not need to implement the code in the decomposed steps. ",
                "Note that the order of the decomposed steps determines the order in which the code is implemented and executed.",
            ]
        }
        prompt["Instructions"] |= {
            "Response format": [
                "Your response should be a single JSON code block (wrapped in ```) which contains the decomposition steps of the proposed solution. ",
                """The generated JSON should have the following format: 
                {
                    "decomposed steps": [
                        {
                            "step": "Name of the step",
                            "details": "Detailed description of the step",
                        },
                        ...
                    ],
                }""",
            ]
        }
        response = None
        for _ in range(retry):
            response = query(
                system_message=prompt,
                user_message=None,
                model=self.model,
                temperature=0.5,
            )

            json_obj = extract_json(response)
            if json_obj and "decomposed steps" in json_obj:
                # Merge model training step and all subsequent steps into one step
                merged_steps = []
                train_step_exist = False
                for i, step in enumerate(json_obj["decomposed steps"]):
                    if (
                        "train" in step["step"].lower()
                        and "split" not in step["step"].lower()
                        and "prepare" not in step["step"].lower()
                        and "preparation" not in step["step"].lower()
                        and "configure" not in step["step"].lower()
                        and "configuration" not in step["step"].lower()
                        and "define" not in step["step"].lower()
                        and "definiton" not in step["step"].lower()
                        and "download" not in step["step"].lower()
                        and "load" not in step["step"].lower()
                        and "set up" not in step["step"].lower()
                    ):
                        train_step_exist = True
                        merged_steps.append(step)

                    if not train_step_exist:
                        merged_steps.append(step)
                    else:
                        if merged_steps[-1]["step"] != step["step"]:
                            merged_steps[-1]["step"] += ". " + step["step"]
                            merged_steps[-1]["details"] += step["details"]

                max_steps = 6
                if len(merged_steps) > max_steps:
                    overflow = merged_steps[max_steps - 1 :]
                    merged_steps = merged_steps[: max_steps - 1]
                    start = overflow[0].copy()
                    steps_text = ". ".join(step["step"] for step in overflow)
                    details_text = "".join(step["details"] for step in overflow)
                    start["step"] = steps_text
                    start["details"] = details_text
                    merged_steps.append(start)

                logger.info(
                    f"JSON extraction successed. Decomposed steps: \n\n{merged_steps}\n\n"
                )
                return merged_steps
            else:
                logger.info("JSON extraction failed, retrying ...")
        logger.error("JSON extraction failed after multiple retries")
        return response

    def prompt_integration(
        self,
        step_plan: dict,
        prev_steps_code: str,
        data_analysis: Optional[str] = None,
        based_code: Optional[str] = None,
        last_try_step_code: Optional[str] = None,
        last_try_exec_output: Optional[str] = None,
    ):

        prompt = self._prompt_introduction
        prompt["Task Description"] = self.task_desc
        prompt["Current Step"] = str(step_plan)
        if prev_steps_code:
            prompt["Previous Steps Code"] = (
                "You should continue the following code for previous steps to implement the current step of the solution plan, but do not repeat it: "
                f"\n\n{wrap_code(prev_steps_code)}\n\n"
            )

        if last_try_step_code and last_try_exec_output:
            removeprefix_step_code = last_try_step_code.removeprefix(
                f"{prev_steps_code}\n"
            )
            prompt["Last Try Step Code"] = (
                "The last try failed. You should modify the following last try step code to implement the current step of solution plan: "
                f"\n\n{wrap_code(removeprefix_step_code)}\n\n"
                "The execution output of the last try step code is as follows: "
                f"\n\n{trim_long_string(last_try_exec_output)}\n\n"
                "You should pay attention to the error message in the execution output and modify the last try step code accordingly to fix the bug.\n"
                # "You should print some debugging information to display the intermidiate traceback result, so whether you have fix the bug in the current stage, the following debugging stage can have chance to refer to your output to fix the bug better. Note: The content should starts with '[debug]', for example, if you are encountering a type error in DataLoader, write 'print('[debug] Type:', type(x))'\n"
                # "DO NOR print debug information in a loop or in a forward function, as it may cause the output to be too long. If you find the last try code contains printing in a loop, you should modify it. \n"
            )

        if based_code:
            prompt["Based Code"] = (
                "You can refer to the following based code to implement the current step, but DO NOT generate code that is not related to the current step: "
                f"\n\n{based_code}\n\n"
            )

        if data_analysis:
            prompt["Data Analysis"] = data_analysis

        prompt["Instructions"] = {}
        prompt["Instructions"] |= self._prompt_resp_fmt
        prompt["Instructions"] |= self._prompt_environment
        prompt["Instructions"] |= self._prompt_code_guideline
        if self.human_guideline:
            prompt["Instructions"] |= {"Human Guideline": self.human_guideline}

        return prompt

    def gen_step_code(
        self,
        step_plan: dict,
        data_analysis: Optional[str] = None,
        based_code: Optional[str] = None,
        retry: int = 5,
        last_try_step_code: Optional[str] = None,
        last_try_exec_output: Optional[str] = None,
    ):
        """
        Generate the code for the current step.

        """
        prev_steps_code = self._prev_code[-1] if self._prev_code else None
        prompt = self.prompt_integration(
            step_plan,
            prev_steps_code,
            data_analysis,
            based_code,
            last_try_step_code,
            last_try_exec_output,
        )
        for _ in range(retry):
            response = query(
                system_message=prompt,
                user_message=None,
                model=self.model,
                temperature=0.5,
            )

            thoughts = extract_xml(response, "think")
            # Extract the code from the response
            step_code = extract_code(response)

            if step_code:
                # Delete the lines that import the code of the previous steps
                # Regular expression matches multiple lines starting with “from prev_steps import (”
                pattern = r"from\s+prev_steps\s+import\s+\(.*?\)"
                step_code = re.sub(pattern, "", step_code, flags=re.DOTALL)
                # Regular expression matches single line starting with “from prev_steps import”
                pattern = r"from\s+prev_steps\s+import\s+.*?\n"
                step_code = re.sub(pattern, "", step_code)

                # Combine the code of the previous steps and the current step
                code = (
                    f"{prev_steps_code}\n{step_code}" if prev_steps_code else step_code
                )
                try:
                    ast.parse(code)
                    # TODO: check the plan-code consistency before return code
                    logger.info(
                        f"Code generation successed for step {len(self._prev_steps)+1} of {len(self._plan_list)}. Implemented code: \n\n{step_code}\n\nThoughts: \n\n{thoughts}\n\n"
                    )
                    return step_code, code
                except Exception as e:
                    logger.info(
                        f"Code generation failed for step {len(self._prev_steps)+1} of {len(self._plan_list)} with the following exception: \n{e}\n"
                    )
            else:
                logger.info(
                    f"Code generation failed for step {len(self._prev_steps)+1} of {len(self._plan_list)}, retrying ..."
                )

        logger.error(
            f"Code generation failed for step {len(self._prev_steps)+1} of {len(self._plan_list)} after {retry} retries"
        )
        return None, None

    def test_step_code(self, code: str, reset_session: bool = True):
        """
        Test the code for the current step.

        """
        exec_result = None
        try:
            exec_result = self.interpreter.run(code=code, reset_session=reset_session)
            # Check if the code has raised an exception
            if exec_result.exc_type:
                logger.info(
                    f"Code testing failed for step {len(self._prev_steps)+1} of {len(self._plan_list)} with the following error: \n{exec_result.term_out}\n"
                )
                return False, exec_result
            else:
                logger.info(
                    f"Code testing successed for step {len(self._prev_steps)+1} of {len(self._plan_list)}. Execution output: \n{exec_result.term_out}\n"
                )
                return True, exec_result
        except Exception as e:
            logger.error(
                f"Code testing failed for step {len(self._prev_steps)+1} of {len(self._plan_list)} with the following exception: \n{e}\n"
            )
            return False, exec_result

    def test_plan_code_consistency(self, code: str, plan: str, verbose: bool = True):
        """
        Test the consistency of the code with the plan.
        """

        prompt = """
            You are an ML expert.  
            I have an ML plan with several components (e.g., functions, calculations, equations).  
            Check if **every component in the plan is implemented in the code**.  

            Rules:  
            1. **Mandatory**: The code must cover the key components listed in the plan. Extra code is allowed.  
            2. If key components are implemented (correctly and completely), please return YES + a short summary(<=50 words).
            3. If any component is missing or incorrect, please return NO, along with the analysis on the plan-code consistency (<= 300 words).
            
            
            # Plan: {} \n\n
            # Code: {}
            
            
            The response should be in JSON format.
            {{
                "response":"YES/NO",
                "analysis": [summary or hints, list type]
            }}
            
        """

        response = query(
            system_message=prompt,
            user_message=prompt.format(plan, code),
            model=self.model,
            temperature=0,
        )

        res = "yes" in response.lower()
        if verbose:
            logger.info(
                f"Check the consistency between plan and code. \n Analysis:{response}"
            )
        return res, response

    def step(
        self,
        plan: str,
        data_analysis: Optional[str] = None,
        based_code: Optional[str] = None,
        verify: bool = False,
        retry: int = 4,
    ):
        plan_list = self.plan_decomposition(plan)
        self._plan_list = plan_list

        exec_result = None
        all_exec_output: list[str] = []

        while self._has_next_step:
            # Get the plan for the next step
            current_step_plan = self._plan_list[len(self._prev_steps)]
            last_try_step_code = None
            last_try_exec_output = None
            code_check = False
            for _ in range(retry):
                current_step_code = None
                current_all_code = None
                # If it is the first try, we should generate the code based on the plan
                if not last_try_step_code and not last_try_exec_output:
                    current_step_code, current_all_code = self.gen_step_code(
                        step_plan=current_step_plan,
                        data_analysis=data_analysis,
                        based_code=based_code,
                        retry=retry,
                    )
                # If it is not the first try and the last try failed, we should try to modify the code based on the last try
                else:
                    current_step_code, current_all_code = self.gen_step_code(
                        step_plan=current_step_plan,
                        data_analysis=data_analysis,
                        based_code=based_code,
                        retry=retry,
                        last_try_step_code=last_try_step_code,
                        last_try_exec_output=last_try_exec_output,
                    )

                if current_step_code and current_all_code:
                    # If it is the first step, start the session to execute the current step code
                    if len(self._prev_steps) == 0:
                        logger.info(
                            "Starting the session to execute the current step code"
                        )
                        code_check, exec_result = self.test_step_code(
                            code=current_step_code,
                            reset_session=True,
                        )
                    # If the execution of the previous code successes, continue the session to execute the current step code
                    elif not last_try_step_code and not last_try_exec_output:
                        logger.info(
                            "Continue the session to execute the current step code"
                        )
                        code_check, exec_result = self.test_step_code(
                            code=current_step_code,
                            reset_session=False,
                        )
                    # If the execution of the previous code fails, restart the session to execute all the current code
                    else:
                        logger.info(
                            "Restart the session to execute all the current code"
                        )
                        code_check, exec_result = self.test_step_code(
                            code=current_all_code,
                            reset_session=True,
                        )

                    if code_check:
                        # If the execution of the previous code successes, we should concat the current execution output with execution output of previous steps
                        if not last_try_step_code and not last_try_exec_output:
                            all_exec_output.extend(exec_result.term_out)
                        # If the execution of the previous code fails, the current execution output is actually the execution output of all steps by far
                        else:
                            all_exec_output = exec_result.term_out
                        break
                    else:
                        last_try_step_code = current_step_code
                        last_try_exec_output = exec_result.term_out
                else:
                    logger.info("Code testing failed, retrying ...")
            self._prev_code.append(current_all_code)
            self._prev_steps.append(current_step_plan)
            if not code_check:
                logger.error(
                    f"Code testing failed for step {len(self._prev_steps)} of {len(self._plan_list)} after {retry} retries"
                )
                all_exec_output = exec_result.term_out
                break

        self.reset()
        exec_result.term_out = all_exec_output
        return current_all_code, exec_result


class MixtureCoder(StepByStepCoder):
    def __init__(
        self,
        model: str,
        task_desc: str,
        interpreter: Optional[Interpreter] = None,
        human_guideline: Optional[str] = None,
    ):
        super().__init__(model, task_desc, human_guideline)
        self.interpreter = interpreter
        self.one_shot_coder = OneShotCoder(
            model, task_desc, interpreter, human_guideline
        )

    def complexity_scoring(
        self,
        plan: str,
        data_analysis: Optional[str] = None,
        retry: int = 3,
    ):
        prompt = dict()
        prompt |= {
            "Introduction": (
                "You are an expert machine learning engineer attempting a task. "
                "In order to complete this task, you are given a discription of the task and a solution plan proposed by another engineer and need to assess the complexity of the task and the proposed solution. "
                "We will now provide a description of the task."
            )
        }
        prompt["Task Description"] = self.task_desc
        prompt["Proposed Solution"] = plan
        prompt["Data Analysis"] = data_analysis
        prompt["Instructions"] = {}
        prompt["Instructions"] |= {
            "Response format": (
                "First, provide a brief explanation of your reasoning for the assessment of the complexity of the task and the proposed solution (wrapped in <think></think>). "
                "Then, provide ONLY ONE average complexity score as floating point number between 1 and 5, which can contain 0.5 points (wrapped in <score></score>). "
            )
        }
        prompt["Instructions"] |= {
            "Task complexity scoring criteria": [
                "5 = Extremely complex and cutting-edge task with high levels of innovation required. This involves solving a unique or highly specialized problem that may push the boundaries of existing knowledge or technology.",
                "4 = Complex task that involves advanced techniques or methodologies, requiring considerable expertise in the domain, such as building novel algorithms, optimization methods, or handling advanced data.",
                "3 = Moderately complex task that requires significant problem-solving, such as integrating different methods or creating custom algorithms for specific use cases.",
                "2 = Simple task with some level of complexity, such as basic algorithms that need some degree of fine-tuning or adjustment to meet the specific requirements of the project.",
                "1 = Very simple task that requires minimal effort, such as basic data manipulation or applying standard algorithms without any customization.",
            ],
            "Proposed solution complexity scoring criteria": [
                "5 = A groundbreaking or transformative solution that pushes the envelope in the field. It introduces a novel approach that is scalable, efficient, and offers long-term value or sets a new standard.",
                "4 = A highly original and effective solution that shows a deep understanding of the problem domain and offers a significant contribution to the field. The solution is well-optimized and efficient.",
                "3 = An original and creative solution with a reasonable level of complexity. It involves designing and implementing custom solutions or combining existing methods in a new way.",
                "2 = A somewhat original solution that involves adapting existing tools or methods with some customization to meet the needs of the project. There may be room for optimization or improvement.",
                "1 = Very basic solution, perhaps using standard, off-the-shelf tools with minimal adaptation, lacking originality or novel contributions.",
            ],
            "Complexity scoring guideline": [
                "Evaluate the complexity of the task and the proposed solution, and assign a score between 1 and 5.",
                "Assign an average score between 1 and 5, consider factors such as the task's complexity, the proposed solution, the dataset size, and the time and hardware resources required for implementation and execution.",
            ],
        }

        for _ in range(retry):
            response = query(
                system_message=prompt,
                user_message=None,
                model=self.model,
                temperature=0,
            )

            explanation = extract_xml(response, "think")
            complexity_score = extract_xml(response, "score")
            if explanation and complexity_score:
                try:
                    complexity_score = float(complexity_score)
                    if 1 <= complexity_score <= 5:
                        logger.info(
                            f"Complexity scoring successed. Complexity score: {complexity_score}. Explanation: \n\n{explanation}\n\n"
                        )
                        return complexity_score, explanation
                    else:
                        logger.error(
                            "Complexity score is out of range (1-5), retrying ..."
                        )
                except ValueError:
                    logger.error("Complexity score is not a valid float, retrying ...")
            else:
                logger.info("Complexity scoring failed, retrying ...")

        logger.error(f"Complexity scoring failed after {retry} retries")
        return 5, None

    def step(
        self,
        plan: str,
        data_analysis: Optional[str] = None,
        retry: int = 3,
        based_code: Optional[str] = None,
        mode: Optional[str] = None,
    ):
        if not mode:
            logger.info(
                "Agent is assessing the complexity of the task and the proposed solution"
            )
            # Check the complexity of the task and the proposed solution
            complexity_score, explanation = self.complexity_scoring(
                plan=plan,
                data_analysis=data_analysis,
            )

            if complexity_score > 3.5:
                logger.info(
                    f"Complexity score is {complexity_score}, Agent is using step-by-step coding"
                )
                return super().step(
                    plan=plan,
                    data_analysis=data_analysis,
                    based_code=based_code,
                    retry=retry,
                )
            else:
                logger.info(
                    f"Complexity score is {complexity_score}, Agent is using one-shot coding"
                )
                return self.one_shot_coder.step(
                    plan=plan,
                    data_analysis=data_analysis,
                    based_code=based_code,
                    retry=retry,
                )
        elif mode == "one-shot":
            logger.info("Agent is using one-shot coding")
            return self.one_shot_coder.step(
                plan=plan,
                data_analysis=data_analysis,
                based_code=based_code,
                retry=retry,
            )
        elif mode == "step-by-step":
            logger.info("Agent is using step-by-step coding")
            return super().step(
                plan=plan,
                data_analysis=data_analysis,
                based_code=based_code,
                retry=retry,
            )
        else:
            logger.error(f"Invalid mode: {mode}.")
            return None, None
