import json
import random
import logging
from abc import ABC, abstractmethod
from typing import Optional, Tuple, List

from .backend import query
from .utils.response import extract_json, extract_xml, wrap_code

logger = logging.getLogger("automind")


class Planner(ABC):
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
                "Your response should be a detailed outline/sketch of your proposed solution in natural language. "
                "You do not need to implement the solution but you should provide enough detail for another engineer to implement it in Python code. "
                "There should be no additional headings or text in your response. Just natural language text followed by a newline."
            )
        }

    @abstractmethod
    def step(self):
        pass


class OneShotPlanner(Planner):
    def __init__(
        self,
        model: str,
        task_desc: str,
        memory: Optional[str] = None,
        human_guideline: Optional[str] = None,
    ):
        super().__init__(
            model,
            task_desc,
            memory,
            human_guideline,
        )

    @property
    def _prompt_introduction(self):
        return {
            "Introduction": (
                "You are an expert machine learning engineer attempting a task. "
                "In order to complete this task, you need to come up with an excellent and creative plan for a solution, which will be implemented by another engineer. "
                "We will now provide a description of the task."
            )
        }

    @property
    def _prompt_plan_guideline(self):
        plan_guideline = [
            "DO NOT CHEAT ON EVALUATION. The solution should calculate the evaluation metric described in the task description on a hold-out validation set.",
            "If the evaluation metric is not provided, you should propose a reasonable evaluation metric for the task and calculate it.",
            "The solution should print the evaluation metric computed on the hold-out validation set at the last step of the solution.",
            "Try to come up with more modern and powerful methods to feature engineering and modelling and avoid using outdated methods. For example, if the task is a classification task, you should use modern transformer-based models instead of traditional models like CNN or LSTM.",
            "The solution should adopt appropriate methods to prevent model overfitting, such as data augmentation, early stopping, regularization, dropout, and others.",
            "Don't suggest to do model ensembling.",
            "Don't suggest to do Exploratory Data Analysis.",
            "Don't suggest to do hyperparameter tuning.",
            "The data is already prepared and available in the `./input` directory. There is no need to unzip any files.",
            "The solution should use os.walk to get the paths of all available files in the `. /input` directory for data loading.",
            "If a `sample_submission.csv` file existes, directly load it and use it as a template for the `submission.csv` file. The solution should save predictions on the provide unlabeled test data in the `submission.csv` file in the ./submission/ directory.",
        ]
        if self.memory:
            plan_guideline.append(
                "Take the Memory section into consideration when proposing the solution plan. Don't propose the similar modelling solution but propose a better one and keep the evaluation metric exactly the same."
            )

        return {"Plan guideline": plan_guideline}

    def prompt_integration(
        self,
        knowledge: Optional[str] = None,
        data_analysis: Optional[str] = None,
    ):
        prompt = self._prompt_introduction
        prompt["Task description"] = self.task_desc

        if self.memory:
            prompt["Memory"] = (
                "Take the Memory section into consideration when proposing the solution plan, don't propose the similar solution but keep the evaluation metric exactlty the same."
                f"\n\n{self.memory}"
            )

        if knowledge:
            prompt["Knowledge"] = (
                "Some of the tricks that have proved useful for the same type of task are provided as follows: "
                f"\n\n{knowledge}\n\n"
                "You should carefully consider these tricks when designing your solution."
            )
        if data_analysis:
            prompt["Data Analysis"] = data_analysis

        prompt["Instructions"] = {}
        prompt["Instructions"] |= self._prompt_resp_fmt
        prompt["Instructions"] |= self._prompt_environment
        prompt["Instructions"] |= self._prompt_plan_guideline
        if self.human_guideline:
            prompt["Instructions"] |= {"Human Guideline": self.human_guideline}

        return prompt

    def step(
        self,
        memory: Optional[str] = None,
        knowledge: Optional[str] = None,
        data_analysis: Optional[str] = None,
    ):
        self.memory = memory
        prompt = self.prompt_integration(knowledge, data_analysis)
        response = query(
            system_message=prompt,
            user_message=None,
            model=self.model,
            temperature=0.5,
        )

        plan_text = response.strip()
        logger.info(f"Plan generation successed. Proposed plan: \n\n{plan_text}\n\n")
        return plan_text


class KnowledgePlanner(OneShotPlanner):
    def __init__(
        self,
        model,
        task_desc,
        task_analysis=None,
        memory=None,
        human_guideline=None,
    ):
        super().__init__(model, task_desc, memory, human_guideline)

        self._task_analysis = task_analysis

    def check_mergeability(self, papers: dict) -> Tuple[bool, str]:
        """
        Evaluate whether the given papers can be merged.

        Parameters
        ----------

        Returns
        -------
        Tuple[bool, str]
            A tuple containing:
            - bool: True if the papers can be merged, False otherwise.
            - str: The reasoning behind the decision.
        """
        if len(papers.keys()) == 0:
            return True, ""

        prompt = {
            "Introduction": (
                "You are an expert machine learning engineer attempting a task. "
                "Please first analyze the following papers and identify the techniques in three aspects: data processing, model architecture and training algorithm. "
                "Can the techniques from these papers above be used together? Please give a YES or NO answer and give your reason. "
                "We will now provide the papers."
            )
        }
        prompt["Papers"] = papers

        response = query(
            system_message=prompt,
            user_message=None,
            model=self.model,
            temperature=0.5,
        )
        res = "yes" in response.lower()
        return res, response

    def get_paper_analysis(self, papers: dict, retry: int = 3) -> dict:
        prompt = {
            "Introduction": (
                "You are a machine learning expert. "
                "Please analyze the following paper step-by-step to help me gain a better understanding of it. "
                "We will now provide the paper."
            )
        }

        prompt["Paper"] = None
        prompt["Instructions"] = {}
        prompt["Instructions"] |= {
            "Response format": [
                "Your response should be a single JSON code block (wrapped in ```) which contains the key modules of the analyzed paper. "
                """The generated JSON should have the following format:
                {
                    "Key Module 1 name": {
                        "description": "abstract description of this method",
                        "Necessary Inputs": "",
                        "Position":[],
                        "Preferred Scenarios":""
                    }
                    "Key Module 2 name": {
                        ...
                    }
                }
                """
            ]
        }
        prompt["Instructions"] |= {
            "Paper analysis guideline": [
                "Please Identify the key modules used in the paper.",
                "For each key module identified in the previous step, explain how it can be used in designing a machine learning plan.",
                "Specifically, address the following points.",
                "**Necessary Inputs**: What inputs are required for the module to function?",
                "**Position in the Plan**: Where in the overall machine learning plan is this module typically used? Select from these candidates: [data processing, model architecture design, (pre-)training framework, evaluation strategy]. Choose one or more candidates from the list, or you can add another candidates.",
                "**Preferred Scenarios**: What scenarios or use cases does the paper mention where this module is particularly effective?",
            ]
        }

        analysis = {}
        for key, paper in papers.items():
            logger.info(f"Agent is analyzing paper {key}")
            prompt["Paper"] = paper
            for _ in range(retry):
                response = query(
                    system_message=prompt,
                    user_message=None,
                    model=self.model,
                    temperature=0.5,
                )
                json_obj = extract_json(response)
                if json_obj:
                    logger.info(
                        f"JSON extraction successed. Paper analyis for {key}: \n\n{json_obj}\n\n"
                    )
                    analysis[key] = json_obj
                    break
                else:
                    logger.info("JSON extraction failed, retrying ...")
        return analysis

    def check_constraints(
        self, plan: dict, constraints: List, retry: int = 3
    ) -> Tuple[bool, str]:
        """
        Check if the generated plan meets the constraints.

        Parameters
        ----------
        plan : str
            The generated plan.
        constraints : list
            The list of constraints to check against.

        Returns
        -------
        bool
            True if the plan meets all constraints, False otherwise.
        """
        prompt = {
            "Introduction": (
                "You are an expert machine learning engineer attempting a task. "
                "Please check the following plan meet all the given constraints. "
                "We will now provide the plan and the constraints."
            )
        }

        prompt["Plan"] = plan
        prompt["Constraints"] = constraints
        prompt["Instructions"] = {}
        prompt["Instructions"] |= {
            "Response format": [
                "First, provide your explanation of whether the plan meets each of the constraints (wrapped in <think></think>).",
                "Then, provide your final answer, 'success' if all the constraints are met, or else 'fail' if any of the constraints is not met (wrapped in <answer></answer>).",
            ]
        }

        for _ in range(retry):
            response = query(
                system_message=prompt,
                user_message=None,
                model=self.model,
                temperature=0.5,
            )

            think = extract_xml(response, "think")
            answer = extract_xml(response, "answer")
            if think and answer:
                logger.info(f"Plan constraints check {answer} because: \n\n{think}\n\n")
                return "success" in answer.lower(), think
            else:
                logger.info("Plan constraints check failed, retrying ...")

        return False, None

    def check_validity(self, plan: str, retry: int = 3) -> Tuple[bool, str]:
        """
        Evaluate the validity of a plan for a given task.

        """

        prompt = {
            "Introduction": (
                "You are an expert machine learning engineer attempting a task. "
                "Please check whether the following designed plan is valid for the given task. "
                "We will now provide the task and the plan."
            )
        }
        prompt["Task description"] = self.task_desc
        prompt["Desgined plan"] = plan

        prompt["Instructions"] = {}
        prompt["Instructions"] |= {
            "Response format": [
                "First, provide your explanation of whether the plan is valid for the task (wrapped in <think></think>).",
                "Then, provide your final answer, 'success' if the plan is valid for the task, or else 'fail' if the plan is not valid for the task (wrapped in <answer></answer>).",
            ]
        }
        prompt["Instructions"] |= {
            "Plan validity guideline": [
                "A valid plan must satisfy the input consistency. For data processing and modeling strategies to produce prediction results, all required inputs(files or variables) for each step must either: a) Be generated in preceding steps, or b) Originate from the original input.",
                "If the plan is​ invalid, check whether it meets any of the following exceptions.",
                "Retroactive Variable Initialization Validity. It is a valid plan when a variable is referenced in an earlier phase of the workflow despite being formally initialized in subsequent warm-up/pre-training/finetuning steps. e.g., step 1 requires a warm-up embedding $V_{{warm}}$ as input, while it is generated in warm up training which placed in step 5.",
                "Derived Structures Validity. It is a valid plan when leveraging transformed data from the task input without any extra information. For instance, the input provides graph edges, it is permissible to utilize the graph adjacency matrix constructed from these edges without any other extra information.",
                "Tolerance of Implementation Mismatches. It is a valid plan when it contains technically mismatched but implementationally harmless details, such as dimensionality mismatch, data split not provided, or temporary variable renaming.",
            ]
        }

        for _ in range(retry):
            response = query(
                system_message=prompt,
                user_message=None,
                model=self.model,
                temperature=0.5,
            )
            think = extract_xml(response, "think")
            answer = extract_xml(response, "answer")
            if think and answer:
                logger.info(f"Plan validity check {answer} because: \n\n{think}\n\n")
                return "success" in answer.lower(), think
            else:
                logger.info("Plan validity check failed, retrying ...")

        logger.error("Plan validity check failed after {} retries".format(retry))
        return False, None

    def plan_paragraph_match(self, plan_json, paragraph=None):
        """
        Identifying the key concepts in the plan that require further explanation, and augmenting the plan with detailed descriptions from the provided papers.

        Args:
            plan_json[List]: the generated step-by-step plan that organized in a List format.
            paragraph[Dict]: the sections selected from paper, which will be used to augment the plan.

        Return:
            plan_json[List]: the augmented plan, which contains the "Related_description" field in each step.
        """

        is_require_context_prompt = """
        You are a ML expert.
        Here is the designed ML plan, and you need to select the key concepts in currect step that lack details.
        
        The procedures you can follow:
        S1: Identify the ML-related key concepts in this step(at most 5 words per concept), e.g., ML functions, calculations, ML algorithms.
        S2: For each concept, identify whether this concept meet these conditions: 1) this concept CANNOT be implemented in Python code, 2) it has not mentioned in previous teps, 3) it is specifically designed and unfamiliar to you.
        
        Please return YES and the concept list when all conditions are met in S2. Otherwise, please return NO.
        
        The response should be a JSON format:
        {{
            "answer":"YES/NO",
            "concept_list":[List of concept str],
        }}
        
        ML plan current step:{}
        Previous steps' plan: {}
        """

        # the inpuyt plan is a json file
        for key in plan_json.keys():
            if isinstance(plan_json[key], list):
                plan = plan_json[key]
        id2concept = {}
        for i in range(len(plan)):
            step = plan[i]
            response = query(
                user_message=is_require_context_prompt.format(step, plan[:i]),
                model=self.model,
            )
            try:
                response = extract_json(response)
                if "yes" in response["answer"].lower():
                    id2concept[i] = response["concept_list"]
            except:
                continue

        # give a analysis for each concept
        summary4concept_prompt = """
        
        You are a ML expert. 
        I have a plan and a set of paragraph. You need to given detailed sumamry for the concept based on the gibven paragraph.
        
        The response should be JSON format.
        {{
            "concept 1": "analysis of this concept, >=150 words",
            "concept 2": "analysis of this concept, >=150 words".
        }}
        
        The designed plan: {} \n\n
        The concept that requires explanation: {} \n\n
        The selected paragraph:{}
        
        """
        for step_id, cur_concept in id2concept.items():
            step_plan = plan[step_id]
            response = query(
                user_message=summary4concept_prompt.format(
                    step_plan, cur_concept, paragraph
                ),
                model=model,
            )
            try:
                response = extract_json(response)
                plan[step_id]["Related_description"] = response

            except:
                pass
        return plan

    @property
    def _prompt_abstract_plan_constraints(self):
        abstract_plan_constraints = [
            "Don't suggest to do model ensembling.",
            "Don't suggest to do Exploratory Data Analysis.",
            "Don't suggest to do hyperparameter tuning.",
            "DO NOT CHEAT ON EVALUATION. The solution should calculate the evaluation metric described in the task description on a hold-out validation set.",
            "If the evaluation metric is not provided, you should propose a reasonable evaluation metric for the task and calculate it.",
            "The solution should print the evaluation metric computed on the hold-out validation set at the last step of the solution.",
        ]
        return {"Abstract plan constraints": abstract_plan_constraints}

    def generate_abstract_plan(self, paper_analysis: dict, retry: int = 3):
        """
        Generate an abstract machine learning plan based on two selected papers and a given task.
        """

        prompt = {
            "Introduction": (
                "You are an expert machine learning engineer attempting a task. "
                "Please use the ideas in the following two papers to generate a new machine learning solution for the given task. "
                "We will now provide the task and the papers."
            )
        }

        prompt["Task description"] = self.task_desc

        if self.memory:
            prompt["Memory"] = (
                "Take the Memory section into consideration when proposing the solution plan, don't propose the similar solution but keep the evaluation metric exactlty the same."
                f"\n\n{self.memory}"
            )

        prompt["Paper analysis"] = json.dumps(paper_analysis, indent=4)
        prompt["Last try abstract plan"] = []

        if self.data_analysis:
            prompt["Data Analysis"] = self.data_analysis

        prompt["Instructions"] = {}
        prompt["Instructions"] |= {
            "Response format": """
            The response should be organzied in the following format. When describing a component in the solution (e.g., data processing, model architecture and training algorithm), first describe its input and output, and then describe each module in the component.

            # Key Module from the two papers: <= 150 words.
            
            # Module analysis: <= 200 words.

            # Merged solution: 
            ## Data processing
            ### Input:
            ### Output:
            ### module 1: ...
            ### module 2: ...

            ## Model architecture:
            ### ...

            ## Training algorithm:
            ### ..."""
        }
        prompt["Instructions"] |= self._prompt_abstract_plan_constraints
        prompt["Instructions"] |= {
            "Abstract plan guideline": [
                "The generated soulution must meet the abstract plan constraints above.",
                "The generated solution should have three components: (1) Data Processing (feature engineering, data augmentation and data split); (2) Model Architecture; (3) Training & Evaluation (training algorithms, mini-batch sampling, loss functions, evaluation metrics, etc).",
                "To generate the solution, you need to do the following processes step-by-step.",
                "Key Module: you need to identify the key techniques proposed by each paper.",
                "Module Analysis: you need to identify the modules that could be merged into one ML solution.",
                "Merging strategy: you need to merge the techniques identified above to develop a new solution for the given task. You need to justify the necessity of adopting the techniques in the solution (e.g., the relevance to the given task, significance of the technique.) You need to output the justification."
                "The generated solution should contain techniques from both papers."
                "Considering the simplicity, the merged strategy should balance between effectiveness and simplicity.",
            ]
        }

        abs_plan = None
        check_constraints_response = None
        constraints_check = True
        for _ in range(retry):
            if abs_plan:
                prompt["Last try abstract plan"].append(abs_plan)

            if not constraints_check:
                prompt["Last try abstract plan"].extend(
                    [
                        "The last try abstract plan does not meet all the constraints, please revise it according to the following check results.",
                        check_constraints_response,
                    ]
                )

            abs_plan = query(
                system_message=prompt,
                user_message=None,
                model=self.model,
                temperature=0.5,
            )

            constraints_check, check_constraints_response = self.check_constraints(
                plan=abs_plan,
                constraints=self._prompt_abstract_plan_constraints[
                    "Abstract plan constraints"
                ],
            )

            if constraints_check:
                logger.info(
                    f"Abstract plan generation successed. Proposed abstract plan: \n\n{abs_plan}\n\n"
                )
                return abs_plan
            else:
                logger.info("Abstract plan generation failed, retrying ...")
        logger.error(f"Abstract plan generation failed after {retry} retries")
        return None

    @property
    def _prompt_detailed_plan_constraints(self):
        detailed_plan_constraints = [
            "DO NOT save intermediate results.",
            "The test data must be loaded from test.csv, rather than splited from train.csv",
            "The solution should contain the test data prediction step or substep.",
            "Don't suggest to do model ensembling.",
            "Don't suggest to do Exploratory Data Analysis.",
            "Don't suggest to do hyperparameter tuning.",
            "DO NOT CHEAT ON EVALUATION. The solution should calculate the evaluation metric described in the task description on a hold-out validation set.",
            "If the evaluation metric is not provided, you should propose a reasonable evaluation metric for the task and calculate it.",
            "The solution should print the evaluation metric computed on the hold-out validation set at the last step of the solution.",
        ]
        return {"Detailed plan constraints": detailed_plan_constraints}

    def generate_detailed_plan(
        self, abs_plan: str, paper_analysis: str, retry: int = 3
    ):
        """
        Generate a detailed machine learning plan based on the abstract plan.
        """
        prompt = {
            "Introduction": (
                "You are an expert machine learning engineer attempting a task. "
                "You are provided an abstract plan designed by another engineer for the task. "
                "You should develop a detailed and complete plan by enriching this abstract plan with the two provided papers. "
                "We will now provide the task, the abstract plan and the papers."
            )
        }
        prompt["Task description"] = self.task_desc
        prompt["Abstract plan"] = abs_plan
        prompt["Paper analysis"] = paper_analysis

        if self.memory:
            prompt["Memory"] = (
                "Take the Memory section into consideration when proposing the solution plan, don't propose the similar solution but keep the evaluation metric exactlty the same."
                f"\n\n{self.memory}"
            )

        if self.data_analysis:
            prompt["Data Analysis"] = self.data_analysis

        prompt["Last try detailed plan"] = []

        prompt["Instructions"] = {}
        prompt["Instructions"] |= {
            "Response format": [
                "Your response should be a single JSON code block (wrapped in ```) which contains the steps of the detailed plan. "
                """The generated JSON should have the following format:
                {
                    {
                        "steps":"step name",
                        "input": "input of this step (files on hardware, variables mentioned in previous steps)",
                        "output": "output of this step"
                        "substeps":[
                            Describe each (sub)step in precisely and clearly, as detailed as possible.
                        ]
                    },
                }
                """
            ]
        }
        prompt["Instructions"] |= self._prompt_detailed_plan_constraints
        prompt["Instructions"] |= {
            "Detailed plan guideline": [
                "The generated soulution must meet the detailed plan constraints above.",
                "You should identify the task input (files).",
                "You should devide the abstract plan into distinct steps based on functionality, i.e., each step MUST be evaluated based on previous steps. If some steps cannot be evaluated, just merged into other steps.",
                "For each step, you MUST list its input variab(all the used les should be pointed out), processing procedures, and list all the outputs that may used in the following steps.",
                "All the learnable parameters/embeddings should be optimized with loss. Please check the optimizer that all the parameters/embeddings are added.",
                "Use **formal notation** in the plan. Variables should have new name if it is updated. Consistent with the notations used in **academic papers**.",
                "Your wording should be **precise**, and the terminology should be consistent throughout the context.",
            ]
        }

        detailed_plan = None
        check_constraints_response = None
        constraints_check = True
        check_validity_response = None
        validity_check = True
        for _ in range(retry):
            if detailed_plan:
                prompt["Last try detailed plan"].append(detailed_plan)

            if not constraints_check:
                prompt["Last try detailed plan"].extend(
                    [
                        "The last try detailed plan does not meet all the constraints, please revise it according to the following check results.",
                        check_constraints_response,
                    ]
                )

            if not validity_check:
                prompt["Last try detailed plan"].extend(
                    [
                        "The last try detailed plan is not valid for the task, please revise it according to the following check results.",
                        check_validity_response,
                    ]
                )

            response = query(
                system_message=prompt,
                user_message=None,
                model=self.model,
                temperature=0.5,
            )

            detailed_plan = extract_json(response)
            if not detailed_plan:
                logger.info("Detailed plan generation failed, retrying ...")
                continue

            constraints_check, check_constraints_response = self.check_constraints(
                plan=json.dumps(detailed_plan),
                constraints=self._prompt_detailed_plan_constraints[
                    "Detailed plan constraints"
                ],
            )
            validity_check, check_validity_response = self.check_validity(
                plan=json.dumps(detailed_plan),
            )

            if constraints_check and validity_check:
                logger.info(
                    f"Detailed plan generation successed. Proposed detailed plan: \n\n{detailed_plan}\n\n"
                )
                return detailed_plan
            else:
                logger.info("Detailed plan generation failed, retrying ...")
        logger.error(f"Detailed plan generation failed after {retry} retries")
        return None

    def step(
        self,
        papers: dict,
        memory: Optional[str] = None,
        knowledge: Optional[str] = None,
        data_analysis: Optional[str] = None,
    ):
        """
        Args:
            papers[dict]: the selected papers, each key is the paper file path, and the value is the selected relative sections from the paper.
            memory (Optional[str]): the memory of the agent, which is a string containing the previous plans and results.
            knowledge (Optional[str]): the knowledge of the agent, which is a string containing the tricks and tips for the task.
            data_analysis (Optional[str]): the data analysis of the agent, which is a string containing the analysis of the data.
        Returns:
            dict: the generation status and the generated plan.
        """

        self.memory = memory
        self.knowledge = knowledge
        self.data_analysis = data_analysis

        # Step 1: Check if the retrieved papers are mergeable
        logger.info("Agent is checking the mergeability of the retrieved papers")
        is_mergeable, response = self.check_mergeability(papers)
        if not is_mergeable:
            logger.info(
                "Retrieved papers are not mergeable, agent is generating a plan independently"
            )
            return super().step(
                memory=memory,
                knowledge=knowledge,
                data_analysis=data_analysis,
            )

        # Step 2: Generate paper analysis
        logger.info("Agent is analyzing the retrieved papers")
        paper_analysis = self.get_paper_analysis(papers)

        # Step 3: Generate abstract plan
        logger.info("Agent is generating the abstract plan")
        abs_plan = self.generate_abstract_plan(
            paper_analysis=wrap_code(json.dumps(paper_analysis))
        )
        if not abs_plan:
            logger.error("Abstract plan generation failed")
            return super().step(
                memory=memory,
                knowledge=knowledge,
                data_analysis=data_analysis,
            )

        # Step 3: Generate detailed plan
        logger.info("Agent is generating the detailed plan")
        detailed_plan = self.generate_detailed_plan(
            abs_plan=abs_plan, paper_analysis=wrap_code(json.dumps(paper_analysis))
        )
        if not detailed_plan:
            logger.error("Detailed plan generation failed")
            return super().step(
                memory=memory,
                knowledge=knowledge,
                data_analysis=data_analysis,
            )

        # Step 4: Augment the detailed plan with paragraph match
        logger.info("Agent is augmenting the detailed plan with paragraph match")
        augmented_plan = self.plan_paragraph_match(detailed_plan, paragraph=papers)
        if not augmented_plan:
            logger.error("Augmented plan generation failed")
            return super().step(
                memory=memory,
                knowledge=knowledge,
                data_analysis=data_analysis,
            )

        return json.dumps(augmented_plan)
