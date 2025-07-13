import sys
import shutil
import logging
import random
import time
from typing import Optional
from enum import Enum
import humanize
from omegaconf import OmegaConf
import json

from .trajectory import Trajectory, Node
from .utils.config import Config
from .utils.metric import MetricValue, WorstMetricValue
from .utils.response import wrap_code

from .knowledge_retriever import TricksRetriever, PapersRetriever
from .data_analyzer import DataPreviewer
from .planner import OneShotPlanner, KnowledgePlanner
from .coder import OneShotCoder, StepByStepCoder, MixtureCoder
from .improver import OneShotImprover
from .interpreter import Interpreter, ExecutionResult
from .verifier import SubmissionVerifier

logger = logging.getLogger("automind")


def format_time(time_in_sec: int):
    return f"{time_in_sec // 3600}hrs {(time_in_sec % 3600) // 60}mins {time_in_sec % 60}secs"


class Action(Enum):
    DRAFT = "draft"
    IMPROVE = "improve"
    DEBUG = "debug"
    HPO = "hpo"
    VERIFY = "verify"


class Agent:
    def __init__(
        self,
        task_desc: str,
        cfg: Config,
        traj: Trajectory,
    ):
        super().__init__()
        self.task_desc = task_desc
        self.cfg = cfg
        self.acfg = cfg.agent
        self.traj = traj
        self.start_time = time.time()
        self.current_step = 0

        self._retrieved_papers = None
        self._retrieved_tricks = None
        self._data_analysis: str = None

        # Set up the interpreter based on the config
        self.interpreter = Interpreter(
            cfg.workspace_dir, **OmegaConf.to_container(cfg.exec)
        )

        # Set up the knowledge retriever based on the config
        if self.acfg.retriever.strategy == "papers + tricks":
            self.papers_retriever = PapersRetriever(
                model=self.acfg.retriever.model,
                task_desc=self.task_desc,
            )
            self.tricks_retriever = TricksRetriever(
                model=self.acfg.retriever.model,
                task_desc=self.task_desc,
            )
        elif self.acfg.retriever.strategy == "tricks":
            self.tricks_retriever = TricksRetriever(
                model=self.acfg.retriever.model,
                task_desc=self.task_desc,
            )
            self.papers_retriever = None
        elif self.acfg.retriever.strategy == "papers":
            self.papers_retriever = PapersRetriever(
                model=self.acfg.retriever.model,
                task_desc=self.task_desc,
            )
            self.tricks_retriever = None
        else:
            if self.acfg.retriever.strategy != "none":
                logger.warning(
                    f"Unknown retriever strategy: {self.acfg.retriever.strategy}, valid options are 'papers + tricks', 'tricks', 'papers'"
                )
            self.papers_retriever = None
            self.tricks_retriever = None

        if self.papers_retriever:
            assert (
                self.acfg.planner.strategy == "knowledge"
            ), f"Papers retriever should be used with knowledge planner, but {self.acfg.planner.strategy} planner is used"
        else:
            assert (
                self.acfg.planner.strategy != "knowledge"
            ), f"Knowledge planner should be used with papers retriever, but {self.acfg.retriever.strategy} retriever is used"

        # Set up the data analyzer based on the config
        if self.acfg.analyzer.strategy == "preview":
            self.analyzer = DataPreviewer(
                model=self.acfg.analyzer.model,
                task_desc=self.task_desc,
                data_dir=self.cfg.workspace_dir,
            )
        else:
            logger.warning(
                f"Unknown analyzer strategy: {self.acfg.analyzer.strategy}, valid options are 'preview'"
            )
            self.analyzer = None

        # Set up the planner based on the config
        if self.acfg.planner.strategy == "one-shot":
            self.planner = OneShotPlanner(
                model=self.acfg.planner.model,
                task_desc=self.task_desc,
            )
        elif self.acfg.planner.strategy == "knowledge":
            self.planner = KnowledgePlanner(
                model=self.acfg.planner.model,
                task_desc=self.task_desc,
            )
        else:
            raise ValueError(
                f"Unknown planner strategy: {self.acfg.planner.strategy}, valid options are 'one-shot'"
            )

        # Set up the coder based on the config
        if self.acfg.coder.strategy == "one-shot":
            self.coder = OneShotCoder(
                model=self.acfg.coder.model,
                task_desc=self.task_desc,
                interpreter=self.interpreter,
                human_guideline=self._prompt_impl_guideline,
            )
        elif self.acfg.coder.strategy == "step-by-step":
            self.coder = StepByStepCoder(
                model=self.acfg.coder.model,
                task_desc=self.task_desc,
                interpreter=self.interpreter,
                human_guideline=self._prompt_impl_guideline,
            )
        elif self.acfg.coder.strategy == "mixture":
            self.coder = MixtureCoder(
                model=self.acfg.coder.model,
                task_desc=self.task_desc,
                interpreter=self.interpreter,
                human_guideline=self._prompt_impl_guideline,
            )
        else:
            raise ValueError(
                f"Unknown coder strategy: {self.acfg.coder.strategy}, valid options are 'one-shot', 'step-by-step'"
            )

        # Set up the improver based on the config
        if self.acfg.improver.strategy == "one-shot":
            self.improver = OneShotImprover(
                model=self.acfg.improver.model,
                task_desc=self.task_desc,
            )
        else:
            raise ValueError(
                f"Unknown improver strategy: {self.acfg.improver.strategy}, valid options are 'one-shot'"
            )

        # Set up the verifier based on the config
        if self.acfg.verifier.strategy == "submission":
            self.verifier = SubmissionVerifier(
                model=self.acfg.verifier.model,
                task_desc=self.task_desc,
            )
        else:
            raise ValueError(
                f"Unknown verifier strategy: {self.acfg.verifier.strategy}, valid options are 'submission'"
            )

    def search_policy(self) -> tuple[Optional[Node], Action]:
        """Select a node to work on (or None to draft a new node)."""
        search_cfg = self.acfg.search

        # initial drafting
        if len(self.traj.draft_nodes) < search_cfg.num_drafts:
            logger.info("[search policy] drafting new node (not enough drafts)")
            return None, Action.DRAFT

        # debugging
        if random.random() < search_cfg.debug_prob:
            # nodes that are buggy + leaf nodes + debug depth < max debug depth
            debuggable_nodes = [
                n
                for n in self.traj.buggy_nodes
                if (n.is_leaf and n.debug_depth < search_cfg.max_debug_depth)
            ]
            if debuggable_nodes:
                node_to_debug = random.choice(debuggable_nodes)
                logger.info(f"[search policy] debugging node {node_to_debug.id}")
                return node_to_debug, Action.DEBUG

        # back to drafting if no nodes to improve
        good_nodes = self.traj.good_nodes
        if not good_nodes:
            logger.info("[search policy] drafting new node (no good nodes)")
            return None, Action.DRAFT

        # HPO
        sub_best_nodes = self.traj.get_sub_best_nodes_per_tree()
        if random.random() < min(
            search_cfg.max_hpo_prob, search_cfg.hpo_prob * len(sub_best_nodes)
        ):
            if sub_best_nodes:
                node_to_hpo = random.choice(sub_best_nodes)
                logger.info(
                    f"[search policy] improving node {node_to_hpo.id} through HPO"
                )
                return node_to_hpo, Action.HPO

        # greedy
        if random.random() < search_cfg.greedy_prob:
            greedy_node = self.traj.get_best_node()
            logger.info(f"[search policy] greedy node selected: node {greedy_node.id}")
            return greedy_node, Action.IMPROVE
        else:
            sub_best_nodes = self.traj.get_sub_best_nodes_per_tree()
            if sub_best_nodes:
                non_greedy_node = random.choice(sub_best_nodes)
                logger.info(
                    f"[search policy] non greedy node selected: node {non_greedy_node.id}"
                )
                return non_greedy_node, Action.IMPROVE

    @property
    def _prompt_impl_guideline(self):
        tot_time_elapsed = time.time() - self.start_time
        tot_time_remaining = int(self.acfg.time_limit - tot_time_elapsed)
        exec_timeout = int(min(self.cfg.exec.timeout, tot_time_remaining))

        impl_guideline = [
            f"Be aware of the running time of the code, it should complete within {humanize.naturaldelta(exec_timeout)}.",
        ]
        if self.acfg.expose_prediction:
            impl_guideline.append(
                "The implementation should include a predict() function, "
                "allowing users to seamlessly reuse the code to make predictions on new data. "
                "The prediction function should be well-documented, especially the function signature."
            )

        if self.acfg.k_fold_validation > 1:
            impl_guideline.append(
                f"The evaluation should be based on {self.acfg.k_fold_validation}-fold cross-validation but only if that's an appropriate evaluation for the task at hand."
            )

        return {"Implementation guideline": impl_guideline}

    def _draft(self) -> Node:
        local_memory = self.traj.generate_summary()
        if self.acfg.planner.strategy == "knowledge" and self._retrieved_papers:
            plan = self.planner.step(
                papers=dict(random.sample(list(self._retrieved_papers.items()), 2)),
                memory=local_memory,
                data_analysis=self._data_analysis if self.analyzer else None,
            )
        else:
            plan = self.planner.step(
                memory=local_memory,
                data_analysis=self._data_analysis if self.analyzer else None,
            )
        code, exec_result = self.coder.step(
            plan=plan, data_analysis=self._data_analysis if self.analyzer else None
        )
        new_node = Node(plan=plan, code=code, applied_tricks=[], applying_trick_idx=-1)
        new_node.absorb_exec_result(exec_result)
        return new_node

    def _improve(self, parent_node: Node) -> Node:
        local_memory = self.traj.generate_summary(include_code=False, max_nodes=10)
        search_cfg = self.acfg.search
        knowledge = None
        applying_trick_idx = -1
        # tricks retriever
        if self.tricks_retriever and self._retrieved_tricks:
            if random.random() < search_cfg.trick_prob:
                # apply tricks not traversed
                if len(parent_node.applied_tricks) < len(self._retrieved_tricks):
                    unapplied_tricks = [
                        (i, d)
                        for i, d in enumerate(self._retrieved_tricks)
                        if i not in parent_node.applied_tricks
                    ]
                    valid = [
                        (i, d)
                        for i, d in unapplied_tricks
                        if not d.get("invalid", False)
                    ]
                    # valid tricks exist
                    if valid:
                        choose_knowledge = random.choice(valid)
                        parent_node.applied_tricks.append(choose_knowledge[0])
                        knowledge = choose_knowledge[1].get("content", None)
                        applying_trick_idx = choose_knowledge[0]
                    # random choose from invalid tricks or none
                    else:
                        choose_knowledge = random.choice(
                            [random.choice(unapplied_tricks), (-1, None)]
                        )
                        if choose_knowledge[1] is not None:
                            parent_node.applied_tricks.append(choose_knowledge[0])
                            knowledge = choose_knowledge[1].get("content", None)
                            applying_trick_idx = choose_knowledge[0]

        if knowledge is not None:
            logger.info(
                f"Agent is applying trick to improve node {parent_node.id}. Retrieved tricks: \n\n{knowledge}"
            )
            knowledge = json.dumps(knowledge)

        improved_plan, applied_knowledge = self.improver.step(
            plan=parent_node.plan,
            code=wrap_code(parent_node.code),
            exec_output=wrap_code(parent_node.term_out, lang=""),
            is_buggy=False,
            is_hpo=False,
            memory=local_memory,
            data_analysis=self._data_analysis if self.analyzer else None,
            knowledge=knowledge,
        )
        # check if actually applied
        applying_trick_idx = applying_trick_idx if applied_knowledge is not None else -1

        if self.acfg.coder.strategy == "mixture":
            improved_code, exec_result = self.coder.step(
                plan=improved_plan,
                data_analysis=self._data_analysis if self.analyzer else None,
                based_code=wrap_code(parent_node.code),
                mode="one-shot",
            )
        else:
            improved_code, exec_result = self.coder.step(
                plan=improved_plan,
                data_analysis=self._data_analysis if self.analyzer else None,
                based_code=wrap_code(parent_node.code),
            )
        new_node = Node(
            plan=improved_plan,
            code=improved_code,
            parent=parent_node,
            improve_parent=parent_node,
            applied_tricks=parent_node.applied_tricks,
            applying_trick_idx=applying_trick_idx,
        )
        new_node.absorb_exec_result(exec_result)
        return new_node

    def _hpo(self, parent_node: Node) -> Node:
        local_memory = self.traj.generate_summary(include_code=False, max_nodes=10)
        improved_plan, _ = self.improver.step(
            plan=parent_node.plan,
            code=wrap_code(parent_node.code),
            exec_output=wrap_code(parent_node.term_out, lang=""),
            is_buggy=False,
            is_hpo=True,
            memory=local_memory,
            data_analysis=self._data_analysis if self.analyzer else None,
        )
        if self.acfg.coder.strategy == "mixture":
            improved_code, exec_result = self.coder.step(
                plan=improved_plan,
                data_analysis=self._data_analysis if self.analyzer else None,
                based_code=wrap_code(parent_node.code),
                mode="one-shot",
            )
        else:
            improved_code, exec_result = self.coder.step(
                plan=improved_plan,
                data_analysis=self._data_analysis if self.analyzer else None,
                based_code=wrap_code(parent_node.code),
            )
        new_node = Node(
            plan=improved_plan,
            code=improved_code,
            parent=parent_node,
            improve_parent=parent_node,
            applied_tricks=parent_node.applied_tricks,
            applying_trick_idx=-1,
        )
        new_node.absorb_exec_result(exec_result)
        return new_node

    def _debug(self, parent_node: Node) -> Node:
        debugged_plan, _ = self.improver.step(
            plan=parent_node.plan,
            code=wrap_code(parent_node.code),
            exec_output=wrap_code(parent_node.term_out, lang=""),
            is_buggy=True,
            is_hpo=False,
            data_analysis=self._data_analysis if self.analyzer else None,
        )
        debugged_code, exec_result = self.coder.step(
            plan=debugged_plan,
            data_analysis=self._data_analysis if self.analyzer else None,
            based_code=wrap_code(parent_node.code),
        )
        new_node = Node(
            plan=debugged_plan,
            code=debugged_code,
            parent=parent_node,
            improve_parent=parent_node.improve_parent,
            applied_tricks=parent_node.applied_tricks,
            applying_trick_idx=parent_node.applying_trick_idx,
        )
        new_node.absorb_exec_result(exec_result)
        return new_node

    def _verify(
        self, node: Node, exec_result: Optional[ExecutionResult] = None
    ) -> Node:
        if exec_result is not None:
            node.absorb_exec_result(exec_result)

        response = self.verifier.step(
            code=wrap_code(node.code),
            exec_output=wrap_code(node.term_out, lang=""),
        )

        # do an extra check, to catch cases where verifier fails
        has_csv_submission = (
            self.cfg.workspace_dir / "submission" / "submission.csv"
        ).exists()

        node.analysis = response["summary"]
        node.is_buggy = (
            response["is_bug"]
            or response["is_overfitting"]
            or node.exc_type is not None
            or response["metric"] is None
            or response["has_csv_submission"] == False
            or has_csv_submission == False
        )
        # get metric
        if response["is_bug"] or node.exc_type is not None:
            logger.info(
                f"Parsed results: Node {node.id} is buggy for code execution failure or raised exception"
            )
            node.is_buggy = True
            node.metric = WorstMetricValue()
        elif response["is_overfitting"]:
            logger.info(f"Parsed results: Node {node.id} is buggy for overfitting")
            node.is_buggy = True
            node.metric = WorstMetricValue()
        elif response["metric"] is None:
            logger.info(
                f"Parsed results: Node {node.id} is buggy for metric extraction failure"
            )
            node.is_buggy = True
            node.metric = WorstMetricValue()
        elif response["has_csv_submission"] == False or has_csv_submission == False:
            logger.info(
                f"Parsed results: Node {node.id} is buggy for missing submission.csv"
            )
            node.is_buggy = True
            node.metric = WorstMetricValue()
        else:
            logger.info(f"Parsed results: Node {node.id} is not buggy")
            node.is_buggy = False
            node.metric = MetricValue(
                response["metric"], maximize=not response["lower_is_better"]
            )

        # update tricks validity
        if node.is_buggy == False and self.tricks_retriever and self._retrieved_tricks:
            if node.applying_trick_idx != -1 and node.improve_parent is not None:
                if self.traj.get_better_node(node.improve_parent, node) != node:
                    self._retrieved_tricks[node.applying_trick_idx]["invalid"] = True
                    logger.info(
                        f"The retrieved trick {node.applying_trick_idx} is probably invalid. Diminishing its weight..."
                    )
                else:
                    logger.info(
                        f"The retrieved trick {node.applying_trick_idx} is proved valid."
                    )

        return node

    def step(self):
        tot_time_elapsed = time.time() - self.start_time
        tot_time_remaining = int(self.acfg.time_limit - tot_time_elapsed)
        logger.info(f"Step {self.current_step + 1} of {self.acfg.steps}")
        logger.info(f"Time remaining: {format_time(tot_time_remaining)}")

        working_submission_dir = self.cfg.workspace_dir / "submission"
        # clear the submission dir from previous steps
        shutil.rmtree(working_submission_dir, ignore_errors=True)
        # create a new submission directory for this step if it doesn't exist
        working_submission_dir.mkdir(exist_ok=True)

        if self.papers_retriever and not self._retrieved_papers:
            logger.info("Agent is retrieving papers")
            self._retrieved_papers = self.papers_retriever.step()
            logger.info(f"Agent retrieved papers successed")

        if self.tricks_retriever and not self._retrieved_tricks:
            logger.info("Agent is retrieving tricks")
            self._retrieved_tricks = self.tricks_retriever.step()
            logger.info(f"Agent retrieved tricks: \n\n{self._retrieved_tricks}\n\n")

        if self.analyzer and not self._data_analysis:
            logger.info("Agent is analyzing data")
            self._data_analysis = self.analyzer.step()
            logger.info(f"Agent analyzed data: \n\n{self._data_analysis}\n\n")

        parent_node, action = self.search_policy()

        if parent_node is None or action == Action.DRAFT:
            logger.info("Agent is drafting a new node")
            result_node = self._draft()
            logger.info(f"Drafted new node {result_node.id}")
        elif parent_node.is_buggy or action == Action.DEBUG:
            logger.info(f"Agent is debugging node {parent_node.id}")
            result_node = self._debug(parent_node)
            logger.info(
                f"Debugged node {parent_node.id} to create new node {result_node.id}"
            )
        elif action == Action.HPO:
            logger.info(f"Agent is improving node {parent_node.id} through HPO")
            result_node = self._hpo(parent_node)
            logger.info(
                f"Improved node {parent_node.id} to create new node {result_node.id} through HPO"
            )
        else:
            logger.info(f"Agent is improving node {parent_node.id}")
            result_node = self._improve(parent_node)
            logger.info(
                f"Improved node {parent_node.id} to create new node {result_node.id}"
            )

        logger.info(f"Agent is verifying the results of node {result_node.id}")
        result_node = self._verify(result_node)

        # save the result node to the trajectory
        self.traj.append(result_node)
        self.current_step += 1

        # create best submission and solution directories if they don't exist
        best_submission_dir = self.cfg.workspace_dir / "best_submission"
        best_submission_dir.mkdir(exist_ok=True, parents=True)
        best_solution_dir = self.cfg.workspace_dir / "best_solution"
        best_solution_dir.mkdir(exist_ok=True, parents=True)

        # if the result_node is the best node, cache its submission.csv and solution.py
        # to best_solution/ by copying it there
        best_node = self.traj.get_best_node()
        if best_node is not None:
            if best_node.id == result_node.id:
                logger.info(f"Node {result_node.id} is the best node")
                # copy submission/submission.csv to best_submission/submission.csv
                shutil.copy(
                    working_submission_dir / "submission.csv",
                    best_submission_dir,
                )
                # copy solution.py and relevant node info to best_solution/
                with open(best_solution_dir / "solution.py", "w") as f:
                    f.write(result_node.code)
                with open(best_solution_dir / "node_info.txt", "w") as f:
                    f.write(
                        f"node_id: {str(result_node.id)}\n\nmetric: {str(result_node.metric)}\n\nsolution:\n{result_node.plan}"
                    )
                # DEBUG: once get submission then stop
                # self.current_step = self.acfg.steps
            else:
                logger.info(
                    f"Node {result_node.id} is not the best node, while Node {best_node.id} is still the best node"
                )
        else:
            logger.error("No best node found")

        # create sub best submission and solution directories if they don't exist
        root_node = self.traj.get_root_node(result_node.id)
        logger.info(f"Node {root_node.id} is the root node of node {result_node.id}")
        sub_best_submission_dir = best_submission_dir / f"subtree_{root_node.id}"
        sub_best_submission_dir.mkdir(exist_ok=True, parents=True)
        sub_best_solution_dir = best_solution_dir / f"subtree_{root_node.id}"
        sub_best_solution_dir.mkdir(exist_ok=True, parents=True)

        # if the result_node is the sub best node, cache its submission.csv and solution.py
        # to best_solution/ by copying it there
        sub_best_node = self.traj.get_sub_best_node(root_node.id)
        if sub_best_node is not None:
            if sub_best_node.id == result_node.id:
                logger.info(
                    f"Node {result_node.id} is the sub best node for subtree {root_node.id}"
                )
                # copy submission/submission.csv to sub_best_submission/submission.csv
                shutil.copy(
                    working_submission_dir / "submission.csv",
                    sub_best_submission_dir,
                )
                # copy solution.py and relevant node info to sub_best_solution/
                with open(sub_best_solution_dir / "solution.py", "w") as f:
                    f.write(result_node.code)
                with open(sub_best_solution_dir / "node_info.txt", "w") as f:
                    f.write(
                        f"node_id: {str(result_node.id)}\n\nmetric: {str(result_node.metric)}\n\nsolution: {result_node.plan}"
                    )
            else:
                logger.info(
                    f"Node {result_node.id} is not the sub best node, while Node {sub_best_node.id} is still the sub best node for subtree {root_node.id}"
                )
        else:
            logger.error("No sub best node found")

        return result_node
