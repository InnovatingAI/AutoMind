import copy
import time
import uuid
import logging
from dataclasses import dataclass, field
from typing import Literal, Optional
from dataclasses_json import DataClassJsonMixin
from rich.tree import Tree
from .interpreter import ExecutionResult
from .utils.metric import MetricValue
from .utils.response import trim_long_string

logger = logging.getLogger("automind")


@dataclass(eq=False)
class Node(DataClassJsonMixin):
    """A single node in the solution tree. Contains code, execution results, and evaluation information."""

    # ---- code & plan ----
    code: str
    plan: str = field(default=None, kw_only=True)  # type: ignore

    # ---- general attrs ----
    step: int = field(default=None, kw_only=True)  # type: ignore
    id: str = field(default_factory=lambda: uuid.uuid4().hex, kw_only=True)
    ctime: float = field(default_factory=lambda: time.time(), kw_only=True)
    parent: Optional["Node"] = field(default=None, kw_only=True)
    improve_parent: Optional["Node"] = field(default=None, kw_only=True)
    children: set["Node"] = field(default_factory=set, kw_only=True)

    # ---- execution info ----
    _term_out: list[str] = field(default=None, kw_only=True)  # type: ignore
    exec_time: float = field(default=None, kw_only=True)  # type: ignore
    exc_type: str | None = field(default=None, kw_only=True)
    exc_info: dict | None = field(default=None, kw_only=True)
    exc_stack: list[tuple] | None = field(default=None, kw_only=True)
    applied_tricks: list = field(default=None, kw_only=True)
    applying_trick_idx: int = field(default=None, kw_only=True)  # type: ignore

    # ---- evaluation ----
    # post-execution result analysis (findings/feedback)
    analysis: str = field(default=None, kw_only=True)  # type: ignore
    metric: MetricValue = field(default=None, kw_only=True)  # type: ignore
    # whether the agent decided that the code is buggy
    # -> always True if exc_type is not None or no valid metric
    is_buggy: bool = field(default=None, kw_only=True)  # type: ignore

    def __post_init__(self) -> None:
        if self.parent is not None:
            self.parent.children.add(self)

    @property
    def stage_name(self) -> Literal["draft", "debug", "improve"]:
        """
        Return the stage of the node:
        - "stage" if the node is an initial solution draft
        - "debug" if the node is the result of a debugging step
        - "improve" if the node is the result of an improvement step
        """
        if self.parent is None:
            return "draft"
        return "debug" if self.parent.is_buggy else "improve"

    def absorb_exec_result(self, exec_result: ExecutionResult):
        """Absorb the result of executing the code from this node."""
        self._term_out = exec_result.term_out
        self.exec_time = exec_result.exec_time
        self.exc_type = exec_result.exc_type
        self.exc_info = exec_result.exc_info
        self.exc_stack = exec_result.exc_stack

    @property
    def term_out(self) -> str:
        """Get the terminal output of the code execution (after truncating it)."""
        return trim_long_string("".join(self._term_out))

    @property
    def is_leaf(self) -> bool:
        """Check if the node is a leaf node in the solution tree."""
        return not self.children

    def __eq__(self, other):
        return isinstance(other, Node) and self.id == other.id

    def __hash__(self):
        return hash(self.id)

    @property
    def debug_depth(self) -> int:
        """
        Length of the current debug path
        - 0 if the node is not a debug node (parent is not buggy)
        - 1 if the parent is buggy but the skip parent isn't
        - n if there were n consecutive debugging steps
        """
        if self.stage_name != "debug":
            return 0
        return self.parent.debug_depth + 1  # type: ignore


@dataclass
class Trajectory(DataClassJsonMixin):
    """A collection of nodes representing the solution tree."""

    nodes: list[Node] = field(default_factory=list)

    def __getitem__(self, idx: int) -> Node:
        return self.nodes[idx]

    def __len__(self) -> int:
        """Return the number of nodes in the trajectory."""
        return len(self.nodes)

    def append(self, node: Node) -> None:
        """Append a new node to the trajectory."""
        node.step = len(self.nodes)
        self.nodes.append(node)

    @property
    def draft_nodes(self) -> list[Node]:
        """Return a list of nodes representing intial coding drafts"""
        return [n for n in self.nodes if n.parent is None]

    @property
    def buggy_nodes(self) -> list[Node]:
        """Return a list of nodes that are considered buggy by the agent."""
        return [n for n in self.nodes if n.is_buggy]

    @property
    def good_nodes(self) -> list[Node]:
        """Return a list of nodes that are not considered buggy by the agent."""
        return [n for n in self.nodes if not n.is_buggy]

    def get_metric_history(self) -> list[MetricValue]:
        """Return a list of all metric values in the trajectory."""
        return [n.metric for n in self.nodes]

    def get_node_by_id(self, node_id: str) -> Node | None:
        """Return a node by its ID."""
        for node in self.nodes:
            if node.id == node_id:
                return node
        logger.error(f"Node with ID {node_id} not found")
        return None

    def get_best_node(self, only_good=True) -> None | Node:
        """Return the best solution found so far (node with the highest validation metric)."""
        if only_good:
            nodes = self.good_nodes
        else:
            nodes = self.nodes

        if not nodes:
            return None

        max_metric_nodes = [n for n in nodes if n.metric.maximize]
        min_metric_nodes = [n for n in nodes if not n.metric.maximize]
        filtered_nodes = (
            max_metric_nodes
            if len(max_metric_nodes) > len(min_metric_nodes)
            else min_metric_nodes
        )

        try:
            return max(filtered_nodes, key=lambda n: n.metric)
        except Exception as e:
            logger.error(f"Metric comparison error: {e}")
            return nodes[0]

    def generate_summary(self, include_code: bool = False, max_nodes: int = -1) -> str:
        """Generate a summary of the trajectory for the agent."""
        summary = []
        # Add all draft nodes in the memory, whether they are buggy or not
        for n in self.draft_nodes:
            summary_part = "\n----------------------------------\n"
            summary_part += f"Design: {n.plan}\n"
            if include_code:
                summary_part += f"Code: {n.code}\n"
            if n.is_buggy:
                summary_part += f"Results: {n.analysis}\n"
                summary_part += f"This is a plan that failed to implement correctly. Please try to propose a different plan.\n"
            else:
                summary_part += f"Results: {n.analysis}\n"
                summary_part += f"Validation Metric: {n.metric.value}\n"
            summary.append(summary_part)

        # Add the latest good nodes in the memory
        if max_nodes > 0 and len(self.good_nodes) > max_nodes:
            good_nodes = self.good_nodes[-max_nodes:]
        else:
            good_nodes = self.good_nodes
        for n in good_nodes:
            if n not in self.draft_nodes:
                summary_part = f"Design: {n.plan}\n"
                if include_code:
                    summary_part += f"Code: {n.code}\n"
                summary_part += f"Results: {n.analysis}\n"
                summary_part += f"Validation Metric: {n.metric.value}\n"
                summary.append(summary_part)

        # Add the best node in the memory
        best_node = self.get_best_node(only_good=True)
        if best_node and best_node not in good_nodes:
            summary_part = f"Best Design: {best_node.plan}\n"
            if include_code:
                summary_part += f"Code: {best_node.code}\n"
            summary_part += f"Results: {best_node.analysis}\n"
            summary_part += f"Validation Metric: {best_node.metric.value}\n"
            summary.append(summary_part)

        return "\n-------------------------------\n".join(summary)

    def to_string_tree(self) -> str:
        """Return a string representation of the solution tree."""
        best_node = self.get_best_node()
        tree_str = "Solution tree\n"

        def append_rec(node: Node, level: int):
            nonlocal tree_str
            indent = "  " * level
            if node.is_buggy or not node.metric:
                s = f"{indent}◍ bug (ID: {node.id})\n"
            else:
                # support for multiple markers; atm only "best" is supported
                markers = []
                if node is best_node:
                    markers.append("best")
                marker_str = " & ".join(markers)
                if marker_str:
                    s = f"{indent}● {node.metric.value:.4f} ({marker_str}) (ID: {node.id})\n"
                else:
                    s = f"{indent}● {node.metric.value:.4f} (ID: {node.id})\n"
            tree_str += s
            for child in node.children:
                append_rec(child, level + 1)

        for n in self.draft_nodes:
            append_rec(n, 0)

        return tree_str

    def to_rich_tree(self):
        best_node = self.get_best_node()

        def append_rec(node: Node, tree):
            if node.is_buggy:
                s = f"[red]◍ bug (ID: {node.id})"
            else:
                style = "bold " if node is best_node else ""

                if node is best_node:
                    s = f"[{style}green]● {node.metric.value:.3f} (best) (ID: {node.id})"
                else:
                    s = f"[{style}green]● {node.metric.value:.3f} (ID: {node.id})"

            subtree = tree.add(s)
            for child in node.children:
                append_rec(child, subtree)

        tree = Tree("[bold blue]Solution tree")
        for n in self.draft_nodes:
            append_rec(n, tree)
        return tree

    def get_path_to_node(self, node_id: str) -> list[str]:
        path = [node_id]

        node2parent = {n.id: n.parent.id for n in self.nodes if n.parent is not None}
        while node_id in node2parent:
            parent_id = node2parent[node_id]
            path.append(parent_id)
            node_id = parent_id
        return path[::-1]

    def get_root_node(self, node_id: str) -> Node:
        """
        Get the root node of the subtree containing the node with the given ID.
        This is the first node in the path to the root that has no parent.
        """
        path = self.get_path_to_node(node_id)
        return self.get_node_by_id(path[0])

    def get_longest_path(self) -> list[str]:
        longest_path = []
        for node in self.nodes:
            path = self.get_path_to_node(node.id)
            if len(path) > len(longest_path):
                longest_path = path
        return longest_path

    def filter_on_path(self, path: list[str]):
        traj_copy = copy.deepcopy(self)
        traj_copy.nodes = [copy.deepcopy(n) for n in self.nodes if n.id in path]
        # further filter nodes, setting their _term_out and exc_stack to "<OMITTED>"
        for n in traj_copy.nodes:
            n._term_out = "<OMITTED>"
            n.exc_stack = "<OMITTED>"

        return traj_copy

    def filter_for_best_path(self, best_node: str):
        path_to_best = self.get_path_to_node(best_node)
        filtered_traj = self.filter_on_path(path_to_best)
        return filtered_traj

    def filter_for_longest_path(self):
        longest_path = self.get_longest_path()
        filtered_traj = self.filter_on_path(longest_path)
        return filtered_traj

    def filter_traj(self):
        best_node = self.get_best_node(only_good=True)

        if best_node is not None:
            filtered_traj = self.filter_for_best_path(best_node.id)
        else:
            filtered_traj = self.filter_for_longest_path()

        return filtered_traj

    def get_sub_best_node(self, node_id: str) -> Node | None:
        """
        Get the best node in the subtree of the node with the given ID.
        This is the node with the highest metric value.
        If there are no valid nodes in the subtree, return None.
        """
        node = self.get_node_by_id(node_id)
        if node:
            subtree_nodes = self._get_nodes_in_subtree(node)
            valid_nodes = [n for n in subtree_nodes if n in self.good_nodes]

            if not valid_nodes:
                return None
            max_metric_nodes = [n for n in valid_nodes if n.metric.maximize]
            min_metric_nodes = [n for n in valid_nodes if not n.metric.maximize]
            filtered_nodes = (
                max_metric_nodes
                if len(max_metric_nodes) > len(min_metric_nodes)
                else min_metric_nodes
            )

            try:
                return max(filtered_nodes, key=lambda n: n.metric)
            except Exception as e:
                logger.error(f"Metric comparison error: {e}")
                return valid_nodes[0]
        else:
            return None

    def get_sub_best_nodes_per_tree(self) -> list[Node]:
        """
        For each root node (draft node), find the best node in its subtree based on the metric.
        Return a list of these best nodes sorted by their metric values.
        """
        best_nodes = []

        # For each root node (draft node)
        for root in self.draft_nodes:
            sub_best_node = self.get_sub_best_node(root.id)
            if sub_best_node:
                best_nodes.append(sub_best_node)

        return best_nodes

    def get_better_node(self, node1: Node, node2: Node) -> Node:
        """Get the node with the better metric value."""
        logger.info(f"Comparing nodes {node1.id} and {node2.id}")
        if node1.is_buggy and node2.is_buggy:
            logger.info("Both nodes are not valid.")
            return None
        if node1.is_buggy:
            logger.info("Node1 is not valid, returning node2.")
            return node2
        if node2.is_buggy:
            logger.info("Node2 is not valid, returning node1.")
            return node1
        if node1.metric.maximize != node2.metric.maximize:
            logger.info("The nodes have different metrics.")
            return None

        try:
            logger.info(
                f"Comparing metrics: Node1 {node1.metric} and Node2{node2.metric}"
            )
            return max([node1, node2], key=lambda n: n.metric)
        except Exception as e:
            logger.error(f"Metric comparison error: {e}")
            return None

    def _get_nodes_in_subtree(self, root: Node) -> list[Node]:
        """Get all nodes in the subtree starting from the given root."""
        result = [root]
        queue = list(root.children)
        while queue:
            node = queue.pop(0)
            result.append(node)
            queue.extend(node.children)
        return result
