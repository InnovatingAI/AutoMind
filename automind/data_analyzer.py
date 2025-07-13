import json
import humanize
import logging
import pandas as pd
from abc import ABC, abstractmethod
from typing import Optional
from pathlib import Path
from genson import SchemaBuilder
from pandas.api.types import is_numeric_dtype

from .utils.response import trim_long_string

logger = logging.getLogger("automind")


class DataAnalyzer(ABC):
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
    def step(self):
        pass


class DataPreviewer(DataAnalyzer):
    def __init__(
        self,
        model: str,
        task_desc: str,
        data_dir: str,
        human_guideline: Optional[str] = None,
    ):
        super().__init__(model, task_desc, human_guideline)
        self.data_dir = data_dir

    @staticmethod
    def preview_csv(p: Path, file_name: str, simple=False) -> str:
        """Generate a textual preview of a csv file

        Args:
            p (Path): the path to the csv file
            file_name (str): the file name to use in the preview
            simple (bool, optional): whether to use a simplified version of the preview. Defaults to False.

        Returns:
            str: the textual preview
        """
        df = pd.read_csv(p)
        out = []
        out.append(f"-> {file_name} has {df.shape[0]} rows and {df.shape[1]} columns.")

        if simple:
            cols = df.columns.tolist()
            sel_cols = 15
            cols_str = ", ".join(cols[:sel_cols])
            res = f"The columns are: {cols_str}"
            if len(cols) > sel_cols:
                res += f"... and {len(cols)-sel_cols} more columns"
            out.append(res)
        else:
            out.append("Here is some information about the columns:")
            for col in sorted(df.columns):
                dtype = df[col].dtype
                name = f"{col} ({dtype})"

                nan_count = df[col].isnull().sum()

                if dtype == "bool":
                    v = df[col][df[col].notnull()].mean()
                    out.append(f"{name} is {v*100:.2f}% True, {100-v*100:.2f}% False")
                elif df[col].nunique() < 10:
                    out.append(
                        f"{name} has {df[col].nunique()} unique values: {df[col].unique().tolist()}"
                    )
                elif is_numeric_dtype(df[col]):
                    out.append(
                        f"{name} has range: {df[col].min():.2f} - {df[col].max():.2f}, {nan_count} nan values"
                    )
                elif dtype == "object":
                    out.append(
                        f"{name} has {df[col].nunique()} unique values. Some example values: {df[col].value_counts().head(4).index.tolist()}"
                    )

        return "\n".join(out)

    @staticmethod
    def preview_txt(p: Path, file_name: str, header: int = 3) -> str:
        """Generate a textual preview of a text file

        Args:
            p (Path): the path to the text file
            file_name (str): the file name to use in the preview

        Returns:
            str: the textual preview
        """
        with open(p) as f:
            content = f.readlines()

        out = [f"-> {file_name} has {len(content)} lines."]
        if len(content) > header:
            out.append(f"Here are the header {header} lines:\n")
            out.extend(content[:header])
        else:
            out.append("Here is the file content:\n")
            out.extend(content)

        result = "".join(out)
        result = trim_long_string(result, threshold=1100, k=500)

        return result

    @staticmethod
    def preview_json(p: Path, file_name: str) -> str:
        """Generate a textual preview of a json file using a generated json schema"""
        builder = SchemaBuilder()
        with open(p) as f:
            first_line = f.readline().strip()

            try:
                first_object = json.loads(first_line)

                if not isinstance(first_object, dict):
                    raise json.JSONDecodeError(
                        "The first line isn't JSON", first_line, 0
                    )

                # if the the next line exists and is not empty, then it is a JSONL file
                second_line = f.readline().strip()
                if second_line:
                    f.seek(0)  # so reset and read line by line
                    for line in f:
                        builder.add_object(json.loads(line.strip()))
                # if it is empty, then it's a single JSON object file
                else:
                    builder.add_object(first_object)

            except json.JSONDecodeError:
                # if first line isn't JSON, then it's prettified and we can read whole file
                f.seek(0)
                builder.add_object(json.load(f))

        return f"-> {file_name} has auto-generated json schema:\n" + builder.to_json(
            indent=2
        )

    def step(self):
        """
        Generate a textual preview of a directory, including an overview of the directory
        structure and previews of individual files
        """
        # these files are treated as code (e.g. markdown wrapped)
        code_files = {
            ".py",
            ".sh",
            ".yaml",
            ".yml",
            ".md",
            ".html",
            ".xml",
            ".log",
            ".rst",
        }
        # we treat these files as text (rather than binary) files
        plaintext_files = {".txt", ".csv", ".json", ".jsonl", ".tsv"} | code_files

        def get_file_len_size(f: Path) -> tuple[int, str]:
            """
            Calculate the size of a file (#lines for plaintext files, otherwise #bytes)
            Also returns a human-readable string representation of the size.
            """
            if f.suffix in plaintext_files:
                num_lines = sum(1 for _ in open(f))
                return num_lines, f"{num_lines} lines"
            else:
                s = f.stat().st_size
                return s, humanize.naturalsize(s)

        def file_tree(path: Path, depth: int = 0, max_n: int = 8) -> str:
            """Generate a tree structure of files in a directory"""
            result = []
            files = [p for p in Path(path).iterdir() if not p.is_dir()]
            dirs = [p for p in Path(path).iterdir() if p.is_dir()]
            for p in sorted(files)[:max_n]:
                result.append(f"{' '*depth*4}{p.name} ({get_file_len_size(p)[1]})")
            if len(files) > max_n:
                result.append(f"{' '*depth*4}... and {len(files)-max_n} other files")

            for p in sorted(dirs)[:max_n]:
                result.append(f"{' '*depth*4}{p.name}/")
                result.append(file_tree(p, depth + 1))
            if len(dirs) > max_n:
                result.append(
                    f"{' '*depth*4}... and {len(dirs)-max_n} other directories"
                )

            return "\n".join(result)

        def _walk(path: Path):
            """Recursively walk a directory (analogous to os.walk but for pathlib.Path)"""
            for p in sorted(Path(path).iterdir()):
                if p.is_dir():
                    yield from _walk(p)
                    continue
                yield p

        tree = f"```\n{file_tree(self.data_dir)}```"

        out = []
        for fn in _walk(self.data_dir):
            file_name = str(fn.relative_to(self.data_dir))

            if fn.suffix == ".csv":
                out.append(self.preview_csv(fn, file_name))
            if fn.suffix == ".txt":
                out.append(self.preview_txt(fn, file_name))
            elif fn.suffix == ".json":
                out.append(self.preview_json(fn, file_name))
            elif fn.suffix in plaintext_files:
                if get_file_len_size(fn)[0] < 30:
                    with open(fn) as f:
                        content = f.read()
                        if fn.suffix in code_files:
                            content = f"```\n{content}\n```"
                        out.append(f"-> {file_name} has content:\n\n{content}")

        analysis = "\n\n".join(out)
        analysis = trim_long_string(analysis, threshold=5100, k=2500)
        result = f"Here is the directory structure:\n\n{tree}\n\nHere is the data analysis:\n\n{analysis}\n"
        return result
