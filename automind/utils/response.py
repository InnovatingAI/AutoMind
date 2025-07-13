import json
import re
import ast
import logging
import black
from bs4 import BeautifulSoup
from json import JSONDecodeError
import markdown
import numpy as np
import regex

logger = logging.getLogger("automind")


def wrap_code(code: str, lang="python") -> str:
    """Wraps code with three backticks."""
    return f"```{lang}\n{code}\n```"


def is_valid_python_script(script):
    """Check if a script is a valid Python script."""
    try:
        compile(script, "<string>", "exec")
        return True
    except SyntaxError as e:
        logger.error(f"Invalid Python script: {e.msg}")
        return False


def extract_python_code_from_markdown(text):
    html = markdown.markdown(text, extensions=["fenced_code"])
    soup = BeautifulSoup(html, "html.parser")
    python_code_blocks = []
    for code_tag in soup.find_all("code"):
        parent = code_tag.parent
        if parent.name == "pre":
            classes = code_tag.get("class", [])
            language = None
            for cls in classes:
                if cls.startswith("language-"):
                    language = cls.split("-", 1)[1]
                    break
            assert language is not None, "Cannot detect the language. text: {}".format(
                text
            )
            if language.lower() == "python":
                python_code_blocks.append(code_tag.get_text())

    assert len(python_code_blocks) > 0, "Cannot extract python code."
    all_length = []
    for code_id in range(len(python_code_blocks)):
        code_patch = python_code_blocks[code_id]
        code_patch = re.split(r"\n+", code_patch)
        all_length.append(len(code_patch))
    target_id = np.argmax(all_length)
    target_code = python_code_blocks[target_id]
    return target_code


def extract_json_dict(text):
    pattern = regex.compile(
        r"""
        \{                    # Match the opening brace
            (?:
                [^{}"']+       # Match any characters except braces and quotes
                |              # OR
                "(?:\\.|[^"\\])*"   # Match double-quoted strings, handling escaped quotes
                |              # OR
                '(?:\\.|[^'\\])*'   # Match single-quoted strings, handling escaped quotes
                |              # OR
                (?0)           # Recursively match nested braces
            )*
        \}                    # Match the closing brace
    """,
        regex.VERBOSE,
    )

    matches = list(pattern.finditer(text))
    if not matches:
        return text

    last_match = matches[-1]
    json_string = last_match.group()
    json_string = json_string.replace("\n", " ").replace("\r", " ")
    json_string = re.sub(r"\s+", " ", json_string)
    try:
        json_dict = json.loads(json_string)
        return json_dict
    except json.JSONDecodeError:
        return json_string


def extract_jsons(text):
    """Extract all JSON objects from the text. Caveat: This function cannot handle nested JSON objects.

    Parameters
    ----------
    text : str
        The string that contains json code.

    Returns
    -------
    list : the extracted json objects' list.

    """
    json_objects = []
    matches = re.findall(r"\{.*?\}", text, re.DOTALL)
    for match in matches:
        try:
            json_obj = json.loads(match)
            json_objects.append(json_obj)
        except JSONDecodeError as e:
            error_pos = e.pos
            if error_pos < len(match):
                # Mark the error position with bold red
                error_context = (
                    match[:error_pos]
                    + "\033[1;31m"
                    + match[error_pos : error_pos + 1]
                    + "\033[0m"
                    + match[error_pos + 1 :]
                )
                logger.error(f"Failed to load JSON. Error: {e}")
                logger.error(
                    f"Failed JSON content with error highlighted: {error_context}"
                )
                logger.error(f"Full text: {text}")
            else:
                logger.error(f"Failed to load JSON. Error: {e}")
                logger.error(f"Failed JSON content: {match}")
                logger.error(f"Full text: {text}")

    # Sometimes chatgpt-turbo forget the last curly bracket, so we try to add it back when no json is found
    if len(json_objects) == 0 and not text.endswith("}"):
        json_objects = extract_jsons(text + "}")
        if len(json_objects) > 0:
            return json_objects

    return json_objects  # if there is no json object, it will return an empty list


def extract_json_block(json_str):
    """Extract json code from the input string.
    This function is capable of extracting JSON code from a code block:
    ```json
    {
        "key": "value"
    }
    ```

    Parameters
    ----------
    json_str : str
        The string that contains json code.

    Returns
    -------
    str : the extracted json code.
    """
    assert isinstance(json_str, str)
    json_str = json_str.strip()
    lines = json_str.split("\n")
    json_lines = []
    start_collect = False
    for line in lines:
        if line == "```json":
            start_collect = True
            continue
        elif line == "```":
            start_collect = False
            continue

        if start_collect:
            json_lines.append(line)
    try:
        json_obj = json.loads("\n".join(json_lines))
    except JSONDecodeError as e:
        print("fail to load JSON. Error:", e)
        print("\n".join(lines))
        json_obj = None
    return json_obj


def extract_json(json_str):
    assert isinstance(json_str, str)
    json_str = json_str.strip()
    lines = json_str.split("\n")
    json_lines = []
    start_collect = False
    for line in lines:
        if line == "```json":
            start_collect = True
            continue
        elif line == "```":
            start_collect = False
            continue

        if start_collect:
            json_lines.append(line)
    try:
        json_obj = json.loads("\n".join(json_lines))
    except json.JSONDecodeError as e:
        json_obj = None
    return json_obj


def trim_long_string(string, threshold=5100, k=2500):
    # Check if the length of the string is longer than the threshold
    if len(string) > threshold:
        # Output the first k and last k characters
        first_k_chars = string[:k]
        last_k_chars = string[-k:]

        truncated_len = len(string) - 2 * k

        return f"{first_k_chars}\n ... [{truncated_len} characters truncated] ... \n{last_k_chars}"
    else:
        return string


def extract_tuple(text):
    pattern = r"```(\(.*?\))```"
    match = re.search(pattern, text, flags=re.DOTALL)
    if match:
        tuple_str = match.group(1)
        result = ast.literal_eval(tuple_str)
        return result
    else:
        return None


def extract_xml(text, tag):
    pattern = f"<{tag}>\n*(.*?)\n*</{tag}>"
    match = re.search(pattern, text, flags=re.DOTALL)
    if match:
        result = match.group(1)
        return result
    else:
        return None


def extract_code(text):
    """Extract python code blocks from the text."""
    parsed_codes = []

    # When code is in a text or python block
    matches = re.findall(r"```(python)?\n*(.*?)\n*```", text, re.DOTALL)
    for match in matches:
        code_block = match[1]
        parsed_codes.append(code_block)

    # When the entire text is code or backticks of the code block is missing
    if len(parsed_codes) == 0:
        matches = re.findall(r"^(```(python)?)?\n?(.*?)\n?(```)?$", text, re.DOTALL)
        if matches:
            code_block = matches[0][2]
            parsed_codes.append(code_block)

    # validate the parsed codes
    valid_code_blocks = [
        format_code(c) for c in parsed_codes if is_valid_python_script(c)
    ]
    return format_code("\n\n".join(valid_code_blocks))


def extract_text_up_to_code(s):
    """Extract (presumed) natural language text up to the start of the first code block."""
    if "```" not in s:
        return ""
    return s[: s.find("```")].strip()


def format_code(code) -> str:
    """Format Python code using Black."""
    try:
        return black.format_str(code, mode=black.FileMode())
    except black.parsing.InvalidInput:  # type: ignore
        return code


def save_code_to_file(code, file_path):
    code = re.split(r"\n+", code)
    with open(file_path, "w") as f:
        for line in code:
            f.write(line + "\n")


def delete_debug_inform(code):
    """Delete debug information in the code."""
    original_code = code

    try:
        # Match print statements with [debug] pattern
        # Handle both single-line and multi-line print statements

        # First, handle standard single line debug print statements
        pattern = r"^\s*print\(\s*\f*\"\[debug\].*?\"\s*\).*$"
        code = re.sub(pattern, "", code, flags=re.MULTILINE)

        # Handle multi-line debug print statements with triple quotes
        pattern_multi = r"^\s*print\(\s*\f*\"\"\"\[debug\].*?\"\"\"\s*\).*$"
        code = re.sub(pattern_multi, "", code, flags=re.MULTILINE)

        # Handle print statements with [debug] in formatted strings
        pattern_f = r"^\s*print\(\s*f\"\[debug\].*?\"\s*\).*$"
        code = re.sub(pattern_f, "", code, flags=re.MULTILINE)

        # Handle debug print statements with variables
        pattern_vars = r"^\s*print\(\s*\"\[debug\].*?\"\s*,.*\).*$"
        code = re.sub(pattern_vars, "", code, flags=re.MULTILINE)

        # Clean up potential multiple consecutive empty lines caused by removal
        code = re.sub(r"\n\s*\n\s*\n+", "\n\n", code)

        # Verify the modified code is valid Python syntax
        compile(code, "<string>", "exec")

        return code
    except Exception as e:
        # Log the error and return the original code
        logger.error(f"Failed to remove debug information: {str(e)}")
        return original_code


def clean_string(str):
    return re.sub(r"[^\w\s]", "", str).strip().lower()
