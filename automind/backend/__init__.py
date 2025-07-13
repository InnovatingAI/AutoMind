import logging
from . import backend_openai
from .utils import OutputType, PromptType, compile_prompt_to_md, FuncSpec

logger = logging.getLogger("automind")


def query(
    system_message: PromptType | None,
    user_message: PromptType | None,
    model: str,
    temperature: float | None = None,
    max_tokens: int | None = None,
    func_spec: FuncSpec | None = None,
    max_iterations: int = 3,
    valid_check=None,
    **model_kwargs,
) -> OutputType:
    """
    General LLM query for various backends with a single system and user message.
    Supports function calling for some backends.

    Args:
        system_message (PromptType | None): Uncompiled system message (will generate a message following the OpenAI/Anthropic format)
        user_message (PromptType | None): Uncompiled user message (will generate a message following the OpenAI/Anthropic format)
        model (str): string identifier for the model to use (e.g. "gpt-4-turbo")
        temperature (float | None, optional): Temperature to sample at. Defaults to the model-specific default.
        max_tokens (int | None, optional): Maximum number of tokens to generate. Defaults to the model-specific max tokens.
        func_spec (FuncSpec | None, optional): Optional FuncSpec object defining a function call. If given, the return value will be a dict.
        max_iterations (int, optional): Maximum number of attempts to get a valid response. Defaults to 1.
        valid_check (callable, optional): Function to validate the response. If None, all responses are considered valid.

    Returns:
        OutputType: A string completion if func_spec is None, otherwise a dict with the function call details.
    """

    model_kwargs = model_kwargs | {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    # Handle models with beta limitations
    # ref: https://platform.openai.com/docs/guides/reasoning/beta-limitations
    if model.startswith("o1"):
        if system_message:
            user_message = system_message
        system_message = None
        model_kwargs["temperature"] = 1

    # query_func = backend_anthropic.query if "claude-" in model else backend_openai.query
    query_func = backend_openai.query
    system_message = compile_prompt_to_md(system_message) if system_message else None
    user_message = compile_prompt_to_md(user_message) if user_message else None

    logger.info(
        f"Querying {model_kwargs} with message: \n\n{system_message}\n{user_message}\n\n"
    )

    for iteration in range(max_iterations):
        output, req_time, in_tok_count, out_tok_count, info = query_func(
            system_message=system_message,
            user_message=user_message,
            func_spec=func_spec,
            **model_kwargs,
        )

        logger.info(
            f"Query {model_kwargs} completed in {req_time:.2f}s. Input tokens: {in_tok_count}, output tokens: {out_tok_count}. Output: \n\n{output}\n\n"
        )

        # Check if the response is valid
        if valid_check is None:
            return output
        elif (
            valid_check(output)
            and valid_check(output) is not None
            and bool(valid_check(output))
        ):
            return output

        if iteration < max_iterations - 1:
            logger.info(
                f"Invalid response (attempt {iteration+1}/{max_iterations}). Retrying..."
            )

    logger.error(f"Failed to generate valid response after {max_iterations} attempts.")
    raise RuntimeError(
        f"Cannot generate valid response after {max_iterations} attempts."
    )
