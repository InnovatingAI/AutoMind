import atexit
import shutil

from .backend import compile_prompt_to_md
from .agent import Agent
from .interpreter import Interpreter
from .trajectory import Trajectory, Node
from .utils.config import (
    load_task_desc,
    prep_agent_workspace,
    save_run,
    config_manager,
)
from .utils.log import setup_logger

from omegaconf import OmegaConf
from rich.theme import Theme
from rich.tree import Tree


def print_execution_result(exec_result, logger):
    """
    Print execution results including plan, code and output
    """
    logger.info("\n[bold blue]Plan:[/]")
    if exec_result.plan:
        if isinstance(exec_result.plan, (list, tuple)):
            plan_str = "\n".join(exec_result.plan)
        else:
            plan_str = str(exec_result.plan)
        logger.info(plan_str)

    logger.info("\n[bold blue]Code:[/]")
    if exec_result.code:
        code_str = "".join(str(c) for c in exec_result.code if c)
        logger.info(code_str)

    logger.info("\n[bold blue]Execution output:[/]")
    if exec_result.exc_type:
        logger.error("[red]Execution raised an exception:", exec_result.exc_type)
        logger.error("[red]Exception details:", exec_result.exc_info)
        return False
    else:
        if exec_result._term_out is not None:
            if isinstance(exec_result._term_out, (list, tuple)):
                output_str = "".join(str(item) for item in exec_result._term_out)
                logger.info(output_str)
            else:
                logger.info(str(exec_result._term_out))
    logger.info(f"\n[italic]Execution time: {exec_result.exec_time:.2f} seconds[/]")
    return True


def run():
    cfg = config_manager.initialize_config()
    config_manager.cfg = cfg

    # Setup logging system
    logger = setup_logger(cfg)
    logger.info(f'Starting run "{cfg.exp_name}"')

    task_desc = load_task_desc(cfg)

    logger.info("Preparing agent workspace (copying and extracting files) ...")
    prep_agent_workspace(cfg)

    traj = Trajectory()
    agent = Agent(
        task_desc=task_desc,
        cfg=cfg,
        traj=traj,
    )

    def cleanup():
        if agent.current_step == 0:
            shutil.rmtree(cfg.workspace_dir)

    atexit.register(cleanup)

    file_paths = [
        f"Result visualization:\n[yellow]▶ {str((cfg.log_dir / 'tree_plot.html'))}",
        f"Agent workspace directory:\n[yellow]▶ {str(cfg.workspace_dir)}",
        f"Experiment log directory:\n[yellow]▶ {str(cfg.log_dir)}",
    ]
    for path in file_paths:
        logger.info(path)

    while agent.current_step < cfg.agent.steps:
        try:
            result = agent.step()
            # print_execution_result(result, logger)
            logger.info(traj.to_string_tree())
            save_run(cfg, traj)

        except RecursionError:
            logger.error(
                "maximum recursion depth exceeded, pleas check recursive reference."
            )
            break
        except Exception as e:
            logger.error(f"{str(e)}")

    agent.interpreter.cleanup_session()


if __name__ == "__main__":
    run()
