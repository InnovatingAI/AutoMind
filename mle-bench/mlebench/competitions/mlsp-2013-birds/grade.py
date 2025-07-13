import pandas as pd
from sklearn.metrics import roc_auc_score

from mlebench.competitions.utils import prepare_for_auroc_metric


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    if (
        "Id" not in submission.columns
        and "rec_id" in submission.columns
        and "species" in submission.columns
    ):
        submission["Id"] = submission["rec_id"] * 100 + submission["species"]
    if "Probability" not in submission.columns and "probability" in submission.columns:
        submission = submission.rename(columns={"probability": "Probability"})
    roc_auc_inputs = prepare_for_auroc_metric(
        submission=submission, answers=answers, id_col="Id", target_col="Probability"
    )
    return roc_auc_score(
        y_true=roc_auc_inputs["y_true"], y_score=roc_auc_inputs["y_score"]
    )
