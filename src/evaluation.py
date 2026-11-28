"""Shared evaluation tools for the project.

Used by every model (classification and regression) so that all the
comparisons are fair:
    - walk_forward_splits : time-based train / validation splits
    - regression_metrics  : MSE and QLIKE
    - classification_metrics : PR-AUC, recall, precision, F1, cost, accuracy
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_squared_error,
    average_precision_score,
    recall_score,
    precision_score,
    f1_score,
    confusion_matrix,
    accuracy_score,
)


def walk_forward_splits(dates, first_val_year=2012, last_val_year=2019, gap=5):
    """Return a list of (train_index, val_index), one pair per validation year.

    dates : pd.Series of datetimes (one value per row of the dataset,
            already filtered on 2006-2019, with index = row numbers).
    gap   : number of trading days removed between train and validation,
            because the target of a row is the volatility of the next day.
    """

    # 1. Distinct trading days, sorted from oldest to newest
    trading_days = list(pd.Series(dates.unique()).sort_values())

    splits = []

    # 2. One fold per validation year
    for year in range(first_val_year, last_val_year + 1):

        # 3. Position of the first trading day of this year
        pos = 0
        for i in range(len(trading_days)):
            if trading_days[i].year == year:
                pos = i
                break

        # 4. Last training day: go back gap + 1 trading days
        last_train_day = trading_days[pos - (gap + 1)]

        # 5. Go through every row and decide: train, validation, or ignored
        train_index = []
        val_index = []
        for row_number, date in dates.items():
            if date <= last_train_day:
                train_index.append(row_number)
            elif date.year == year:
                val_index.append(row_number)

        splits.append((train_index, val_index))

    return splits


def regression_metrics(y_true, y_pred):
    """Compute MSE and QLIKE between true and predicted log-volatility.

    y_true and y_pred are log of volatility, so the variance is
    exp(2 * log_vol).
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # MSE on the log of the volatility
    mse = mean_squared_error(y_true, y_pred)

    # True variance (sigma2) and predicted variance (h)
    sigma2 = np.exp(2 * y_true)
    h = np.exp(2 * y_pred)

    # Avoid division by zero
    h = np.maximum(h, 1e-12)

    # QLIKE loss: 0 for a perfect forecast, penalizes underestimation more
    ratio = sigma2 / h
    qlike = np.mean(ratio - np.log(ratio) - 1)

    return {
        "MSE": mse,
        "QLIKE": qlike,
    }


def classification_metrics(y_true, proba, threshold=0.5, c_fn=5, c_fp=1):
    """Compute classification metrics for the volatility spike detection.

    proba     : predicted probabilities of a spike (class 1).
    threshold : probability above which we predict a spike.
    c_fn      : cost of a missed spike (false negative).
    c_fp      : cost of a false alarm (false positive).
    """

    y_true = np.asarray(y_true)
    proba = np.asarray(proba)

    # Convert probabilities to 0/1 predictions with the threshold
    y_pred = []
    for p in proba:
        if p >= threshold:
            y_pred.append(1)
        else:
            y_pred.append(0)

    # Confusion matrix (labels fixed so it always has 4 values)
    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    # PR-AUC uses the probabilities, so it does not depend on the threshold
    pr_auc = average_precision_score(y_true, proba)

    # Metrics based on the 0/1 predictions
    recall = recall_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # Cost: a missed spike costs more than a false alarm
    cost = c_fn * fn + c_fp * fp

    # Accuracy: only to show that it is misleading here
    accuracy = accuracy_score(y_true, y_pred)

    return {
        "PR_AUC": pr_auc,
        "Recall": recall,
        "Precision": precision,
        "F1": f1,
        "Cost": cost,
        "Accuracy": accuracy,
        "FN": fn,
        "FP": fp,
    }