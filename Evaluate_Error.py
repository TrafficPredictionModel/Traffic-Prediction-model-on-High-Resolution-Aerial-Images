import numpy as np
import math

def evaluat_error_(sp, act, threshold=0.1, profit_limits=(200000, 800000)) -> object:
    r = np.squeeze(act)
    x = np.squeeze(sp)
    points = np.zeros(len(x))
    abs_r = np.abs(r)
    abs_x = np.abs(x)
    abs_r_x = np.abs(r - x)

    # Calculate differences between successive predictions
    for j in range(1, len(x)):
        points[j] = abs(x[j] - x[j - 1])

    # Calculate error metrics
    md = (100 / len(x)) * np.sum(abs_r_x / abs_r)
    smape = (1 / len(x)) * np.sum(abs_r_x / ((abs_r + abs_x) / 2))
    mase = np.sum(abs_r_x) / ((1 / (len(x) - 1)) * np.sum(points))
    mae = np.sum(abs_r_x) / len(r)
    rmse = np.sqrt(np.sum(abs_r_x ** 2) / len(r))
    mse = np.sum(abs_r_x ** 2) / len(r)  # Mean Squared Error
    onenorm = np.sum(abs_r_x)
    twonorm = np.sqrt(np.sum(abs_r_x ** 2))
    mape = np.mean(abs_r_x / abs_r) * 100  # Mean Absolute Percentage Error
    accurate_predictions = np.sum(abs_r_x <= threshold) / len(r)  # Proportion of predictions within the threshold

    # Calculate profit and loss metrics
    total_profit = np.sum((r - x)[(r - x) > 0])
    average_profit_per_month = total_profit / 12 if total_profit > 0 else 0

    # Combine metrics into a single evaluation list
    EVAL_ERR = [
        accurate_predictions,
        md,
        smape,
        mase,
        mae,
        rmse,
        mse,
        onenorm,
        twonorm,
        mape
    ]

    return EVAL_ERR


def evaluat_error_2(sp, act, threshold=0.1, profit_limits=(200000, 800000)):
    sp = np.array(sp)
    act = np.array(act)

    if sp.shape != act.shape:
        raise ValueError("Shape mismatch: predicted and actual arrays must be the same shape.")

    n_outputs = sp.shape[1]
    metrics_per_output = []

    for i in range(n_outputs):
        x = np.squeeze(sp[:, i])
        r = np.squeeze(act[:, i])

        abs_r = np.abs(r)
        abs_x = np.abs(x)
        abs_r_x = np.abs(r - x)

        # Calculate differences between successive predictions
        points = np.zeros(len(x))
        for j in range(1, len(x)):
            points[j] = abs(x[j] - x[j - 1])

        # Calculate error metrics
        md = (100 / len(x)) * np.sum(abs_r_x / abs_r)
        smape = (1 / len(x)) * np.sum(abs_r_x / ((abs_r + abs_x) / 2))
        mase = np.sum(abs_r_x) / ((1 / (len(x) - 1)) * np.sum(points))
        mae = np.sum(abs_r_x) / len(r)
        rmse = np.sqrt(np.sum(abs_r_x ** 2) / len(r))
        mse = np.sum(abs_r_x ** 2) / len(r)
        onenorm = np.sum(abs_r_x)
        twonorm = np.sqrt(np.sum(abs_r_x ** 2))
        mape = np.mean(abs_r_x / abs_r) * 100
        accurate_predictions = np.sum(abs_r_x <= threshold) / len(r)

        # Profit metrics (if applicable for this output)
        total_profit = np.sum((r - x)[(r - x) > 0])
        average_profit_per_month = total_profit / 12 if total_profit > 0 else 0

        metrics = [
            accurate_predictions,
            md,
            smape,
            mase,
            mae,
            rmse,
            mse,
            onenorm,
            twonorm,
            mape,
            total_profit,
            average_profit_per_month
        ]

        metrics_per_output.append(metrics)

    return np.array(metrics_per_output)
