import numpy as np

def main() -> int:
    y_true = np.array([3, 5, 7, 9])
    y_pred = np.array([2, 6, 8, 7])

    errors = y_pred - y_true
    absolute_errors = np.abs(errors)
    squared_errors = errors**2

    mean_error = np.sum(errors) / len(errors)
    mae = np.sum(absolute_errors) / len(absolute_errors)
    mse = np.sum(errors**2) / len(errors)
    rmse = np.sqrt(mse)

    print(f"y_true: {y_true}")
    print(f"y_pred: {y_pred}")
    print(f"errors: {errors}")
    print(f"absolute errors: {absolute_errors}")
    print(f"squared errors: {squared_errors}")
    print()
    print(f"ME: {mean_error}")
    print(f"MAE: {mae}")
    print(f"MSE: {mse}")
    print(f"RMSE: {rmse}")

    return 0