import pandas as pd
import os

log_path = "logs/training_metrics.csv"

def init_logger():

    if not os.path.exists("logs"):
        os.makedirs("logs")

    df = pd.DataFrame(columns=[
        "step",
        "avg_reward",
        "contradiction_rate",
        "consistency_rate",
        "loss"
    ])

    df.to_csv(log_path, index=False)


def log_metrics(step, reward, contradiction, consistency, loss):

    df = pd.read_csv(log_path)

    df.loc[len(df)] = [
        step,
        reward,
        contradiction,
        consistency,
        loss
    ]

    df.to_csv(log_path, index=False)