import flwr as fl
from typing import Dict, Optional, Tuple, List
import numpy as np
from utils import plot_metrics

rounds = []
accuracy = []
loss = []

def weighted_average(metrics: List[Tuple[int, Dict[str, float]]]) -> Dict[str, float]:
    """Aggregates metrics weighted by number of samples."""
    total_samples = sum([num_samples for num_samples, _ in metrics])
    weighted_metrics = {}
    
    for metric_name in metrics[0][1].keys():
        weighted_sum = sum([m[metric_name] * num_samples for num_samples, m in metrics])
        weighted_metrics[metric_name] = weighted_sum / total_samples
    
        if metric_name == "accuracy":
            accuracy.append(weighted_metrics["accuracy"])
        elif metric_name == "loss":
            loss.append(weighted_metrics["loss"])

    return weighted_metrics

def main():
    global rounds

    # Define strategy with custom aggregation
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=3,
        min_evaluate_clients=3,
        min_available_clients=3,
        evaluate_metrics_aggregation_fn=weighted_average,
        fit_metrics_aggregation_fn=weighted_average
    )

    # Start server with custom config
    fl.server.start_server(
        server_address="localhost:8080",
        config=fl.server.ServerConfig(num_rounds=10),
        strategy=strategy
    )

    if accuracy and loss:  # Ensure both lists have data
        rounds = list(range(1, len(accuracy) + 1))
        plot_metrics(rounds, accuracy, loss, save_path='plots/')
    else:
        print("No metrics were recorded. Skipping plot generation.")

if __name__ == "__main__":
    main()