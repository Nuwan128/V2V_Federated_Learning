import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os

def load_client_data(client_id):
    """
    Loads training and testing data for a specific client, with error handling.

    Args:
        client_id: The ID of the client.

    Returns:
        A tuple containing the training features (X_train), training labels (y_train),
        testing features (X_test), and testing labels (y_test), or None if data is missing.
    """

    # Adjust data directory path as needed
    data_dir = "data"
    train_path = f"{data_dir}/client_{client_id}/train.csv"
    test_path = f"{data_dir}/client_{client_id}/test.csv"

    try:
        # Read training data
        train_data = pd.read_csv(train_path)
        X_train = train_data[['speed', 'pos_x', 'pos_y', 'heading', 'acceleration']].values
        y_train = train_data['target'].values

        # Read testing data
        test_data = pd.read_csv(test_path)
        X_test = test_data[['speed', 'pos_x', 'pos_y', 'heading', 'acceleration']].values
        y_test = test_data['target'].values

        # Scale features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        return X_train, y_train, X_test, y_test

    except FileNotFoundError:
        # Handle missing data files
        print(f"Error: Data files not found for client {client_id}")
        return None, None, None, None


def plot_metrics(rounds, accuracy, loss, save_path='plots/'):
    if not rounds or not accuracy or not loss:
        print("Error: Insufficient data to generate plots.")
        return

    os.makedirs(save_path, exist_ok=True)

    # Plot accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(rounds, accuracy, marker='o', label='Accuracy', color='b')
    plt.title('Federated Learning Accuracy Over Rounds')
    plt.xlabel('Round')
    plt.ylabel('Accuracy')
    plt.grid(alpha=0.5)
    plt.legend()
    plt.savefig(os.path.join(save_path, 'accuracy_plot.png'))
    plt.close()

    # Plot loss
    plt.figure(figsize=(10, 5))
    plt.plot(rounds, loss, marker='o', label='Loss', color='r')
    plt.title('Federated Learning Loss Over Rounds')
    plt.xlabel('Round')
    plt.ylabel('Loss')
    plt.grid(alpha=0.5)
    plt.legend()
    plt.savefig(os.path.join(save_path, 'loss_plot.png'))
    plt.close()

    print(f"Graphs saved in {save_path}")
