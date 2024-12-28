import flwr as fl
import tensorflow as tf
from utils import load_client_data

def create_v2v_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(5,)),  # Fix the input shape warning
        tf.keras.layers.Dense(64, activation='relu'),  # Increased network capacity
        tf.keras.layers.BatchNormalization(),  # Add normalization
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    return model

class V2VClient(fl.client.NumPyClient):
    def __init__(self, client_id):
        self.client_id = client_id
        self.X_train, self.y_train, self.X_test, self.y_test = load_client_data(client_id)
        if self.X_train is None:
            raise ValueError(f"Client {client_id} has no data.")
        self.model = create_v2v_model()
        
    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        
        # Add early stopping and learning rate scheduling
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='loss',
                patience=3,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='loss',
                factor=0.5,
                patience=2
            )
        ]
        
        # Increase epochs and add validation split
        history = self.model.fit(
            self.X_train, 
            self.y_train,
            epochs=5,  # Increased from 1
            batch_size=32,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=0
        )
        
        return self.model.get_weights(), len(self.X_train), {
            "loss": history.history["loss"][-1],
            "accuracy": history.history["accuracy"][-1]
        }

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy, precision, recall = self.model.evaluate(
            self.X_test, 
            self.y_test, 
            verbose=0
        )
        print(f"Client {self.client_id} Evaluation - "
              f"Loss: {loss:.4f}, Accuracy: {accuracy:.4f}, "
              f"Precision: {precision:.4f}, Recall: {recall:.4f}")
        return loss, len(self.X_test), {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall)
        }

def main():
    import sys
    if len(sys.argv) != 2:
        print("Usage: python client.py <client_id>")
        sys.exit(1)
    
    client_id = int(sys.argv[1])
    client = V2VClient(client_id)
    
    fl.client.start_numpy_client(
        server_address="localhost:8080",
        client=client
    )

if __name__ == "__main__":
    main()