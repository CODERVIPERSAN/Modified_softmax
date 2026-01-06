"""
MNIST Dataset Comparison for Different Softmax Variants
Compares standard softmax vs modified softmax vs weighted modified softmax
"""

import numpy as np
import gzip
import os
import urllib.request
import matplotlib.pyplot as plt
from tqdm import tqdm

# =============================================================================
# SOFTMAX FUNCTIONS
# =============================================================================

def sigmoid(x):
    """Sigmoid function with numerical stability"""
    return np.where(x >= 0, 
                    1 / (1 + np.exp(-x)), 
                    np.exp(x) / (1 + np.exp(x)))


def standard_softmax(logits):
    """Standard softmax: exp(x) / sum(exp(x))"""
    # Numerical stability: subtract max
    logits = logits - np.max(logits, axis=-1, keepdims=True)
    exp_logits = np.exp(logits)
    return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)


def modified_softmax(logits):
    """
    Modified Softmax: sigmoid(x) / sum(sigmoid(x))
    This normalizes sigmoid outputs to form probabilities
    """
    sig_logits = sigmoid(logits)
    return sig_logits / np.sum(sig_logits, axis=-1, keepdims=True)


def weighted_modified_softmax(logits, alpha=0.1):
    """
    Weighted Modified Softmax: sigmoid(αx) / sum(sigmoid(αx))
    Alpha controls sensitivity - lower values retain more distinction
    """
    scaled_logits = alpha * logits
    sig_logits = sigmoid(scaled_logits)
    return sig_logits / np.sum(sig_logits, axis=-1, keepdims=True)


# =============================================================================
# MNIST DATA LOADING
# =============================================================================

def download_mnist():
    """Download MNIST dataset if not already present"""
    base_url = 'http://yann.lecun.com/exdb/mnist/'
    files = {
        'train_images': 'train-images-idx3-ubyte.gz',
        'train_labels': 'train-labels-idx1-ubyte.gz',
        'test_images': 't10k-images-idx3-ubyte.gz',
        'test_labels': 't10k-labels-idx1-ubyte.gz'
    }
    
    os.makedirs('data', exist_ok=True)
    
    for name, file in files.items():
        if not os.path.exists(f'data/{file}'):
            print(f"Downloading {file}...")
            urllib.request.urlretrieve(base_url + file, f'data/{file}')


def load_mnist():
    """Load MNIST dataset"""
    download_mnist()
    
    with gzip.open('data/train-images-idx3-ubyte.gz', 'rb') as f:
        # Skip header
        f.read(16)
        train_images = np.frombuffer(f.read(), dtype=np.uint8).reshape(-1, 28*28)
    
    with gzip.open('data/train-labels-idx1-ubyte.gz', 'rb') as f:
        # Skip header
        f.read(8)
        train_labels = np.frombuffer(f.read(), dtype=np.uint8)
    
    with gzip.open('data/t10k-images-idx3-ubyte.gz', 'rb') as f:
        # Skip header
        f.read(16)
        test_images = np.frombuffer(f.read(), dtype=np.uint8).reshape(-1, 28*28)
    
    with gzip.open('data/t10k-labels-idx1-ubyte.gz', 'rb') as f:
        # Skip header
        f.read(8)
        test_labels = np.frombuffer(f.read(), dtype=np.uint8)
    
    # Normalize images to [0, 1]
    train_images = train_images.astype(np.float32) / 255.0
    test_images = test_images.astype(np.float32) / 255.0
    
    return train_images, train_labels, test_images, test_labels


# =============================================================================
# SIMPLE NEURAL NETWORK FOR MNIST
# =============================================================================

class SimpleNN:
    """Simple 2-layer neural network for MNIST"""
    
    def __init__(self, softmax_fn, alpha=None, hidden_size=128):
        self.softmax_fn = softmax_fn
        self.alpha = alpha
        self.hidden_size = hidden_size
        self.W1 = None
        self.b1 = None
        self.W2 = None
        self.b2 = None
    
    def _compute_probs(self, logits):
        if self.alpha is not None:
            return self.softmax_fn(logits, self.alpha)
        return self.softmax_fn(logits)
    
    def _relu(self, x):
        return np.maximum(0, x)
    
    def _init_params(self, input_size, output_size):
        # Xavier initialization
        np.random.seed(42)
        self.W1 = np.random.randn(input_size, self.hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros(self.hidden_size)
        self.W2 = np.random.randn(self.hidden_size, output_size) * np.sqrt(2.0 / self.hidden_size)
        self.b2 = np.zeros(output_size)
    
    def fit(self, X, y, epochs=10, batch_size=128, lr=0.01):
        n_samples, input_size = X.shape
        output_size = 10  # 10 digits for MNIST
        
        # Initialize parameters
        self._init_params(input_size, output_size)
        
        # One-hot encode labels
        y_onehot = np.zeros((n_samples, output_size))
        y_onehot[np.arange(n_samples), y] = 1
        
        # Training history
        history = {'loss': [], 'accuracy': []}
        
        # Training loop
        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_onehot_shuffled = y_onehot[indices]
            
            # Mini-batch training
            loss_epoch = 0
            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_onehot_shuffled[i:i+batch_size]
                batch_size_actual = X_batch.shape[0]
                
                # Forward pass
                h1 = X_batch @ self.W1 + self.b1
                h1_relu = self._relu(h1)
                logits = h1_relu @ self.W2 + self.b2
                probs = self._compute_probs(logits)
                
                # Compute loss
                loss = -np.sum(y_batch * np.log(probs + 1e-10)) / batch_size_actual
                loss_epoch += loss * batch_size_actual
                
                # Backward pass
                dprobs = probs - y_batch
                dW2 = h1_relu.T @ dprobs / batch_size_actual
                db2 = np.sum(dprobs, axis=0) / batch_size_actual
                
                dh1_relu = dprobs @ self.W2.T
                dh1 = dh1_relu * (h1 > 0)
                dW1 = X_batch.T @ dh1 / batch_size_actual
                db1 = np.sum(dh1, axis=0) / batch_size_actual
                
                # Update parameters
                self.W2 -= lr * dW2
                self.b2 -= lr * db2
                self.W1 -= lr * dW1
                self.b1 -= lr * db1
            
            # Compute epoch metrics
            loss_epoch /= n_samples
            train_preds = self.predict(X)
            accuracy = np.mean(train_preds == y)
            
            history['loss'].append(loss_epoch)
            history['accuracy'].append(accuracy)
            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss_epoch:.4f}, Accuracy: {accuracy:.4f}")
        
        return history
    
    def predict_proba(self, X):
        h1 = X @ self.W1 + self.b1
        h1_relu = self._relu(h1)
        logits = h1_relu @ self.W2 + self.b2
        return self._compute_probs(logits)
    
    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)


# =============================================================================
# EVALUATION METRICS
# =============================================================================

def evaluate_model(model, X_test, y_test):
    """Evaluate model on test data"""
    y_pred = model.predict(X_test)
    accuracy = np.mean(y_pred == y_test)
    
    # Compute probabilities
    y_proba = model.predict_proba(X_test)
    
    # Compute confidence (max probability)
    confidence = np.max(y_proba, axis=1)
    avg_confidence = np.mean(confidence)
    
    # Compute confidence for correct and incorrect predictions
    correct_mask = (y_pred == y_test)
    confidence_correct = np.mean(confidence[correct_mask])
    confidence_incorrect = np.mean(confidence[~correct_mask]) if np.any(~correct_mask) else 0
    
    # Compute cross-entropy loss
    y_onehot = np.zeros((len(y_test), 10))
    y_onehot[np.arange(len(y_test)), y_test] = 1
    loss = -np.mean(np.sum(y_onehot * np.log(y_proba + 1e-10), axis=1))
    
    return {
        'accuracy': accuracy,
        'loss': loss,
        'avg_confidence': avg_confidence,
        'confidence_correct': confidence_correct,
        'confidence_incorrect': confidence_incorrect
    }


def plot_misclassified(model, X_test, y_test, n_samples=5):
    """Plot some misclassified examples"""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    # Find misclassified examples
    misclassified = np.where(y_pred != y_test)[0]
    
    if len(misclassified) == 0:
        print("No misclassified examples found!")
        return
    
    # Select random misclassified examples
    indices = np.random.choice(misclassified, min(n_samples, len(misclassified)), replace=False)
    
    # Plot
    fig, axes = plt.subplots(1, len(indices), figsize=(15, 3))
    if len(indices) == 1:
        axes = [axes]
    
    for i, idx in enumerate(indices):
        img = X_test[idx].reshape(28, 28)
        true_label = y_test[idx]
        pred_label = y_pred[idx]
        confidence = y_proba[idx, pred_label]
        
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f"True: {true_label}, Pred: {pred_label}\nConf: {confidence:.2f}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('misclassified.png')
    plt.close()


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def run_mnist_experiment():
    """Run experiment comparing softmax variants on MNIST"""
    print("\n" + "=" * 70)
    print("MNIST EXPERIMENT: Comparing Softmax Variants")
    print("=" * 70)
    
    # Load data
    print("\nLoading MNIST dataset...")
    train_images, train_labels, test_images, test_labels = load_mnist()
    
    print(f"Train samples: {len(train_images)}, Test samples: {len(test_images)}")
    
    # Define models to compare
    models = {
        "Standard Softmax": SimpleNN(standard_softmax),
        "Modified Softmax": SimpleNN(modified_softmax),
        "Weighted (α=0.05)": SimpleNN(weighted_modified_softmax, alpha=0.05),
        "Weighted (α=0.1)": SimpleNN(weighted_modified_softmax, alpha=0.1),
        "Weighted (α=0.5)": SimpleNN(weighted_modified_softmax, alpha=0.5)
    }
    
    # Train and evaluate each model
    results = {}
    
    for name, model in models.items():
        print(f"\n{'-' * 50}")
        print(f"Training {name}...")
        history = model.fit(train_images, train_labels, epochs=5)
        
        print(f"\nEvaluating {name}...")
        metrics = evaluate_model(model, test_images, test_labels)
        results[name] = metrics
        
        print(f"Test Accuracy: {metrics['accuracy']:.4f}")
        print(f"Test Loss: {metrics['loss']:.4f}")
        print(f"Average Confidence: {metrics['avg_confidence']:.4f}")
        print(f"Confidence (Correct): {metrics['confidence_correct']:.4f}")
        print(f"Confidence (Incorrect): {metrics['confidence_incorrect']:.4f}")
        
        # Plot some misclassified examples
        if name == "Standard Softmax":
            plot_misclassified(model, test_images, test_labels)
    
    # Compare results
    print("\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)
    
    print(f"\n{'Method':<20} {'Accuracy':<10} {'Loss':<10} {'Avg Conf':<10} {'Correct':<10} {'Incorrect'}")
    print("-" * 75)
    
    for name, metrics in results.items():
        print(f"{name:<20} {metrics['accuracy']:.4f}     {metrics['loss']:.4f}     "
              f"{metrics['avg_confidence']:.4f}     {metrics['confidence_correct']:.4f}     "
              f"{metrics['confidence_incorrect']:.4f}")
    
    # Find best model
    best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
    print(f"\nBest model by accuracy: {best_model[0]} with {best_model[1]['accuracy']:.4f}")
    
    # Analyze confidence calibration
    print("\nConfidence Calibration Analysis:")
    for name, metrics in results.items():
        confidence_gap = metrics['confidence_correct'] - metrics['confidence_incorrect']
        print(f"{name:<20} - Confidence gap: {confidence_gap:.4f}")
    
    return results


if __name__ == "__main__":
    run_mnist_experiment()
