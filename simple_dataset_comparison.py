"""
Comparison of Softmax Variants on a Simple Generated Dataset
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


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
# SIMPLE NEURAL NETWORK
# =============================================================================

class SimpleNN:
    """Simple 2-layer neural network"""
    
    def __init__(self, softmax_fn, alpha=None, hidden_size=64):
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
    
    def fit(self, X, y, epochs=100, batch_size=32, lr=0.01):
        n_samples, input_size = X.shape
        output_size = len(np.unique(y))
        
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
            
            if (epoch + 1) % 20 == 0:
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
    confidence_correct = np.mean(confidence[correct_mask]) if np.any(correct_mask) else 0
    confidence_incorrect = np.mean(confidence[~correct_mask]) if np.any(~correct_mask) else 0
    
    # Compute cross-entropy loss
    n_classes = y_proba.shape[1]
    y_onehot = np.zeros((len(y_test), n_classes))
    y_onehot[np.arange(len(y_test)), y_test] = 1
    loss = -np.mean(np.sum(y_onehot * np.log(y_proba + 1e-10), axis=1))
    
    return {
        'accuracy': accuracy,
        'loss': loss,
        'avg_confidence': avg_confidence,
        'confidence_correct': confidence_correct,
        'confidence_incorrect': confidence_incorrect
    }


def plot_decision_boundary(model, X, y, title="Decision Boundary"):
    """Plot decision boundary for 2D data"""
    # Set min and max values with some padding
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    
    # Step size in the mesh
    h = 0.02
    
    # Generate a grid of points
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # Predict the function value for the whole grid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot the contour and training examples
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.Paired)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.Paired)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
    plt.savefig(f"{title.replace(' ', '_').lower()}.png")
    plt.close()


def plot_confidence_distribution(models_results, title="Confidence Distribution"):
    """Plot confidence distribution for correct and incorrect predictions"""
    plt.figure(figsize=(12, 8))
    
    x = np.arange(len(models_results))
    width = 0.35
    
    correct_conf = [results['confidence_correct'] for results in models_results.values()]
    incorrect_conf = [results['confidence_incorrect'] for results in models_results.values()]
    
    plt.bar(x - width/2, correct_conf, width, label='Correct Predictions')
    plt.bar(x + width/2, incorrect_conf, width, label='Incorrect Predictions')
    
    plt.xlabel('Model')
    plt.ylabel('Average Confidence')
    plt.title(title)
    plt.xticks(x, list(models_results.keys()), rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig("confidence_distribution.png")
    plt.close()


def plot_accuracy_vs_confidence(models_results, title="Accuracy vs Confidence"):
    """Plot accuracy vs confidence for each model"""
    plt.figure(figsize=(10, 6))
    
    for name, results in models_results.items():
        plt.scatter(results['avg_confidence'], results['accuracy'], label=name, s=100)
    
    plt.xlabel('Average Confidence')
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("accuracy_vs_confidence.png")
    plt.close()


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def run_experiment():
    """Run experiment comparing softmax variants on synthetic data"""
    print("\n" + "=" * 70)
    print("EXPERIMENT: Comparing Softmax Variants on Synthetic Data")
    print("=" * 70)
    
    # Generate synthetic data
    print("\nGenerating synthetic data...")
    X, y = make_classification(
        n_samples=1000,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_clusters_per_class=1,
        n_classes=4,
        class_sep=1.5,
        random_state=42
    )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    print(f"Classes: {len(np.unique(y))}")
    
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
    trained_models = {}
    
    for name, model in models.items():
        print(f"\n{'-' * 50}")
        print(f"Training {name}...")
        history = model.fit(X_train, y_train, epochs=100)
        trained_models[name] = model
        
        print(f"\nEvaluating {name}...")
        metrics = evaluate_model(model, X_test, y_test)
        results[name] = metrics
        
        print(f"Test Accuracy: {metrics['accuracy']:.4f}")
        print(f"Test Loss: {metrics['loss']:.4f}")
        print(f"Average Confidence: {metrics['avg_confidence']:.4f}")
        print(f"Confidence (Correct): {metrics['confidence_correct']:.4f}")
        print(f"Confidence (Incorrect): {metrics['confidence_incorrect']:.4f}")
        
        # Plot decision boundary
        plot_decision_boundary(model, X_test, y_test, title=f"Decision Boundary - {name}")
    
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
    
    # Plot confidence distribution
    plot_confidence_distribution(results)
    
    # Plot accuracy vs confidence
    plot_accuracy_vs_confidence(results)
    
    # Analyze confidence calibration
    print("\nConfidence Calibration Analysis:")
    for name, metrics in results.items():
        confidence_gap = metrics['confidence_correct'] - metrics['confidence_incorrect']
        print(f"{name:<20} - Confidence gap: {confidence_gap:.4f}")
    
    return results, trained_models


if __name__ == "__main__":
    results, models = run_experiment()
