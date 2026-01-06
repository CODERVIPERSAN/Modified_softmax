"""
Flask UI for Modified Softmax Comparison
"""

from flask import Flask, render_template, request, jsonify
import numpy as np

app = Flask(__name__)


# =============================================================================
# SOFTMAX FUNCTIONS
# =============================================================================

def sigmoid(x):
    """Sigmoid function with numerical stability"""
    x = np.array(x, dtype=float)
    return np.where(x >= 0, 
                    1 / (1 + np.exp(-x)), 
                    np.exp(x) / (1 + np.exp(x)))


def standard_softmax(logits):
    """Standard softmax: exp(x) / sum(exp(x))"""
    logits = np.array(logits, dtype=float)
    logits = logits - np.max(logits)  # numerical stability
    exp_logits = np.exp(logits)
    return (exp_logits / np.sum(exp_logits)).tolist()


def modified_softmax(logits):
    """Modified Softmax: sigmoid(x) / sum(sigmoid(x))"""
    sig_logits = sigmoid(logits)
    return (sig_logits / np.sum(sig_logits)).tolist()


def weighted_modified_softmax(logits, alpha=0.1):
    """Weighted Modified Softmax: sigmoid(αx) / sum(sigmoid(αx))"""
    scaled_logits = alpha * np.array(logits, dtype=float)
    sig_logits = sigmoid(scaled_logits)
    return (sig_logits / np.sum(sig_logits)).tolist()


# =============================================================================
# ROUTES
# =============================================================================

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/compute', methods=['POST'])
def compute():
    data = request.json
    values = [float(v.strip()) for v in data['values'].split(',') if v.strip()]
    alpha = float(data.get('alpha', 0.1))
    
    if len(values) < 2:
        return jsonify({'error': 'Need at least 2 values'}), 400
    
    results = {
        'values': values,
        'labels': [f'Class {i}' for i in range(len(values))],
        'standard': standard_softmax(values),
        'modified': modified_softmax(values),
        'weighted': weighted_modified_softmax(values, alpha),
        'alpha': alpha
    }
    
    return jsonify(results)


if __name__ == '__main__':
    app.run(debug=True, port=5000)
