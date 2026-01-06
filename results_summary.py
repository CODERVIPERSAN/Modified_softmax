"""
Results Summary for Modified Softmax Comparison
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image

def create_summary_visualization():
    """Create a comprehensive summary visualization of all results"""
    
    # Load all the generated images
    decision_boundaries = [
        Image.open('decision_boundary_-_standard_softmax.png'),
        Image.open('decision_boundary_-_modified_softmax.png'),
        Image.open('decision_boundary_-_weighted_(α=0.05).png'),
        Image.open('decision_boundary_-_weighted_(α=0.1).png'),
        Image.open('decision_boundary_-_weighted_(α=0.5).png')
    ]
    
    confidence_dist = Image.open('confidence_distribution.png')
    accuracy_conf = Image.open('accuracy_vs_confidence.png')
    
    # Create figure
    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(3, 5, figure=fig)
    
    # Plot decision boundaries
    model_names = [
        "Standard Softmax",
        "Modified Softmax",
        "Weighted (α=0.05)",
        "Weighted (α=0.1)",
        "Weighted (α=0.5)"
    ]
    
    for i, (img, name) in enumerate(zip(decision_boundaries, model_names)):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(img)
        ax.set_title(name, fontsize=12)
        ax.axis('off')
    
    # Plot confidence distribution
    ax_conf = fig.add_subplot(gs[1, :3])
    ax_conf.imshow(confidence_dist)
    ax_conf.axis('off')
    
    # Plot accuracy vs confidence
    ax_acc = fig.add_subplot(gs[1, 3:])
    ax_acc.imshow(accuracy_conf)
    ax_acc.axis('off')
    
    # Add results table
    results_data = [
        ["Standard Softmax", "0.9450", "0.2430", "0.9071", "0.9222", "0.6469", "0.2754"],
        ["Modified Softmax", "0.9450", "0.3259", "0.8812", "0.8957", "0.6311", "0.2646"],
        ["Weighted (α=0.05)", "0.9450", "0.3668", "0.8391", "0.8525", "0.6095", "0.2430"],
        ["Weighted (α=0.1)", "0.9450", "0.3525", "0.8513", "0.8651", "0.6133", "0.2519"],
        ["Weighted (α=0.5)", "0.9450", "0.3294", "0.8738", "0.8882", "0.6261", "0.2622"]
    ]
    
    col_labels = ["Model", "Accuracy", "Loss", "Avg Conf", "Conf (Correct)", "Conf (Incorrect)", "Conf Gap"]
    
    ax_table = fig.add_subplot(gs[2, :])
    table = ax_table.table(
        cellText=results_data,
        colLabels=col_labels,
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)
    ax_table.axis('off')
    
    # Add summary text
    plt.figtext(0.5, 0.02, """
    Key Findings:
    1. All models achieved the same accuracy (94.5%), showing all softmax variants can perform well for classification
    2. Standard Softmax had the lowest loss and highest confidence (90.7%)
    3. Weighted Modified Softmax with α=0.05 had the most balanced confidence (83.9%)
    4. Standard Softmax had the largest confidence gap between correct/incorrect predictions (27.5%)
    5. Modified approaches may be better calibrated for uncertainty
    """, ha='center', fontsize=14, bbox=dict(facecolor='white', alpha=0.8))
    
    # Save the summary visualization
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig('softmax_comparison_summary.png', dpi=150)
    plt.close()
    
    print("Created comprehensive summary visualization: softmax_comparison_summary.png")


if __name__ == "__main__":
    create_summary_visualization()
