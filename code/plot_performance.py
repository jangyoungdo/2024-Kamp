import matplotlib.pyplot as plt
import numpy as np

def plot_stacked_model_performance(results_df):
    metrics = ['Stacked Model Accuracy', 'Stacked Model Precision', 'Stacked Model Recall',
               'Stacked Model F1 Score', 'Stacked Model ROC AUC']
    plt.figure(figsize=(12, 8))
    stages = results_df['Stage']
    bar_width = 0.15
    index = np.arange(len(stages))

    for i, metric in enumerate(metrics):
        plt.bar(index + i * bar_width, results_df[metric], bar_width, label=metric)

    plt.xlabel('Stage (Model Combination)')
    plt.ylabel('Score')
    plt.title('Comparison of Stacked Model Performance by Metric')
    plt.xticks(index + bar_width * (len(metrics) - 1) / 2, stages)
    plt.legend()
    plt.tight_layout()
    plt.show()
