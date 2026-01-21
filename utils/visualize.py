import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# 绘制柱状图
def visualize_layer_importance(avg_score, task_id=None, save_path=None):
    L = len(avg_score)
    plt.figure(figsize=(10, 4))
    plt.bar(np.arange(L), avg_score)
    plt.xlabel("Layer Index")
    plt.ylabel("Importance Score")
    plt.title(f"Layer Importance after Task {task_id}" if task_id is not None else "Layer Importance")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

# 绘制热图
def plot_layer_importance_heatmap(all_scores, save_path=None):
    """
    all_scores: List of [L] arrays, shape = [num_tasks, L]
    """
    scores_matrix = np.stack(all_scores, axis=0)  # [T, L]
    plt.figure(figsize=(12, 6))
    sns.heatmap(scores_matrix, annot=True, cmap="viridis")
    plt.xlabel("Layer Index")
    plt.ylabel("Task ID")
    plt.title("Layer Importance Across Tasks")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()