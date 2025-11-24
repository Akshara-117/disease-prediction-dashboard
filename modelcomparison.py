import matplotlib.pyplot as plt
import numpy as np

# Data
models = ['DecisionTree', 'RandomForest', 'NaiveBayes', 'SVM', 'Stacking']
accuracy = [0.78, 0.88, 0.81, 0.83, 0.90]
f1_scores = [0.74, 0.85, 0.77, 0.79, 0.87]

x = np.arange(len(models))
width = 0.35

plt.figure(figsize=(9,5))
bars1 = plt.bar(x - width/2, accuracy, width, color='#4A90E2', label='Accuracy')
bars2 = plt.bar(x + width/2, f1_scores, width, color='#F5A623', label='F1-Score')

# Add values on top
for bar in bars1 + bars2:
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{bar.get_height():.2f}', ha='center', fontsize=9, color='black')

# Add connecting line for ranking
plt.plot(x, accuracy, 'o--', color='green', linewidth=1.5, label='Accuracy Trend')

plt.xticks(x, models)
plt.ylabel('Score')
plt.ylim(0.6, 1.0)
plt.title('Model Comparison: Accuracy vs F1-Score (with Ranking)')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("model_comparison_enhanced.png", dpi=300)
plt.show()
