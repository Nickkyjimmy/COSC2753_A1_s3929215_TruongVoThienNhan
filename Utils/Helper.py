import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import ticker
from sklearn.metrics import confusion_matrix, classification_report, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV


def impute_outliers_iqr(col, df, imputer):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
    
    # Impute outliers using KNN imputer
    if np.sum(outliers_mask) > 0:
        imputed_values = imputer.fit_transform(df[col].values.reshape(-1, 1))
        df[col] = np.where(outliers_mask, imputed_values.flatten(), df[col])

def cap_outliers_iqr(col, df):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Cap remaining outliers
    df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
    df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

def plot_confusion_matrices(true_labels_train, predicted_labels_train, true_labels_val, predicted_labels_val, title: str = None,
                                    save_path: str = None, save_dpi: int = 300):
    classes= ["Negative", "Positive"]
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))  # Create subplot with two columns
    
    # Plot confusion matrix for training performance
    cm_train = confusion_matrix(true_labels_train, predicted_labels_train)
    axes[0].imshow(cm_train, interpolation='nearest', cmap=plt.cm.Blues)
    axes[0].set_title('Train Confusion Matrix')
    axes[0].set_xticks(np.arange(len(classes)))
    axes[0].set_yticks(np.arange(len(classes)))
    axes[0].set_xticklabels(classes)
    axes[0].set_yticklabels(classes)
    thresh_train = cm_train.max() / 2.
    for i in range(cm_train.shape[0]):
        for j in range(cm_train.shape[1]):
            axes[0].text(j, i, format(cm_train[i, j], 'd'), ha="center", va="center", color="white" if cm_train[i, j] > thresh_train else "black")

    # Plot confusion matrix for validation performance
    cm_val = confusion_matrix(true_labels_val, predicted_labels_val)
    axes[1].imshow(cm_val, interpolation='nearest', cmap=plt.cm.Blues)
    axes[1].set_title('Validation Confusion Matrix')
    axes[1].set_xticks(np.arange(len(classes)))
    axes[1].set_yticks(np.arange(len(classes)))
    axes[1].set_xticklabels(classes)
    axes[1].set_yticklabels(classes)
    thresh_val = cm_val.max() / 2.
    for i in range(cm_val.shape[0]):
        for j in range(cm_val.shape[1]):
            axes[1].text(j, i, format(cm_val[i, j], 'd'), ha="center", va="center", color="white" if cm_val[i, j] > thresh_val else "black")

    plt.tight_layout()
    plt.show()

    if title is not None:
        fig.suptitle(title, fontsize=16, fontweight="bold", y=1.2)

    plt.show()
    if save_path is not None:
        fig.savefig(save_path, dpi=save_dpi)
    print(f"{'TRAINING PERFORMANCE'}\n{classification_report(true_labels_train, predicted_labels_train)}")
    print(f"{'VALIDATION PERFORMANCE'}\n{classification_report(true_labels_val, predicted_labels_val)}")

def print_evaluation_metrics(y_true_train, y_pred_train, y_true_val, y_pred_val, y_proba_train, y_proba_test):
    # Calculate F1 score and ROC AUC score for training set
    f1_train = f1_score(y_true_train, y_pred_train)
    roc_auc_train = roc_auc_score(y_true_train, y_proba_train)

    # Calculate F1 score and ROC AUC score for validation set
    f1_val = f1_score(y_true_val, y_pred_val)
    roc_auc_val = roc_auc_score(y_true_val, y_proba_test)

    print("Training Set:")
    print("F1 Score:", f1_train)
    print("ROC AUC Score:", roc_auc_train)

    print("\nValidation Set:")
    print("F1 Score:", f1_val)
    print("ROC AUC Score:", roc_auc_val)


import numpy as np
import matplotlib.pyplot as plt

def plot_hyperparameter_tuning_results(grid_clf):
    # Extract results
    results = grid_clf.cv_results_
    mean_f1_scores = results['mean_test_f1']
    mean_train_f1_scores = results['mean_train_f1']
    mean_roc_auc_scores_train = results['mean_train_roc_auc']
    mean_roc_auc_scores_val = results['mean_test_roc_auc']
    hyperparameters = results['params']  # assuming hyperparameters are enumerated

    # Get best hyperparameters
    best_index = np.argmax(mean_f1_scores)
    best_hyperparameters = hyperparameters[best_index]

    # Plotting
    fig, ax = plt.subplots(2, figsize=(8, 10))
    
    # Plot F1 Scores
    ax[0].plot(range(len(hyperparameters)), mean_f1_scores, marker='o', linestyle='-', label='Validation F1 Score')
    ax[0].plot(range(len(hyperparameters)), mean_train_f1_scores, marker='o', linestyle='-', label='Training F1 Score')
    # ax[0].scatter(best_index, mean_f1_scores[best_index], color='red', label='Best Hyperparameters')
    # ax[0].axvline(x=best_index, color='red', linestyle='--')  # Add a vertical line to mark the best hyperparameter
    ax[0].set_title('Hyperparameter Tuning F1 Score')
    ax[0].set_xlabel('Hyperparameter Combinations')
    ax[0].set_ylabel('F1 Score')
    ax[0].set_ylim(0, 1)  # Setting y-axis limit
    ax[0].legend()
    ax[0].grid(True)

    # Plot ROC AUC Scores
    ax[1].plot(range(len(hyperparameters)), mean_roc_auc_scores_train, marker='o', linestyle='-', label='Training ROC AUC Score')
    ax[1].plot(range(len(hyperparameters)), mean_roc_auc_scores_val, marker='o', linestyle='-', label='Validation ROC AUC Score')
    # ax[1].scatter(best_hyperparameters, mean_roc_auc_scores_val[best_index], color='red', label='Best Hyperparameters')
    # ax[1].axvline(x=best_hyperparameters, color='red', linestyle='--')  # Add a vertical line to mark the best hyperparameter
    ax[1].set_title('Hyperparameter Tuning ROC AUC Score')
    ax[1].set_xlabel('Hyperparameter Combinations')
    ax[1].set_ylabel('ROC AUC Score')
    ax[1].legend()
    ax[1].grid(True)

    plt.tight_layout()
    plt.show()

