import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, log_loss

import mlflow
import mlflow.sklearn

DATASET_PATH = 'MLProject/student_performance_prediction_preprocessing.csv'

def plot_confusion_matrix(y_true, y_predict, img_name):
    conf_matrix = confusion_matrix(y_true, y_predict)

    plt.figure(figsize = (6, 4))
    sns.heatmap(conf_matrix, annot = True, fmt = 'd', cmap = 'YlGnBu')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(img_name)
    plt.close()

def main():
    student_performance_prediction_df = pd.read_csv(DATASET_PATH)
    X = student_performance_prediction_df.drop(columns = ['Passed'])
    y = student_performance_prediction_df['Passed']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 7, stratify = y)

    param_grid = {
        'C': [0.01, 0.1, 1, 10],
        'class_weight': [None, 'balanced'],
        'solver': ['lbfgs']
    }

    base_model = LogisticRegression(max_iter = 1000)
    grid = GridSearchCV(estimator = base_model, param_grid = param_grid, scoring = 'f1', cv = 5, n_jobs = -1)

    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    y_predict = best_model.predict(X_test)
    y_probability = best_model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_predict)
    precision = precision_score(y_test, y_predict)
    recall = recall_score(y_test, y_predict)
    f1 = f1_score(y_test, y_predict)
    roc_auc = roc_auc_score(y_test, y_probability)
    log_loss_score = log_loss(y_test, y_probability)
    training_score = best_model.score(X_test, y_test)

    with mlflow.start_run(run_name = 'logistic-regression_skilled'):
        mlflow.log_params(grid.best_params_)
        mlflow.log_metric('training_accuracy_score', accuracy)
        mlflow.log_metric('training_precision_score', precision)
        mlflow.log_metric('training_recall_score', recall)
        mlflow.log_metric('training_f1_score', f1)
        mlflow.log_metric('training_roc_auc', roc_auc)
        mlflow.log_metric('training_log_loss', log_loss_score)
        mlflow.log_metric('training_score', training_score)
        mlflow.sklearn.log_model(best_model, artifact_path = 'model', input_example = X_test.iloc[:5])

        conf_matrix_path = 'artifact/confusion_matrix.png'

        plot_confusion_matrix(y_test, y_predict, conf_matrix_path)
        mlflow.log_artifact(conf_matrix_path)

        print(f"Best parameters: {grid.best_params_}")
        print(f"Accuracy: {accuracy * 100:.2f}%")
        print(f"Precision: {precision * 100:.2f}%")
        print(f"Recall: {recall * 100:.2f}%")
        print(f"F1-Score: {f1 * 100:.2f}%")

if __name__ == '__main__':
    main()
