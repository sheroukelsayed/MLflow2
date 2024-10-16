import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris  # Replace with your own dataset
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, f1_score

# Load your dataset (replace with your customer churn dataset)
df = pd.read_csv('data/churn_final.csv')  # Ensure this path is correct
X = df.drop('Churn', axis=1)  # Features
y = df['Churn']  # Target variable

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Set up MLflow experiment
mlflow.set_experiment('Customer_Churn_Prediction')


# Hyperparameters to try
n_estimators_list = [50, 100, 150]
max_depth_list = [None, 10, 20]

# Iterate over hyperparameters
for n_estimators in n_estimators_list:
    for max_depth in max_depth_list:
        with mlflow.start_run(run_name=f'RandomForest_n{n_estimators}_d{max_depth}'):
            # Initialize the model with current hyperparameters
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
            
            # Log hyperparameters
            mlflow.log_param('model_name', 'RandomForestClassifier')
            mlflow.log_param('n_estimators', n_estimators)
            mlflow.log_param('max_depth', max_depth)
            
            # Train and evaluate the model
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            # Calculate accuracy
            accuracy = accuracy_score(y_test, y_pred)
            # After making predictions
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')

            # Log additional metrics
            mlflow.log_metric('precision', precision)
            mlflow.log_metric('recall', recall)
            mlflow.log_metric('f1_score', f1)
            # Log the accuracy metric
            mlflow.log_metric('accuracy', accuracy)
        
            
            # Log the model
            mlflow.sklearn.log_model(model, artifact_path='model')
            
            print(f'RandomForestClassifier (n_estimators={n_estimators}, max_depth={max_depth}) accuracy: {accuracy}')
