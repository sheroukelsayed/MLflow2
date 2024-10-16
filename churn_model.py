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

# Define a list of models to try
models = {
    'GradientBoostingClassifier': GradientBoostingClassifier(),
    'RandomForestClassifier': RandomForestClassifier(),
    'LogisticRegression': LogisticRegression(max_iter=1000),
    'SupportVectorMachine': SVC()
}

# Iterate over the models
for model_name, model in models.items():
    with mlflow.start_run(run_name=model_name):
        # Log the model name as a parameter
        mlflow.log_param('model_name', model_name)
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
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
        
        # Log the model parameters (if available)
        if hasattr(model, 'get_params'):
            params = model.get_params()
            mlflow.log_params(params)
        
        # Log the model
        mlflow.sklearn.log_model(model, artifact_path='model')
        
        print(f'{model_name} accuracy: {accuracy}')
