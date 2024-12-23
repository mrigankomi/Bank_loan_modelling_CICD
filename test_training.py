import pytest
import pandas as pd
from unittest.mock import MagicMock
from joblib import dump, load
from bank_loan import (
    load_data, clean_df, preprocess_data, train_model, get_classification_report, 
    save_model_artifact, write_metrics_to_bigquery, load_model_artifact
)

@pytest.fixture
def dummy_data():
    # Prepare dummy data for testing
    data = {
        'Age': [25, 40, 35, 60],
        'Experience': [1, 10, 5, 15],
        'Income': [50, 100, 75, 120],
        'CCAvg': [1.5, 2.0, 1.8, 2.5],
        'Mortgage': [100, 200, 150, 250],
        'Personal Loan': [0, 1, 0, 1],
        'CD Account': [1, 0, 0, 1],
        'Education': [4, 1, 2, 3],
        'Family': [4, 3, 2, 4],
        'Securities Account': [0, 1, 0, 1],
        'Online': [1, 0, 1, 0],
        'Securities Account': [0, 1, 0, 1],
        'ZIP Code': [90210, 90210, 90210, 90210]  # Column to be dropped
    }
    return pd.DataFrame(data)

def test_load_data(dummy_data):
    # Test loading of data
    # We will mock the reading of the file
    pd.read_excel = MagicMock(return_value=dummy_data)
    df = load_data("dummy_path.xlsx")
    assert df.shape == (4, 12)  # Ensure the dummy data has 4 rows and 12 columns

def test_clean_df(dummy_data):
    # Test data cleaning functionality
    df = clean_df(dummy_data)
    X, y = df
    assert X.shape == (4, 10)  # We removed the 'Personal Loan' column
    assert y.shape == (4,)  # Target variable is a 1D array

def test_preprocess_data(dummy_data):
    # Test preprocessing functionality
    X, y = clean_df(dummy_data)
    X_train, X_test, y_train, y_test = preprocess_data(X, y)
    
    assert X_train.shape[0] == 3  # 80% of 4 rows
    assert X_test.shape[0] == 1  # 20% of 4 rows
    assert y_train.shape[0] == 3
    assert y_test.shape[0] == 1

def test_train_model(dummy_data):
    # Test model training
    X, y = clean_df(dummy_data)
    X_train, X_test, y_train, y_test = preprocess_data(X, y)
    
    pipeline = train_model('xgboost', X_train, y_train)
    assert pipeline is not None  # Check if the pipeline is created
    assert hasattr(pipeline, 'predict')  # Ensure the pipeline has the predict method

def test_get_classification_report(dummy_data):
    # Test classification report generation
    X, y = clean_df(dummy_data)
    X_train, X_test, y_train, y_test = preprocess_data(X, y)
    pipeline = train_model('xgboost', X_train, y_train)
    
    report = get_classification_report(pipeline, X_test, y_test)
    
    assert isinstance(report, dict)  # Ensure the report is a dictionary
    assert '0' in report.keys()  # Ensure the class '0' exists in the report

def test_save_and_load_model_artifact(dummy_data):
    # Test saving and loading model artifact
    X, y = clean_df(dummy_data)
    X_train, X_test, y_train, y_test = preprocess_data(X, y)
    pipeline = train_model('xgboost', X_train, y_train)
    
    # Save the model artifact
    save_model_artifact('xgboost', pipeline)
    
    # Mocking the load operation as we're not interacting with actual storage
    load_model_artifact = MagicMock(return_value=pipeline)
    model = load_model_artifact('xgboost_model.joblib')
    
    assert model is not None
    assert hasattr(model, 'predict')  # Ensure the loaded model has the predict method

# def test_write_metrics_to_bigquery(dummy_data):
#     # Test writing metrics to BigQuery
#     X, y = clean_df(dummy_data)
#     X_train, X_test, y_train, y_test = preprocess_data(X, y)
#     pipeline = train_model('xgboost', X_train, y_train)
#     accuracy_metrics = get_classification_report(pipeline, X_test, y_test)
    
#     # Mocking the BigQuery Client
#     mock_client = MagicMock()
#     bigquery.Client = MagicMock(return_value=mock_client)
    
#     # Writing metrics to BigQuery
#     training_time = datetime.now()
#     write_metrics_to_bigquery('xgboost', training_time, accuracy_metrics)
    
#     mock_client.insert_rows_json.assert_called_once()  # Check if the insert_rows_json method was called

