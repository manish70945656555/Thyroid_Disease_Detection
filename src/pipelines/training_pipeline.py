import os
import sys
import pandas as pd

from src.logger import logging
from src.exception import CustomException
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

def main():
    # Data Ingestion
    train_data_path, test_data_path = perform_data_ingestion()
    print(train_data_path, test_data_path)

    # Data Transformation
    train_arr, test_arr, preprocessor_path = perform_data_transformation(train_data_path, test_data_path)

    # Model Training
    perform_model_training(train_arr, test_arr)

def perform_data_ingestion():
    data_ingestion = DataIngestion()
    train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()
    return train_data_path, test_data_path

def perform_data_transformation(train_data_path, test_data_path):
    data_transformation = DataTransformation()
    train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation(train_data_path, test_data_path)
    return train_arr, test_arr, preprocessor_path

def perform_model_training(train_arr, test_arr):
    model_trainer = ModelTrainer()
    model_trainer.initiate_model_training(train_arr, test_arr)

if __name__ == '__main__':
    main()
