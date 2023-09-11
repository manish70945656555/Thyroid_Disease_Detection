# Thyroid Disease Detection

Thyroid disease is a very common problem in India, more than one crore people are suffering with the disease every year. Thyroid disorder can speed up or slow down the metabolism of the body.

The main objective of this project is to predict if a person is having hypothyroid or negative (no thyroid) with the help of Machine Learning. Classification algorithms such as Random Forest, Decision Tree and KNN Model have been trained on the thyroid dataset, UCI Machine Learning repository. After that Random Forest classfier model has performed well with better accuracy, precision and recall. Application has deployed on AWS with the help of flask framework.

# Webpage Link

## For Prediction

AWS: https://2hxjm2y4me.us-east-1.awsapprunner.com

# Demo






# Technical Aspects

- Python 3.7 and more
- Important Libraries: sklearn, pandas, numpy, matplotlib & seaborn
- Front-end: HTML, CSS 
- Back-end: Flask framework
- IDE: Jupyter Notebook & VSCode
- Database: Mongo DB
- Deployment: AWS

# How to run this app 

Code is written in Python 3.7 and more. If you don't have python installed on your system, click here https://www.python.org/downloads/ to install.

- Create virtual environment - conda create -p venv python=3.8 -y
- Activate the environment - conda activate venv
- Install the packages - pip install -r requirements.txt
- Run the app - python run app.py

# Workflow

## Data Collection

Thyroid Disease Data Set from UCI Machine Learning Repository.

Link:https://archive.ics.uci.edu/ml/datasets/thyroid+disease

## Data Pre-processing

- Missing values handling by Simple imputation (Simple Imputer)
- Categorical features handling by mapping and encoding.
- Feature scaling done by Minmax Scalar method
- Imbalanced dataset handled by SMOTE
- Drop unnecessary columns

## Model Creation and Evaluation

- Various classification algorithms like Random Forest, Decision Tree,SVC, KNN etc tested.
- Random Forest, XGBoost and KNN were all performed well. Random Forest was chosen for the final model training and testing.
- Model performance evaluated based on accuracy, confusion matrix, classification report.


## Database Connection
MongoDB database used for this project.

## Model Deployment
The final model is deployed on AWS using Flask framework.

## User Interface
### Batch File Prediction User Interface
#### Homepage: We will have a single page UI which will facilitate bulk prediction for Thyroid Disease Detection. 
![Homepage](https://github.com/manish70945656555/Thyroid_Disease_Detection/assets/111861277/c5ae009f-db17-4714-a2a8-188f41144b5b)

#### First thing anyone will see is a button on Homepage for predict new data , if anyone click on that browser redirect to other page for taking prediction data.

![Predict-Button](https://github.com/manish70945656555/Thyroid_Disease_Detection/assets/111861277/462123fe-71c3-44cf-8958-7f6df2d10706)

#### On Home page we will have an option which enable user to download sample submission CSV file for reference.

![Sample-file-button](https://github.com/manish70945656555/Thyroid_Disease_Detection/assets/111861277/4eca605b-d5ce-4799-b70f-d4456c11150e)

#### When Client click on predict new data buutton then client will redirect to new page and the    First thing client will see is a pop-up window on Homepage which will ask for CSV file for prediction.

![Pop-Up](https://github.com/manish70945656555/Thyroid_Disease_Detection/assets/111861277/5bb13a9d-f253-49a6-b450-e2b142ba1ede)

#### After choosing the csv file for prediction, user has to click on upload file button for prediction. And then automatically a new pop up window will come which give option to download the prediction in csv file format.

![File-Choosing](https://github.com/manish70945656555/Thyroid_Disease_Detection/assets/111861277/5bdeefb2-7a8c-414b-aff8-f444f43b28d1)

#### Prediction CSV file will contain a new column target which show person is suffering from hypothyroid or not.

![Predicted-csv](https://github.com/manish70945656555/Thyroid_Disease_Detection/assets/111861277/c08ff395-88a2-4758-8c0b-4c122b243555)


## Project Documents

- HLD: https://github.com/manish70945656555/Thyroid_Disease_Detection/blob/main/Docs/TDD_HLD_V1.0.pdf

- LLD: https://github.com/manish70945656555/Thyroid_Disease_Detection/blob/main/Docs/TDD_LLD_V1.0.pdf

- Architecture: https://github.com/manish70945656555/Thyroid_Disease_Detection/blob/main/Docs/TDD_Architecture_V1.0.pdf

- Wireframe: https://github.com/manish70945656555/Thyroid_Disease_Detection/blob/main/Docs/TDD_Wireframe_V1.0.pdf

- Detailed Project Report: https://github.com/manish70945656555/Thyroid_Disease_Detection/blob/main/Docs/TDD_DPR.pdf


# Author

Manish Kumawat: https://www.linkedin.com/in/manish-kumawat-8b27a224b


# Help Me Improve

Hello Reader if you find any bug please consider raising issue I will address them asap.

