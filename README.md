# Disaster Response Pipeline Project

## Summary
Data Engineering Project for Udacity Data Science Nanodegree. Aim is to set up ETL und ML-Pipelines for a text classification task and visualize the results in a web app.

### Project Components

#### 1. ETL Pipeline:
  Loads the messages and categories datasets
  Merges the two datasets
  Cleans the data
  Stores it in a SQLite database
#### 2. ML Pipeline:
  Loads data from the SQLite database
  Splits the dataset into training and test sets
  Builds a text processing and machine learning pipeline
  Trains and tunes a model using GridSearchCV
  Outputs results on the test set
  Exports the final model as a pickle file
#### 3. Flask Web App
  Data visualizations using Plotly

## Instructions:

Run the following commands in the project's root directory to set up your database and model.

### 1. To run ETL pipeline that cleans data and stores in database:

  python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db

### 2. To run ML pipeline that trains classifier and saves:

  python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl

### 3. Run the following command in the app's directory to run your web app:

  python run.py

  Go to http://0.0.0.0:3001/

## Important Files
### data:

process_data.py: The data cleaning pipeline

disaster_messages.csv: Data of the messages

disaster_categories.csv: Data of the categories

### models:

train_classifier.py: The machine learning pipeline 

### app:

run.py: Flask Web App



