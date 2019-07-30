# DSND-DisasterResponse

## Description
This project is part of Udacity's Data Science Nanodegree assignment. It analyses disaster text message data from Figure Eight to build a classifier powered API to tag messages against their categories. 

The project is divided into three main components
1. ETL - Extracts, cleans and loads data into an SQL database
2. Model training - A machine learning pipeline that uses NLTK, scikit-learn's Pipeline and GridSearchCV to conduct multioutput classification using random forest. It outputs the final model into a pickle file (Please note that the pickle file cannot be uploaded as the size exceeds the maximum allowed on github. The model will need to be retrained should users wish to run it on a local machine.)
3. Web app - After users enter new messages, the Flask web app will tag the messages with different categories using the trained model. It also displays simple visualisation of the dataset.

## Dependencies
Python 3.5 is used to create this project
- Machine Learning Libraries: NumPy, Pandas, Sciki-Learn
- Natural Language Processing: NLTK, genism
- SQLite Database: SQLalchemy
- Web App: Flask
- Data Visualisation: Plotly

## Installation

1. First, clone or download this GIT repository

2. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

3. Run the following command in the app's directory to run your web app.
    `python run.py`

4. Go to http://0.0.0.0:3001/

## Credits
Licensed under a Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License - Please refer to Udacity Terms of Service for further information.

Udacity and FigureEight - for providing the dataset and designing the project
