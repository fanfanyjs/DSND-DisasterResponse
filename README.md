# DSND-DisasterResponse
This project is part of Udacity's Data Science Nanodegree assignment. It analyses disaster text message data from Figure Eight to build a classifier powered API to tag messages against their categories. 

## Dependencies
- Machine Learning Libraries: NumPy, SciPy, Pandas, Sciki-Learn
- Natural Language Process Libraries: NLTK, genism
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
