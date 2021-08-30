This is a technical test given by MP Data and it deals with a MLOps solution to deploy a NBA Investment Model using a Flask API

1. APP FOLDER ARCHTECTURY

APP
|__ api
|     |__ constants.py: useful constants and parameters of the modelisation and data processing
|     |__ NBADataImputer: Handles data processing and cleaning
|     |__ NBAInvestmentModel: Holds the catboost classifier
|     |__ NBATrainer: handles the whole pipeline from data cleaning to training and validation with grid search to tune hyperparameters
|
|__ catboost_models: stored catboost classifier weights with a json file to indicate best validation score (models are updated only if newer best score is better than older)
|
|__ static: docs and data .csv
|
|__ templated: 
|     |__ list_investments.html: view list of all prediction entries
|     |__ nba_player_form.html: enter new player stats
|     |__ new_investment.html: view prediction of added player
|
|__ config.py: config file to hold app constants
|
|__ requirements.txt: file of all package requirements to install at the begining
|
|__ investment.db: sql database to store all prediction entries
|
|__ run.py: main flask app code that defines Investment model and routing between different html catboost_models

2. HOW TO run

- cmd pip install -r requirements.txt to download all needed packages for the project
- make sure training dataset is found in ./static/data with coherent filename as in ./api/constants.py (DEFAULT_DATA_PATH)
- cmd python run.py to launch the Flask web server, wait for the model to train for the first time and then launch the given url in the browser
(default: http://localhost:5000/)
- Enjoy the experience :)

3. AUTHOR

- Name: Maher OUALI
- email: maher.ouali1996@gmail.com 
- phone: +330613616675















