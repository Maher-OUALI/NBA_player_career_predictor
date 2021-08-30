#!/usr/bin/env python
"""
Constants holder to manage model_path, model default and best parameters
(Brest, France)
"""
__author__ = "OUALI Maher"

import os
from config import BASE_DIR

# absolute boolean variables
VERBOSE = True
SAVE_MODEL = True

# data related constants
DEFAULT_DATA_PATH = os.path.join(BASE_DIR, "static\\data\\nba_logreg.csv")
FEATURES_COLUMNS = ["GP", "MIN" , "PTS", "FGM", "FGA", "FG%", "3P Made",
        "3PA", "3P%", "FTM", "FTA", "FT%", "OREB", "DREB", "REB", "AST",
        "STL", "BLK", "TOV"]
TARGET_VARIABLE = "TARGET_5Yrs"
DTYPES = {
    "GP":"int", 
    "MIN":"float64",
    "PTS":"float64",
    "FGM":"float64",
    "FGA":"float64",
    "FG%":"float64",
    "3P Made":"float64",
    "3PA":"float64",
    "3P%":"float64",
    "FTM":"float64",
    "FTA":"float64",
    "FT%":"float64",
    "OREB":"float64",
    "DREB":"float64",
    "REB":"float64",
    "AST":"float64",
    "STL":"float64",
    "BLK":"float64",
    "TOV":"float64",
    "TARGET_5Yrs":"int"
}

# modelisation related constants
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "catboost_models\\nba_investment_catboost")
DEFAULT_MODEL_PARAM = {
    "loss_function":'Logloss',
    "learning_rate":0.005,
    "depth":4,
    "early_stopping_rounds":50,
}
DEFAULT_SPLIT_PARAMETERS = {
    "test_size":0.1,
    "random_state":50,
    "shuffle":True
}
DEFAULT_FIT_PARAMETERS = {
    "use_best_model":True,
}
DEFAULT_GRID_SEARCH_PARAM = {
    'iterations' : [1000, 5000, 10000],
    'loss_function' : ['Logloss'],
    'depth' : list(range(3,6)),
    'early_stopping_rounds' : [20,50, 70],
    'verbose' : [False],
}
BEST_FIT_PARAMETERS = {
    'iterations' : [5000],
    'loss_function' : ['Logloss'],
    'depth' : [3],
    'early_stopping_rounds' : [50],
    'verbose' : [False],
}
