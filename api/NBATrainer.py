#!/usr/bin/env python
"""
NBA Trainer class definition and methods
(Brest, France)
"""
__author__ = "OUALI Maher"

from .NBADataImputer import NBADataImputer
from .NBAInvestmentModel import NBAInvestmentModel
from sklearn.metrics import recall_score
from sklearn.model_selection import KFold
import itertools
import pandas as pd
from .constants import *
import numpy as np

class NBATrainer():
    """
    A class to handle the training of the the NBA Investement problem model it takes care of data reading, cleaning and model training and validations
    """
    verbose = VERBOSE
    def __init__(self, data_path=None, verbose_=None):
        if not(verbose_) == self.verbose:
            self.verbose = verbose_
        if data_path is None:
            data_path = DEFAULT_DATA_PATH
        self.data_path = data_path
        if self.verbose:
            print("Fetching data from %s"%self.data_path)
        self.data = pd.read_csv(self.data_path)
        self._best_model = None
    
    def _create_grid(self):
        """
        Transform the grid search param dictionnary to a list of all possible combination of values for different hyperparameters

        ---------------------------
        Example: for input: {
                    "n_epochs": [10,20],
                    "depth": [1,2,3]
                }
                
                it creates output: [
                    {"n_epochs": 10, "depth":1}, {"n_epochs": 10, "depth":2}, {"n_epochs": 10, "depth":3}
                    {"n_epochs": 20, "depth":1}, {"n_epochs": 20, "depth":2}, {"n_epochs": 20, "depth":3}
                ]
        ---------------------------

        :return None
        """
        labels, terms = zip(*self.grid_search_parameters.items())
        self._grid = [dict(zip(labels, term)) for term in itertools.product(*terms)]

    def train(self, grid_search_param=None, n_splits=3, **kwargs):
        """
        Train and validate an NBA Investement model using KFolds for every possible configuration in the grid search

        :param grid_search_param(dict): a dictionary with all possible values for different CatBoostClassifier hyperparameters
        :param n_splits(int): an integer defining the number of splits in the folds for each configuration

        :return None
        """
        self.number_splits = n_splits
        if grid_search_param is None:
            grid_search_param = DEFAULT_GRID_SEARCH_PARAM
        self.grid_search_parameters = grid_search_param
        self._create_grid()
        # clean and process
        self.imputer = NBADataImputer()
        self.imputer.fit(self.data)
        if self.verbose:
            print(self.imputer.to_dict()["X"].head())
        self._scores = []
        for param_dict in self._grid:
            kf = KFold(n_splits=self.number_splits,random_state=50,shuffle=True)
            model = NBAInvestmentModel(model_param=param_dict)
            score = 0
            for training_ids,test_ids in kf.split(self.imputer.to_dict()["X"]):
                X_train = self.imputer.to_dict()["X"].loc[training_ids]
                y_train = self.imputer.to_dict()["y"].loc[training_ids]
                X_test = self.imputer.to_dict()["X"].loc[test_ids]
                y_test = self.imputer.to_dict()["y"].loc[test_ids][TARGET_VARIABLE].values
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                score += recall_score(y_test, y_pred)
            self._scores.append(score/self.number_splits)
        # store configuration with best score
        self._best_parameters = self._grid[np.argmax(self._scores)]
        self._best_model = NBAInvestmentModel(model_param=self._best_parameters)
        self._best_model.fit(**self.imputer.to_dict(), force_save=True)
        if self.verbose:
            print("Finished crossvalidating the model on available dataset.\nBest parameters are : \n%s"%self._best_parameters)
        
    def predict(self, X):
        """
        Predict for a set of features whether the given player would have a 5+ years of career at the nba

        :param X: features of the player in questions
        
        :returns: 0(No) or 1(Yes)
        """
        if self._best_model is None:
            self.train()
        return self._best_model.predict(X)

    def predict_proba(self, X):
        """
        Predict for a set of features how much probable the given player would have a 5+ years of career at the nba

        :param X: features of the player in questions
        
        :returns: float
        """
        if self._best_model is None:
            self.train()
        return self._best_model.predict_proba(X)

    def get_best_model(self):
        """
        Returns best model after cross-validation using the KFold/Recall based metric
        """
        if self._best_model is None:
            self.train()
        return self._best_model
