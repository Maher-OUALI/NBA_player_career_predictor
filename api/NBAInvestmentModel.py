#!/usr/bin/env python
"""
NBA Investment model definition and methods
(Brest, France)
"""
__author__ = "OUALI Maher"

#imports
import pandas as pd
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from .constants import * 
import os
import json


class NBAInvestmentModel():
    """
    A class to handle the modelisation of the NBA Investment problem using a catboost based classifier
    """
    verbose = VERBOSE
    save = SAVE_MODEL
    def __init__(self, model_param=None, model_path=None, verbose_=None):
        """
        Contruct NBAInvestmentModel attributes and initialize model
        
        :param model_param(dict): a dict containing initialization parameters for the catboost model
        :param path_model(str): the path of the model content
        """
        if not(verbose_) == self.verbose:
            self.verbose = verbose_
        #initalize attributes either with correct input or default values in constants file
        self.is_fitted=False
        if model_path is None:
            model_path = DEFAULT_MODEL_PATH
        self.model_path=model_path
        # if model parameters are not specified, we load model from either the given path or the default path 
        if model_param is None:
            model_param = DEFAULT_MODEL_PARAM
            try:
                self.model = CatBoostClassifier()
                self.model.load_model(self.model_path)
                self.is_fitted = True
                if self.verbose:
                    print("Finished loading model from %s"%self.model_path)
            except:
                # if we can't load model we create a default catboost classifer
                self.model = CatBoostClassifier(**model_param)
                print("Model path or content not valid abort loading and fit a new one rather")
        else:
            self.model = CatBoostClassifier(**model_param)
            if self.verbose:
                print("Finished creating model with the following parameters \n%s"%model_param)
        self.model_parmeters = model_param

    def fit(self, X, y, split_kwargs=None, fit_kwargs=None, force_save=False):
        """
        Train the model on available cleaned datasets

        :param X: features of nba players cleaned and normalized
        :param y: target of +5 year carreer of nba players
        :param split_kwargs: additional arguments to be added to the split function (must be a dictionary with valid keys)
        :param split_kwargs: additional arguments to be added to the split function (must be a dictionary with valid keys)

        :returns: None
        """
        if split_kwargs is None:
            split_kwargs = DEFAULT_SPLIT_PARAMETERS
        if fit_kwargs is None:
            fit_kwargs = DEFAULT_FIT_PARAMETERS
        # split data to train and validation
        X_train, X_val, y_train, y_val = train_test_split(X, y, **split_kwargs)
        # validation is used to avoid overfitting and tune hyperparameters of the model
        self.model.fit(X_train, y_train,
            eval_set=(X_val, y_val),
            verbose=self.verbose,
            **fit_kwargs)
        self.best_score = self.model.get_best_score()["validation"]
        self.is_fitted=True
        if self.verbose:
            print("Finished training model on available cleaned dataset")
        if self.save:
            try:
                # confirm save is used to check if model has improved best score and therefor save it otherwise keep last model
                confirm_save = True
                try:
                    os.makedirs('\\'.join(self.model_path.split("\\")[:-1]))
                except:
                    # directory already exists
                    # check if best score is better
                    best_score_path = os.path.join('\\'.join(self.model_path.split("\\")[:-1]), "best_score.json")
                    try:
                        old_score = open(best_score_path)
                        old_score = json.load(old_score)
                        for k, v in old_score.items():
                            if k in self.best_score.keys():
                                confirm_save = confirm_save and self.best_score[k] >= v
                    except:
                        pass
                
                if confirm_save or force_save:
                    # save model
                    self.model.save_model(self.model_path)
                    # save best score
                    with open(best_score_path, 'w') as f:
                        json.dump(self.best_score, f)
                    if self.verbose:
                        print("Finished saving model at %s"%self.model_path)
                else:
                    if self.verbose:
                        print("Model didn't enhance best score so abort save")
            except Exception as e:
                print("Got following error: {} \nwhen saving model at {}".format(e, self.model_path))

    def predict(self, X):
        """
        Predict for a set of features whether the given player would have a 5+ years of career at the nba

        :param X: features of the player in questions
        
        :returns: 0(No) or 1(Yes)
        """
        if self.is_fitted:
            return self.model.predict(X)
        raise NotFittedError("model is not fitted yet ! Make sure to call .fit in order to train it on available training datasets")

    def predict_proba(self, X):
        """
        Predict for a set of features how much probable the given player would have a 5+ years of career at the nba

        :param X: features of the player in questions
        
        :returns: float
        """
        if self.is_fitted:
            return self.model.predict_proba(X)
        raise NotFittedError("model is not fitted yet ! Make sure to call .fit in order to train it on available training datasets")

