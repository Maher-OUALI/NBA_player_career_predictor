#!/usr/bin/env python
"""
NBA Data Imputer class definition and methods
(Brest, France)
"""
__author__ = "OUALI Maher"

from .constants import *

class NBADataImputer():
    """
    A class to handle data cleaning and processing
    """

    verbose = VERBOSE
    def __init__(self, verbose_=None):
        """
        Contruct NBADataImputer attributes 
        """
        if not(verbose_) == self.verbose:
            self.verbose = verbose_
        
    def fit(self, data, dtypes=None):
        """
        Clean data and seperate it into features and target

        :param data(pd.DataFrame): a dataframe containing nba players stats and a target column about if they had +5 years of career
        :param dtypes(dict): dictionary of expected new types of data columns where key is column name and value is type ('int16', 'string', 'float64' etc...)
        """
        if self.verbose:
            print("Start cleaning data, data size before processing is %s"%len(data))
        if dtypes is None:
            dtypes = DTYPES
        self.data_types = dtypes
        self.data = data
        if 'Name' in self.data.columns:
            self.data.drop('Name', axis=1, inplace=True)
        self.data = self.data.astype(self.data_types)
        #we fill missing data with 0 given that all features are floats or integer
        #the choice of 0 is due to the fact that when players don't have stats about a certain type of play (3 points, blocks etc...) meaning that he hardly or never tries
        #so we expect it to be 0
        self.data.fillna(0, inplace=True)
        #remove outliers because it affects heavily boosting algorihtms
        self._remove_outliers(FEATURES_COLUMNS)
        if self.verbose:
            print("Finished cleaning data, data size after processing is %s"%len(self.data))
        #we could keep data as it is whithout normalization since we're using boosted decision trees with catboost
        self.X, self.y = self.seperate_data(FEATURES_COLUMNS, [TARGET_VARIABLE])
    
    def seperate_data(self, features_cols=None, target_cols=None):
        """
        Seperate data into a feature dataframe and a target dataframe

        :param features_cols(list): a list of features column names
        :param target_cols(list): a list of target column names (in this case it would be a one element list)

        :return (feature_df, target_df) (tuple)
        """
        #define correcly both feature and target column names
        if features_cols == None:
            features_cols = FEATURES_COLUMNS
        if target_cols == None:
            target_cols = [TARGET_VARIABLE]
        self.features_columns = features_cols
        self.target_columns = target_cols
        try:
            #return seperate data
            return self.data[self.features_columns], self.data[self.target_columns]
        except:
            return None, None
    
    def to_dict(self):
        try:
            return {"X":self.X, "y":self.y}
        except:
            return None

    def _remove_outliers(self, columns):
        """
        Remove outliers from data by filtering values <= Q1 - 1.5*IQR and values >= Q3 + 1.5*IQR

        :param columns(str): a string of columns to be considered by the filtering of outliers

        :return None
        """
        outliers_ids = set(self.data.index.values)
        for col_name in columns:
            q1 = self.data[col_name].quantile(0.25)
            q3 = self.data[col_name].quantile(0.75)
            iqr = q3-q1 #Interquartile range
            lower_limit  = q1-1.5*iqr
            upper_limit = q3+1.5*iqr
            #keep only indexes that respect the inequality for the column in question
            outliers_ids = outliers_ids.intersection(set(self.data[(self.data[col_name] > lower_limit) & (self.data[col_name] < upper_limit)].index.values))
        #keep data of remaining indexes
        self.data = self.data.loc[list(outliers_ids)]
        self.data.reset_index(drop=False, inplace=True)
    