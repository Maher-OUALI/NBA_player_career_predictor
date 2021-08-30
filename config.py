#!/usr/bin/env python
"""
Config file
"""
__author__ = "OUALI Maher"

import os
from datetime import timedelta

# flask app related constants
DEBUG = True
PORT_NUMBER = 5000
DB_URI = 'sqlite:///Investment.db'
SECRET_KEY = "nba"
SAVE_DB = True
APP_SESSION_LIFETIME = timedelta(minutes=10)
BASE_DIR = os.path.dirname(os.path.realpath(__file__))