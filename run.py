#!/usr/bin/env python
"""
Flask Rest API for NBA Investment problem
(Brest, France)
"""
__author__ = "OUALI Maher"

from flask import Flask, request, render_template, redirect
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import os


from api.NBATrainer import NBATrainer
import config as config
import api.constants as const

verbose = const.VERBOSE
save = config.SAVE_DB


# initialize app and database
app = Flask(__name__)
app.secret_key = config.SECRET_KEY
app.config['SQLALCHEMY_DATABASE_URI'] = config.DB_URI
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.permanent_session_lifetime = config.APP_SESSION_LIFETIME
db = SQLAlchemy(app)

# initialize model
model = NBATrainer()


# define an investment table in the database to hold old investment prediction
class Investment(db.Model):
    _id = db.Column("id", db.Integer, primary_key=True)
    name = db.Column("name", db.String(50))
    date = db.Column("date", db.DateTime, default=datetime.now())
    prediction = db.Column("prediction", db.Integer)
    probability = db.Column("probability", db.Float)

    def __init__(self, name, date, prediction, probability):
        self.name = name    
        self.prediction = prediction
        self.date = date
        self.probability = probability

@app.route("/", methods=['GET', 'POST'])
def index():
    return(redirect("/form"))

@app.route("/form", methods=['GET', 'POST'])
def form():
    return(render_template("nba_player_form.html"))

@app.route("/predict", methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        name = request.form.get('name')
        X  = []
        for col in const.FEATURES_COLUMNS:
            if 'int' in const.DTYPES[col]:
                X.append(int(request.form.get(col)))
            elif 'float' in const.DTYPES[col]:
                X.append(float(request.form.get(col)))
        prediction = model.predict(X)
        probability = model.predict_proba(X)[0]
        if verbose:
            print("for input features are {}, we predict {} with probability {}".format(X, prediction, probability))
        # store investment in database
        if save:
            try:
                investment = Investment(
                    name = str(name), 
                    date = datetime.now(),
                    prediction = int(prediction),
                    probability = float(probability)
                )
                db.session.add(investment)
                db.session.commit()
            except Exception as e:
                if verbose:
                    print("Traceback error %s"%e)
                pass
        return(render_template("new_investment.html", name=name, prediction=prediction, probability=probability))
    else:
        return(redirect("/form"))

@app.route("/old", methods=['GET'])
def list_investment():
    # show old investments predictions stored in database
    investments = Investment.query.all()
    for investment in investments:
        print(investment.prediction)
    return render_template("list_investments.html", investments=investments)



if __name__ == "__main__":

    # train model for first time
    model.train(grid_search_param=const.BEST_FIT_PARAMETERS)
    # launch app
    app.run(port=config.PORT_NUMBER, debug=config.DEBUG)


