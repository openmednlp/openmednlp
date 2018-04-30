import logging
import os
from datetime import datetime, timedelta

import pandas as pd
import schedule
from flask import Flask, g, jsonify, make_response, render_template, request
from flask_assets import Bundle, Environment

app = Flask(__name__, instance_relative_config=True)
app.config.from_object('openmednlp.default_config')
app.config.from_pyfile('config.cfg')
version = app.config['VERSION'] = '0.0.1'


@app.route('/')
def main():
    return render_template('index.html', version=app.config['VERSION'])