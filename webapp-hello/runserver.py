
#import os
#import logging
#import daiquiri
#from repo.app import app

from flask import Flask
from flask import request, jsonify, render_template, redirect, url_for

app = Flask(__name__)

username_list = []


@app.route('/') # root directory, homepage of our website
def index():
    return render_template('form.html', username_list=username_list)


@app.route('/success/<username>')
def post_success(username):
    return render_template('response.html', username=username)


@app.route('/hello', methods=['GET', 'POST'])  # variables in URL with <...>
def hello():
    if request.method == 'POST':
        #username = request.json['username']
        #print(username)
        #username_list.append(username)
        #print(username_list)
        #return redirect(url_for('post_success', username=username))
        username = request.form["usernameInputField"]
        username_list.append(username)
        ##return render_template('response.html', username=username)
        return redirect(url_for('post_success', username))
    elif request.method == 'GET':
        return render_template('response.html',
                               username_list=username_list,
                               username=username_list[len(username_list)-1])


if __name__ == '__main__':
    print('Running \'hello world\' service on localhost:5000 ...')
    app.run(host='127.0.0.1', port=5000, debug=True)
