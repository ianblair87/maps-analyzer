import flask
from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename
import uuid
from helpers import *

app = Flask(__name__)

ALLOWED_EXTENSIONS = ['jpg'] #['jpeg', 'jpg', 'png']
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/new_session', methods=['POST'])
def new_session():
    # check if the post request has the file part
    if 'file' not in request.files:
        flash('No file part')
        return 'bad'
    file = request.files['file']
    # if user does not select file, browser also
    # submit an empty part without filename
    if file.filename == '':
        flash('No selected file')
        return 'bad'
    if file and allowed_file(file.filename):
        session_id = str(uuid.uuid4())
        # filename = secure_filename(file.filename)
        # extension = filename.split('.')[1]
        file.save(f'sessions/{session_id}/image.jpg')
        return session_id

@app.route('/get_separated_layers', methods=['GET'])
def get_separated_layers():
    data = request.json
    if 'session_id' not in data:
        return 'bad'
    session_id = data['session_id']

    res = get_separated_layers(session_id)
    return res

@app.route('/get_runnable_mask', methods=['GET'])
def get_runnable_mask():
    data = request.json
    if 'session_id' not in data:
        return 'bad'
    session_id = data['session_id']

    res = get_runnable_mask(session_id)
    return res


@app.route('/get_triangle', methods=['GET'])
def get_triangle():
    data = request.json
    if 'session_id' not in data:
        return 'bad'
    session_id = data['session_id']

    triangle_data = get_triangle(session_id)
    return triangle_data

@app.route('/get_circles', methods=['GET'])
def get_circles():
    data = request.json
    if 'session_id' not in data:
        return 'bad'
    session_id = data['session_id']

    circles_data = get_circles(session_id)
    return circles_data

@app.route('/get_configuration', methods=['GET'])
def get_configuration():
    data = request.json
    if 'session_id' not in data:
        return 'bad'
    session_id = data['session_id']

    configuration_data = get_configuration(session_id)
    return configuration_data

@app.route('/get_route', methods=['GET'])
def get_route():
    data = request.json
    if 'session_id' not in data:
        return 'bad'
    session_id = data['session_id']

    res = get_route(session_id)
    return res