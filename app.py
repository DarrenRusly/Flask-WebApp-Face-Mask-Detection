from flask import Flask, request, session, render_template, redirect
from werkzeug.utils import secure_filename
from static.model.model import predict, loadImgData
import cv2
import tempfile
import os

UPLOAD_FOLDER = "static/images/"
app = Flask(__name__)
app.secret_key = b"2iue2#@91^%&&(u2u/[[;dai021]'.')*(^*"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html', data=None)

@app.route('/predict', methods=['GET', 'POST'])
def predictimg():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file"
        image = request.files['file']
        if image.filename == '':
            return redirect('/index')
        else:
            filename = secure_filename(image.filename)
            imgpath = os.path.join("static/images/", filename)
            image.save(imgpath)
            image = cv2.imread(imgpath)
            image = loadImgData(image)
            pred = predict(image)[0][0]
            return render_template('index.html', data=[imgpath, pred])
    return redirect('/index')

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port = port)