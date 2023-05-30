from flask import jsonify, Flask
from flask import request, url_for
import os
from werkzeug.utils import secure_filename
from exec_model import *
from PIL import Image
import glob

app = Flask(__name__)

conversion = {0: 'angry', 1: 'disgust', 2: 'fear',
              3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

UPLOAD_FOLDER = os.path.abspath(os.path.dirname(__file__)) + '\\static\\'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])


def get_overall_emotion(txt):
    if txt == 'happy' or txt == 'surprise':
        return 'positive'
    elif txt == 'neutral':
        return 'neutral'
    else:
        return 'negative'


def get_top_3(prec):
    ind_sorted = sorted(range(len(prec)), reverse=True, key=lambda i: prec[i])
    labs = []
    values = []
    for i in range(3):
        labs.append(conversion[ind_sorted[i]])
        values.append(str(round(100*prec[ind_sorted[i]], 2))+'%')
    return labs, values


def allowedFile(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/upload', methods=['POST', 'GET'])
# API to upload file
def fileUpload():
    if request.method == 'POST':
        file = request.files.getlist('file')
        for f in file:
            filename = secure_filename(f.filename)
            if allowedFile(filename):

                # remove all previous files - for privacy / space
                files_to_remove = glob.glob(UPLOAD_FOLDER+'*')
                for ind_file in files_to_remove:
                    os.remove(ind_file)

                temp_path = os.path.join(UPLOAD_FOLDER, filename)

                # temporaryily save content to folder for use
                p1 = f.save(os.path.join(temp_path))

                # load image as np array
                rawimg = cv2.imread(temp_path)

                # get expression and modified image from model
                result, softmax, img = compute_expression(rawimg)
                print(result)

                # convert numpy array to image
                im = Image.fromarray(img)

                # save modeified image
                im.save(os.path.join(UPLOAD_FOLDER, "new_"+filename), 'jpeg')

                mod_path = url_for('static', filename="new_"+filename)

                labs, values = get_top_3(softmax)

                # return expression determined
                return jsonify({'emotion': result,
                                'path': mod_path,
                                "overall_emotion": get_overall_emotion(result),
                                "precents": {
                                    "first": {"lab": labs[0], "val": values[0]},
                                    "second": {"lab": labs[1], "val": values[1]},
                                    "third": {"lab": labs[2], "val": values[2]}
                                }
                                }
                               )

            else:
                return jsonify({'message': 'File type not allowed'}), 400
        return jsonify({"name": filename, "status": "success"})
    else:
        return jsonify({"status": "failed"})


if __name__ == "__main__":
    app.run(debug=True)
