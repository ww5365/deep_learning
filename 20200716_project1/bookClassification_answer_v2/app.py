'''
@Author: your name
@Date: 2020-06-28 14:02:35
@LastEditTime: 2020-07-08 18:40:32
@LastEditors: xiaoyao jiang
@Description: In User Settings Edit
@FilePath: /bookClassification/app.py
'''
from flask import Flask, request
from src.utils import config
from src.ML.models import Models
import json
import tensorflow as tf
import keras

global graph, sess

graph = tf.get_default_graph()
sess = keras.backend.get_session()


model = Models(model_path=config.root_path + '/model/ml_model/lightgbm', train_mode=False)

app = Flask(__name__)
# depth filepath


@app.route('/predict', methods=["POST"])
def gen_ans():
    result = {}
    title = request.form['title']
    desc = request.form['desc']
    with sess.as_default():
        with graph.as_default():
            label, score = model.predict(title, desc)
    result = {
        "label": label,
        "proba": str(score)
    }
    return json.dumps(result, ensure_ascii=False)


# python3 -m flask run
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)