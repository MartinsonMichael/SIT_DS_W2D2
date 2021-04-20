#!/usr/bin/env python

# try:
#     import joblib
#     import flask
#     import pandas as pd
# except ImportError as e:
#     print('Make sure that your virtual environment is plugin')
# except Exception as e:
#     print('Unexpected error accrue')
#     raise e

import joblib
import flask
import pandas as pd
import os
from typing import Dict, Union


# setup
BASE_PATH = '..'
FEATURES = sorted([
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
    "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
    "pH", "sulphates", "alcohol", "quality",
])

# load model
model = joblib.load(os.path.join(BASE_PATH, 'models', 'regression_model'), 'r')

# initiate flask app
app = flask.Flask(__name__)


def validate_request(json_data: Dict[str, Dict[str, Union[list, float, int]]]) -> bool:
    """
    Check request data for validity.
    :param json_data: input json from request body
    :return: boolean, True if data is valid, False otherwise
    """
    # check data structure
    if not isinstance(json_data, dict):
        return False
    if 'feature' not in json_data.keys():
        return False
    if not isinstance(json_data['features'], dict):
        return False

    # check that features have the same count
    feature_num = None
    for feature in FEATURES:
        if feature not in json_data['features'].keys():
            return False
        if isinstance(json_data['features'][feature], (int, float)):
            if feature_num is None:
                feature_num = 1
            else:
                if feature_num != 1:
                    return False
        elif isinstance(json_data['features'][feature], list):
            if feature_num is None:
                feature_num = len(json_data['features'][feature])
            else:
                if feature_num != len(json_data['features'][feature]):
                    return False
        else:
            return False

    return True


def prepare_data(json_data: Dict[str, Dict[str, Union[list, int, float]]]) -> pd.DataFrame:
    return pd.DataFrame(
        data={
            feature: json_data['features'][feature]
            for feature in FEATURES
        }
    )


@app.route('/', methods=['POST', 'GET'])
def make_prediction():
    json_data = flask.request.get_json()
    if not validate_request(json_data):
        return flask.jsonify(
            message=(
                f"Wrong request structure.\n"
                f"You should have dict('features': dict(<all features>: feature value))\n"
                f"Required feature list: {FEATURES}\n",
            ),
            json=True,
        )
    df = prepare_data(json_data)
    predictions = model.predict(df)
    return flask.jsonify(
        predictions=predictions,
        json=True,
    )


if __name__ == '__main__':
    app.run()
