#!/usr/bin/env python

import joblib
import flask
import pandas as pd
import os
from typing import Dict, Union, Tuple

# setup
BASE_PATH = '..'
FEATURES = sorted([
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
    "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
    "pH", "sulphates", "alcohol",
])

# load model
model = joblib.load(os.path.join(BASE_PATH, 'models', 'regression_model'), 'r')

# initiate flask app
app = flask.Flask(__name__)


def validate_request(json_data: Dict[str, Dict[str, Union[list, float, int]]]) -> Tuple[bool, str]:
    """
    Check request data for validity.
    :param json_data: input json from request body
    :return: boolean, True if data is valid, False otherwise
    """
    # check data structure
    if not isinstance(json_data, dict):
        return False, f"data isn't a dictionary, it's type is {type(json_data)}"
    if 'features' not in json_data.keys():
        return False, "No 'features' key"
    if not isinstance(json_data['features'], dict):
        return False, "data['features'] isn't a dictionary"

    # check that features have the same count
    feature_num = None
    for feature in FEATURES:
        if feature not in json_data['features'].keys():
            return False, f"There is no feature {feature}"
        if isinstance(json_data['features'][feature], (int, float)):
            if feature_num is None:
                feature_num = 1
            else:
                if feature_num != 1:
                    return False, "Mismatch in feature count"
        elif isinstance(json_data['features'][feature], list):
            if feature_num is None:
                feature_num = len(json_data['features'][feature])
            else:
                if feature_num != len(json_data['features'][feature]):
                    return False, "Mismatch in feature count"
        else:
            return False, "Features must bu of type 'int, float' - for single prediction, 'list' for multiple"

    return True, "Ok"


def prepare_data(json_data: Dict[str, Dict[str, Union[list, int, float]]]) -> pd.DataFrame:
    return pd.DataFrame(
        data={
            feature: [json_data['features'][feature]]
            for feature in FEATURES
        }
    )


@app.route('/', methods=['POST'])
def make_prediction():
    # get data from request
    json_data = flask.request.get_json()
    # validate data
    is_data_valid, validator_message = validate_request(json_data)
    if not is_data_valid:
        return flask.jsonify(
            error_message=(
                f"Wrong request structure.\n\r"
                f"You should have dict('features': dict(<all features>: feature value))\n\r"
                f"Required feature list: {FEATURES}\n\r",
                f"Validation message: {validator_message}"
            ),
        )
    # create df
    df = prepare_data(json_data)
    # run model prediction
    predictions = model.predict(df)
    # return answer
    return flask.jsonify(
        predictions=list(map(float, predictions)),
    )


# run flack application
if __name__ == '__main__':
    app.run()
