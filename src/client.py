#!/usr/bin/env python

import requests
import json


def main() -> None:
    response = requests.request(
        method="POST",
        url="http://127.0.0.1:5000/",
        headers={
            "Content-Type": "application/json",
        },
        data=json.dumps({
            "features": {
                "fixed acidity": 7.4,
                "volatile acidity": 0.7,
                "citric acid": 0,
                "residual sugar": 1.9,
                "chlorides": 0.076,
                "free sulfur dioxide": 11,
                "total sulfur dioxide": 34,
                "density": 0.9978,
                "pH": 3.51,
                "sulphates": 0.56,
                "alcohol": 9.4,
            }
        }),
    )
    print(
        f"Got response from Wine Prediction server: {response}\n"
        f"{response.text}"
    )


if __name__ == '__main__':
    main()
