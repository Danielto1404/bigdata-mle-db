import argparse
import configparser

import numpy as np
import pandas as pd

from train import TweetsClassificationTrainer

config = configparser.ConfigParser()
config.read("config.ini")


def main():
    parser = argparse.ArgumentParser("Twitter sentiment prediction")
    parser.add_argument("--data", default="tests/samples.csv")
    parser.add_argument("--model", default="experiments/tfidf_logreg/model.pkl")
    args = parser.parse_args()

    trainer = TweetsClassificationTrainer.from_pretrained(args.model)
    test_data = pd.read_csv(args.data, encoding=config["data"]["encoding"])
    predictions = trainer.predict(test_data)

    assert np.all(predictions == test_data.Sentiment.values), "Functional test: Failed"
    print("Functional test: OK")


if __name__ == '__main__':
    main()
