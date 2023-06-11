import argparse
import configparser
import logging

import numpy as np
import pandas as pd

import db as db_tools
from train import TweetsClassificationTrainer

config = configparser.ConfigParser()
config.read("config.ini")


def main():
    logging.info("Launching functional test")
    parser = argparse.ArgumentParser("Twitter sentiment testing")
    parser.add_argument("--data", default="tests/samples.csv")
    parser.add_argument("--model", default="experiments/tfidf_logreg/model.pkl")
    parser.add_argument("--db-user")
    parser.add_argument("--db-password")
    parser.add_argument("--db-name")
    parser.add_argument("--db-host", default=db_tools.POSTGRES_HOST)
    parser.add_argument("--db-port", default=db_tools.POSTGRES_PORT)

    args = parser.parse_args()

    trainer = TweetsClassificationTrainer.from_pretrained(args.model)
    test_data = pd.read_csv(args.data, encoding=config["data"]["encoding"])
    predictions = trainer.predict(test_data)

    if not np.all(predictions == test_data.Sentiment.values):
        logging.error("Functional test: Failed")
        return
    else:
        logging.info("Functional test: Passed")

    params = dict(
        user=args.db_user,
        password=args.db_password,
        host=args.db_host,
        port=args.db_port,
        dbname=args.db_name
    )

    db = db_tools.get_db(**params)

    test_data["login"] = params["user"]
    test_data["prediction"] = predictions

    logging.info("Writing predictions to database")
    db_tools.write_predictions(db, test_data)

    logging.info("Reading predictions from database")
    predictions_from_db = db_tools.read_predictions(db)
    print(predictions_from_db[:5])


if __name__ == "__main__":
    main()
