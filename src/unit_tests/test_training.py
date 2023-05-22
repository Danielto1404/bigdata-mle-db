import configparser
import os
import sys
import unittest

import pandas as pd

sys.path.insert(1, os.path.join(os.getcwd(), "src"))

from train import TweetsClassificationTrainer

config = configparser.ConfigParser()
config.read("config.ini")


class TestTraining(unittest.TestCase):
    def setUp(self) -> None:
        self.trainer = TweetsClassificationTrainer.default_trainer(
            config["data"]["train"],
            config["data"]["test"]
        )

    def test_get_train_data(self):
        self.assertTrue(type(self.trainer.get_train_data()) is pd.DataFrame)

    def test_get_test_data(self):
        self.assertTrue(type(self.trainer.get_test_data()) is pd.DataFrame)

    def test_training_with_validation(self):
        train_f1, val_f1 = self.trainer.fit(use_validation=True)
        self.assertTrue(train_f1 is not None)
        self.assertTrue(val_f1 is not None)

    def test_training_without_validation(self):
        train_f1, val_f1 = self.trainer.fit(use_validation=False)
        self.assertTrue(train_f1 is not None)
        self.assertTrue(val_f1 is None)


if __name__ == "__main__":
    unittest.main()
