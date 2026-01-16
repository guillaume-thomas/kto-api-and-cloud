import logging

import fire

FEATURES = ["Pclass", "Sex", "SibSp", "Parch"]

TARGET = "Survived"


def split_train_test(data_path: str) -> tuple[str, str, str, str]:
    logging.warning(f"split on {data_path}")


if __name__ == "__main__":
    fire.Fire(split_train_test)
