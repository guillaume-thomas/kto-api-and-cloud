import logging
import fire


def train(x_train_path: str, y_train_path: str, n_estimators: int, max_depth: int, random_state: int) -> str:
    logging.warning(f"train {x_train_path} {y_train_path}")


if __name__ == "__main__":
    fire.Fire(train)
