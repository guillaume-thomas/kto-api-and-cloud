import logging

import fire


def validate(model_path: str, x_test_path: str, y_test_path: str) -> None:
    logging.warning(f"validate {model_path}")



if __name__ == "__main__":
    fire.Fire(validate)
