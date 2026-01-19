import logging

import fire


def workflow(input_data_path: str, n_estimators: int, max_depth: int, random_state: int) -> None:
    logging.warning(f"workflow input path : {input_data_path}")


if __name__ == "__main__":
    fire.Fire(workflow)
    