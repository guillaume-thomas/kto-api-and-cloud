import logging

import fire



def load_data(path: str) -> str:
    logging.warning(f"load_data on path : {path}")


if __name__ == "__main__":
    fire.Fire(load_data)
