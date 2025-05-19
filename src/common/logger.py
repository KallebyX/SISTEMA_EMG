import logging

def get_logger(name="biomove"):
    logging.basicConfig(
        level=logging.INFO,
        format="🔧 [%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S"
    )
    return logging.getLogger(name)
