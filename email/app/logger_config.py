import logging

LOG_FORMAT_STR = "%(asctime)s - %(levelname)-5s - %(message)s - (%(filename)s:%(lineno)s)"
formatter = logging.Formatter(LOG_FORMAT_STR)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.propagate = False

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

if not logger.hasHandlers():
    logger.addHandler(console_handler)
