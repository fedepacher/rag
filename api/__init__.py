import os
import logging

LOG_FORMAT_STR = "%(asctime)s - %(levelname)-5s - %(name)-20.20s - %(message)s - (%(filename)s:%(lineno)s)"
LOG_FILE = os.getenv('LOG_FILE', None)

if LOG_FILE is not None:
    logging.basicConfig(filename=LOG_FILE,
                        level=logging.DEBUG,
                        format=LOG_FORMAT_STR
                        )
else:
    logging.basicConfig(
        level=logging.DEBUG,
        format=LOG_FORMAT_STR,
        handlers=[logging.StreamHandler()]
    )
