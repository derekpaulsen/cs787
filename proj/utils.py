import sys
import logging
import time

logging.basicConfig(
        stream=sys.stderr,
        format='[%(filename)s:%(lineno)s - %(funcName)s() ] %(asctime)-15s : %(message)s',
)

log_level = logging.DEBUG
logger = logging.getLogger(__name__)

class Timer:

    def __init__(self):
        self.start_time = time.time()
        self.last_interval = time.time()

    def get_interval(self):
        t = time.time()
        interval = t - self.last_interval
        self.last_interval = t
        return interval

    def get_total(self):
        return time.time() - self.start_time

    def set_start_time(self):
        self.start_time = time.time()


def get_logger(name, level=None):
    level = level if level is not None else log_level
    logger = logging.getLogger(name)
    logger.setLevel(level)

    return logger
