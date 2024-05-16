from running import rw_directory
from datetime import datetime as dt
import logging

# # base logger的用处？
# base_logger = logging.getLogger('base')
# base_logger.setLevel(logging.DEBUG)
# base_format = logging.Formatter("[%(asctime)s]{%(pathname)s:%(lineno)d}%(levelname)s:%(message)s")
# base_file_name = rw_directory.log_file_path(dt.now().strftime("%Y__%m__%d__%H__%M__%S" + ".log"))
# base_handler = logging.FileHandler(filename=base_file_name, mode='a+', encoding='utf-8')
# base_handler.setFormatter(base_format)
# base_handler.setLevel(logging.INFO)


def get_logger(name=""):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    test_format = logging.Formatter("[%(asctime)s]{%(pathname)s:%(lineno)d}%(levelname)s:%(message)s")
    file_name = rw_directory.log_file_path(dt.now().strftime("%Y__%m__%d__%H__%M__%S " + name + ".log"))
    test_handler = logging.FileHandler(filename=file_name, mode='a+', encoding='utf-8')
    test_handler.setFormatter(test_format)
    test_handler.setLevel(logging.INFO)
    logger.addHandler(test_handler)
    return logger
