import logging

from logging.config import dictConfig


def get_logger():
    logging_config = {
        'version': 1,
        'formatters': {
            'f': {
                'format': '%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
            }
        },
        'handlers': {
            'h': {
                'class': 'logging.StreamHandler',
                'formatter': 'f',
                'level': logging.DEBUG,
            }
        },
        'root': {
            'handlers': ['h'],
            'level': logging.DEBUG,
        },
    }

    dictConfig(logging_config)

    return logging.getLogger()
