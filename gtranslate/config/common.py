import json
import os
import sys
from functools import lru_cache

class __GTranslateCommonConfig:

    # Internal settings used for logging.
    LOG_TASK = 21

# Export the class for import by other modules
@lru_cache(maxsize=1)
def __get_config():
    return __GTranslate()

CONFIG = __get_config()
