import json
import os
import sys
from functools import lru_cache

class __GTranslateCommonConfig:

    # Internal settings used for logging.
    LOG_TASK = 21

    @property
    def SCALER_4_11(self):
        # Get the path to the current directory where common.py is located
        current_dir = os.path.dirname(__file__)
        # Build the path to classifiers/classifier.model.pkl
        classifier_model_path = os.path.join(current_dir, '..', 'classifiers', 'classifier_models','scaler_4_11.pkl')
        # Resolve the path to its absolute form
        classifier_model_path = os.path.abspath(classifier_model_path)
        return classifier_model_path

    @property
    def CLASSIFIER_4_11(self):
        # Get the path to the current directory where common.py is located
        current_dir = os.path.dirname(__file__)
        # Build the path to classifiers/classifier.model.pkl
        classifier_model_path = os.path.join(current_dir, '..', 'classifiers', 'classifier_models','ada_4_11.pkl')
        # Resolve the path to its absolute form
        classifier_model_path = os.path.abspath(classifier_model_path)
        return classifier_model_path

# Export the class for import by other modules
@lru_cache(maxsize=1)
def __get_config():
    return __GTranslateCommonConfig()

CONFIG = __get_config()
