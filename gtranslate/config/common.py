import json
import os
import sys
from functools import lru_cache

class __GTranslateCommonConfig:

    # Internal settings used for logging.
    LOG_TASK = 21

    # Tetranucleotide used by Jellyfish
    JELLYFISH_4MERS = ['AAAA','AAAC','AAAG','AAAT','AACA','AACC','AACG','AACT','AAGA','AAGC','AAGG','AAGT','AATA','AATC',
                  'AATG','AATT','ACAA','ACAC','ACAG','ACAT','ACCA','ACCC','ACCG','ACCT','ACGA','ACGC','ACGG','ACGT',
                  'ACTA','ACTC','ACTG','AGAA','AGAC','AGAG','AGAT','AGCA','AGCC','AGCG','AGCT','AGGA','AGGC','AGGG',
                  'AGTA','AGTC','AGTG','ATAA','ATAC','ATAG','ATAT','ATCA','ATCC','ATCG','ATGA','ATGC','ATGG','ATTA',
                  'ATTC','ATTG','CAAA','CAAC','CAAG','CACA','CACC','CACG','CAGA','CAGC','CAGG','CATA','CATC','CATG',
                  'CCAA','CCAC','CCAG','CCCA','CCCC','CCCG','CCGA','CCGC','CCGG','CCTA','CCTC','CGAA','CGAC','CGAG',
                  'CGCA','CGCC','CGCG','CGGA','CGGC','CGTA','CGTC','CTAA','CTAC','CTAG','CTCA','CTCC','CTGA','CTGC',
                  'CTTA','CTTC','GAAA','GAAC','GACA','GACC','GAGA','GAGC','GATA','GATC','GCAA','GCAC','GCCA','GCCC',
                  'GCGA','GCGC','GCTA','GGAA','GGAC','GGCA','GGCC','GGGA','GGTA','GTAA','GTAC','GTCA','GTGA','GTTA',
                  'TAAA','TACA','TAGA','TATA','TCAA','TCCA','TCGA','TGAA','TGCA','TTAA']

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
        classifier_model_path = os.path.join(current_dir, '..', 'classifiers', 'classifier_models','classifier_4_11.pkl')
        # Resolve the path to its absolute form
        classifier_model_path = os.path.abspath(classifier_model_path)
        return classifier_model_path

    @property
    def SCALER_25(self):
        # Get the path to the current directory where common.py is located
        current_dir = os.path.dirname(__file__)
        # Build the path to classifiers/classifier.model.pkl
        classifier_model_path = os.path.join(current_dir, '..', 'classifiers', 'classifier_models','scaler_4_25.pkl')
        # Resolve the path to its absolute form
        classifier_model_path = os.path.abspath(classifier_model_path)
        return classifier_model_path

    @property
    def CLASSIFIER_25(self):
        # Get the path to the current directory where common.py is located
        current_dir = os.path.dirname(__file__)
        # Build the path to classifiers/classifier.model.pkl
        classifier_model_path = os.path.join(current_dir, '..', 'classifiers', 'classifier_models','classifier_4_25.pkl')
        # Resolve the path to its absolute form
        classifier_model_path = os.path.abspath(classifier_model_path)
        return classifier_model_path

# Export the class for import by other modules
@lru_cache(maxsize=1)
def __get_config():
    return __GTranslateCommonConfig()

CONFIG = __get_config()
