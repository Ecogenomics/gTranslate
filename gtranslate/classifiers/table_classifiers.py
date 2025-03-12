import logging
import joblib
import pandas as pd

from gtranslate.config.common import CONFIG


class GenericTableClassifier:
    """A generic table classifier that uses a list of column classifiers to classify tables."""
    def __init__(self, scaler_path :str, classifier_path : str):
        self.scaler_path = scaler_path
        self.classifier_path = classifier_path
        self.logger = logging.getLogger('timestamp')


    def scale_data(self, data : pd.DataFrame):
        """Scale the data using the scaler."""
        scaler = joblib.load(self.scaler_path)
        # make sure temp_df has the same columns as the scaler
        temp_df = data.reindex(columns=scaler.feature_names_in_, fill_value=0)
        # transform the genome info
        temp_df_scaled = scaler.transform(temp_df)
        # Convert the scaled array back into a DataFrame with the original column names
        temp_df_scaled = pd.DataFrame(temp_df_scaled, columns=temp_df.columns, index=temp_df.index)
        return temp_df_scaled

    def classify_table(self, data : pd.DataFrame):
        """Classify the table using the classifier."""
        classifier = joblib.load(self.classifier_path)
        # The classifier is train on a specific subset of columns so we need to make sure the data has the same columns
        trimmed_data = data.reindex(columns=classifier.feature_names_in_, fill_value=0)
        probabilities = classifier.predict_proba(trimmed_data)
        predictions = classifier.predict(trimmed_data)
        return predictions[0], max(probabilities[0])

    def scale_and_classify(self, data : pd.DataFrame):
        """Scale and classify the data."""
        scaled_data = self.scale_data(data)
        return self.classify_table(scaled_data)

class Classifier_4_11(GenericTableClassifier):
    """A classifier that distinguishes between translation tables 4 and 11."""

    def __init__(self,classifier_path :str=None, scaler_path : str=None):
        if classifier_path is None:
            classifier_path = CONFIG.CLASSIFIER_4_11
        if scaler_path is None:
            scaler_path = CONFIG.SCALER_4_11
        super().__init__(scaler_path, classifier_path)

    def get_translation_table(self, data : pd.DataFrame):
        """Get the translation table for the list of genomes."""
        binary_value_prediction,probability=self.scale_and_classify(data)
        # convert the list to 4 or 11
        # to keep a standard the TT with the majority of values will be 1 and the other 0
        converted_tt = 4 if binary_value_prediction == 0 else 11

        return converted_tt,probability

class Classifier_25(GenericTableClassifier):
    """A classifier that distinguishes between translation table 25 and 4."""

    # we can add the path to the classifiers and scalers None by default
    def __init__(self, classifier_path :str=None, scaler_path : str=None):
        if classifier_path is None:
            classifier_path = CONFIG.CLASSIFIER_25
        if scaler_path is None:
            scaler_path = CONFIG.SCALER_25
        super().__init__(scaler_path, classifier_path)

    def get_translation_table(self, data : pd.DataFrame):
        """Get the translation table for the list of genomes."""
        binary_value_prediction,probability=self.scale_and_classify(data)
        # convert the list to 25 or 4
        # to keep a standard the TT with the majority of values will be 1 and the other 0
        converted_tt = 25 if binary_value_prediction == 0 else 4
        return converted_tt,probability


