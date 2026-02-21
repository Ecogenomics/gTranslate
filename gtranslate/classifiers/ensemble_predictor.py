import joblib
import numpy as np
from scipy import stats

from gtranslate.config.common import CONFIG


class TTPredictor:
    def __init__(self):
        """
        Initializes the predictor by loading models directly via the centralized CONFIG.
        """
        self.ada = joblib.load(CONFIG.ADA_MULTI_CLASS)
        self.dt = joblib.load(CONFIG.DT_MULTI_CLASS)
        self.knn = joblib.load(CONFIG.KNN_MULTI_CLASS)
        self.lgbm = joblib.load(CONFIG.LGBM_MULTI_CLASS)
        self.rf = joblib.load(CONFIG.RF_MULTI_CLASS)

        self.models = [self.ada, self.dt, self.knn, self.lgbm, self.rf]

        # Load label encoder
        self.label_encoder = joblib.load(CONFIG.LABEL_ENCODER)

    def predict_translation_table(self, df):
        """
        Predicts using all 5 models, takes the majority vote, and decodes the label.
        """
        expected_features = self.ada.feature_names_in_

        try:
            df_aligned = df[expected_features]
        except KeyError as e:
            raise ValueError(f"Your DataFrame is missing expected features: {e}")

        all_predictions = np.column_stack([model.predict(df_aligned) for model in self.models])
        majority_votes, _ = stats.mode(all_predictions, axis=1, keepdims=False)
        final_tt_predictions = self.label_encoder.inverse_transform(majority_votes)
        print(f"Predicted translation tables: {final_tt_predictions}, all model predictions: {all_predictions}")

        return final_tt_predictions[0]