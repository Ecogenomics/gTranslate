from collections import Counter

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
        warnings = []

        try:
            df_aligned = df[expected_features]
        except KeyError as e:
            raise ValueError(f"Your DataFrame is missing expected features: {e}")

        all_predictions = np.column_stack([model.predict(df_aligned) for model in self.models])
        decoded_preds = self.label_encoder.inverse_transform(all_predictions[0])

        vote_preds = Counter(decoded_preds)
        top_preds = vote_preds.most_common()

        best_class, max_votes = top_preds[0]

        if max_votes == 2:
            warnings.append(f"Low confidence: maximum model agreement was only {max_votes}/5.")

            recoding_votes = vote_preds.get('4', 0) + vote_preds.get('25', 0)

            if recoding_votes >= 3:
                # if 2x(TT4) vs 2x(TT25) Default to 4 because it is more biologically common, but add a warning
                # Pick the recoding table that had the most votes
                if vote_preds.get('4', 0) > vote_preds.get('25', 0):
                    best_class = '4'
                    warnings.append(f"Tie broken: Ensemble vote detects a recoding event with predictions {','.join(decoded_preds)}")
                elif vote_preds.get('4', 0) == vote_preds.get('25', 0):
                    best_class = '4'
                    warnings.append("Tie broken: 2 votes for TT4 and 2 votes for TT25. Defaulted to TT4.")
                else:
                    warnings.append(f"Tie broken: Ensemble vote detects a recoding event with predictions {','.join(decoded_preds)}")
                    best_class = '25'


        confidence_score = vote_preds.get(best_class, 0) / len(self.models)
        bonus_confidence = 0.0

        # Add 0.05 partial confidence for each vote going to the other recoding event
        if best_class == '4':
            bonus_confidence = vote_preds.get('25', 0) * 0.05
        elif best_class == '25':
            bonus_confidence = vote_preds.get('4', 0) * 0.05

        confidence_score += bonus_confidence

        return best_class, confidence_score, warnings
