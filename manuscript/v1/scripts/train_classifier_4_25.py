#!/usr/bin/env python

###############################################################################
#                                                                             #
#    This program is free software: you can redistribute it and/or modify     #
#    it under the terms of the GNU General Public License as published by     #
#    the Free Software Foundation, either version 3 of the License, or        #
#    (at your option) any later version.                                      #
#                                                                             #
#    This program is distributed in the hope that it will be useful,          #
#    but WITHOUT ANY WARRANTY; without even the implied warranty of           #
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the            #
#    GNU General Public License for more details.                             #
#                                                                             #
#    You should have received a copy of the GNU General Public License        #
#    along with this program. If not, see <http://www.gnu.org/licenses/>.     #
#                                                                             #
###############################################################################


__prog_name__ = 'train_classifier_4_25.py'
__prog_desc__ = ('This script outlines the steps to generate a classifier model and scaler using the feature table. '
                 'It includes data preprocessing, model training, and scaling to ensure optimal performance for '
                 'classification tasks.')

__author__ = 'Pierre Chaumeil'
__copyright__ = 'Copyright 2025'
__credits__ = ['Pierre Chaumeil']
__license__ = 'GPL3'
__version__ = '0.0.1'
__maintainer__ = 'Pierre Chaumeil'
__email__ = 'uqpchaum@uq.edu.au'
__status__ = 'Development'

import argparse
import json
import os
import sys
from itertools import product
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import compute_class_weight

from mlxtend.feature_selection import SequentialFeatureSelector

from script_logger import CustomLogger

#prevents warnings from being displayed.
import warnings
warnings.simplefilter("ignore")

class ScalerClassifier(object):
    """
    A class to train a classifier on genomic feature data, including feature scaling,
    optional feature selection, and model persistence.

    Attributes:
        training_data (pd.DataFrame): The input training dataset.
        output_dir (str): Output directory to store models and logs.
        seed (int): Random seed for reproducibility.
        thread_count (int): Number of threads to use for parallel tasks.
        logger (Logger): Custom logger for tracking events.
    """
    def __init__(self, train_data_path,output_dir,threads_count,seed=None):

        if not os.path.isfile(train_data_path):
            raise FileNotFoundError(f"Training data file not found: {train_data_path}")

        # Initialize the logger
        logger_instance = CustomLogger(output_dir,__prog_name__)
        self.logger = logger_instance.get_logger()

        self.seed = seed
        self.thread_count = threads_count
        self.training_data = pd.read_csv(train_data_path, sep='\t')
        self.training_data = self.check_percentage(self.training_data)
        self.output_dir = output_dir
        # make sure the output directory exists
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # non Canonical tetranucleotides
        list_kmers=["".join(kmer) for kmer in product('ACGT', repeat=4)]
        # add the suffix _4 to the tetranucleotides
        self.tetra_list = [f'{kmer}_4' for kmer in list_kmers]

        #we do the same for the amino acid
        list_aa = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
        self.aa_list_4 = [f'{aa}_4' for aa in list_aa]
        self.aa_list_25 = [f'{aa}_25' for aa in ['G','W']]

    def run(self, sfs=False,split_data=False):
        """
                Execute the full training pipeline.

                Args:
                    sfs (bool): Whether to apply Sequential Feature Selection.
                    split_data (bool): Whether to split the data into training and validation sets.
        """
        self.logger.info(f'{__prog_name__} {" ".join(sys.argv[1:])}')

        # Set the seed only when `run()` is called
        seed_to_use = self.seed if self.seed is not None else np.random.randint(0, 1000)

        # Initialize the StandardScaler
        scaler = StandardScaler()

        self.logger.info(f"We use seed: {seed_to_use}")
        # list all rows with NaN values
        rows_with_nan = self.training_data[self.training_data.isnull().any(axis=1)]
        self.logger.info(f'There is {len(rows_with_nan)} rows with NaN values in the training data')
        # replace NaN values with 0
        for tetra in self.tetra_list:
            self.training_data[tetra] = self.training_data[tetra].replace(np.nan, 0)

        # list of columns to fit are Coding_density_4, Coding_density_11 and cd11_cd4_delta ,GC, all the tetranucleotide and amino acid differences
        # TODO: we may have to change the columns to fit based on the results of the analysis
        columns_to_fit = ['Coding_density_4', 'Coding_density_11', 'cd11_cd4_delta', 'GC']
        columns_to_fit.extend([f'{tetra}' for tetra in self.tetra_list])
        columns_to_fit.extend([f'{aa}' for aa in self.aa_list_4])
        columns_to_fit.extend([f'{aa}' for aa in self.aa_list_25])

        # let's fit_transform the training data except for the genome column
        self.training_data[columns_to_fit] = scaler.fit_transform(self.training_data[columns_to_fit])

        # save the scaler to disk
        try:
            joblib.dump(scaler, os.path.join(self.output_dir, 'scaler_4_25.pkl'))
        except Exception as e:
            self.logger.error(f"Error saving scaler: {e}")
            raise

        # # show the mean and standard deviation of the training data Coding_density_4,
        self.logger.info(f"Mean of Coding_density_4: {self.training_data['Coding_density_4'].mean()}")
        self.logger.info(f"Standard deviation of Coding_density_4: {self.training_data['Coding_density_4'].std()}")

        train, labels, genome_list_full = self.datasplit(self.training_data)
        train.head(1)

        # Assuming y contains your target labels
        class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)
        weights_dict = dict(enumerate(class_weights))
        self.logger.info(f"Class weights: {json.dumps(weights_dict)}")

        #Parameters for the classifier
        CLASS_WEIGHT = 'balanced'
        NESTIMATORS = 250
        MIN_SAMPLES_LEAF = 2

        # TODO: we may have to change the classifier/scaler based on the results of the analysis
        picked_classifier = RandomForestClassifier(verbose=1,n_jobs=self.thread_count,
                                                   class_weight=CLASS_WEIGHT,n_estimators=NESTIMATORS,
                                                    min_samples_leaf=MIN_SAMPLES_LEAF,
                                                   random_state=seed_to_use)

        if sfs:
            sss = StratifiedKFold(n_splits=5, random_state=seed_to_use, shuffle=True)
            sf_selector = SequentialFeatureSelector(picked_classifier, forward=True, n_jobs=self.thread_count, verbose=1, cv=sss,
                                      scoring='balanced_accuracy', k_features=(1, 15))
            sf_selector.fit(train, labels)
            selected_features = list(sf_selector.k_feature_idx_)
            # Print the selected features
            list_columns = list(train.columns)
            selected_features_names = [list_columns[i] for i in selected_features]
            self.logger.info(f"Selected features: {selected_features_names}")
            train = train[selected_features_names]

        if split_data:
            # we split the data into training and validation sets
            self.logger.info("Splitting the data into training and validation sets...")
            x_train, x_val, y_train, y_val = train_test_split(train, labels, test_size=0.2, stratify=labels, random_state=seed_to_use)
        else:
            self.logger.info("Using the entire dataset for training...")
            # if we don't split the data, we use the whole dataset
            x_train = train
            y_train = labels
            x_val = train
            y_val = labels

        picked_classifier.fit(x_train, y_train)

        try:
            joblib.dump(picked_classifier, os.path.join(self.output_dir, 'classifier_4_25.pkl'))
        except Exception as e:
            self.logger.error(f"Error saving classifier: {e}")
            raise


        # we predict the test data
        predictions = picked_classifier.predict(x_val)

        # we get the balanced accuracy
        balanced_accuracy = balanced_accuracy_score(y_val, predictions)
        self.logger.info(f"Balanced accuracy: {balanced_accuracy}")


    @staticmethod
    def check_percentage(training_data):
        """
                    Ensure coding density and GC content columns are percentages (0–100 range).
                    Multiplies values by 100 if they appear to be in 0–1 range.

                    Args:
                        training_data (pd.DataFrame): The dataset to check and correct.

                    Returns:
                        pd.DataFrame: Corrected dataset with appropriate value ranges.
                    """
        # we want to make sure columns Coding_density_4, Coding_density_11 and cd11_cd4_delta are between 0 and 100
        # and not between 0 and 1

        # we check if the columns are between 0 and 1 and if so we multiply by 100
        for col_to_check in ['Coding_density_4', 'Coding_density_11', 'cd11_cd4_delta', 'GC']:
            if training_data[col_to_check].max() <= 1:
                training_data[col_to_check] = training_data[col_to_check] * 100
        return training_data

    @staticmethod
    def datasplit(train):
        """
        Split the training data into features and labels, and extract genome list.
        Args:
            train (pd.DataFrame): The input training dataset.
        Returns:
            tuple: A tuple containing the features DataFrame, labels Series, and genome list.

        """
        labels = train['tt_label']
        genome_list = train.Genome
        train = train.drop(['Genome', 'tt_label', 'TT', 'Ground_truth'], axis=1)
        return train, labels, genome_list

if __name__ == "__main__":
    print(__prog_name__ + ' v' + __version__ + ': ' + __prog_desc__)
    print('  by ' + __author__ + ' (' + __email__ + ')' + '\n')

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i','--training_data_path', dest="tdp",
                        required=True, help='Path to the training data.')
    parser.add_argument('-o','--output_dir', dest="od",
                        required=True, help='Path to the output directory.')
    # the seed has a random default value
    parser.add_argument('-s','--seed', dest="seed",
                        default=np.random.randint(0, 1000), help='Seed for reproducibility.')
    parser.add_argument('-t','--threads', dest="threads", type=int,
                        default=1, help='Number of threads to use.')
    # add a flag to enable the SequentialFeatureSelector
    parser.add_argument('-sfs', '--sequential_feature_selector', dest="sfs",
                        action='store_true', help='Enable the SequentialFeatureSelector.')
    # add a flag to split or not the data
    parser.add_argument('--split_data', dest="split_data", action='store_true',
                        help='Enable data splitting into training and validation sets.')


    args = parser.parse_args()

    try:
        classifier = ScalerClassifier(args.tdp, args.od, args.threads,args.seed)
        classifier.run(args.sfs,args.split_data)
    except Exception as e:
        print(f"ERROR: {str(e)}", file=sys.stderr)
        sys.exit(1)


