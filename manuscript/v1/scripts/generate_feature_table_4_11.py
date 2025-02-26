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


__prog_name__ = 'generate_feature_table_4_11.py'
__prog_desc__ = ('This script generates a TSV file listing all the features used by classifiers to differentiate '
                 'between translation tables 11 and 4. It extracts relevant feature data, formats it appropriately, '
                 'and outputs a structured file for further analysis.')

__author__ = 'Pierre Chaumeil'
__copyright__ = 'Copyright 2025'
__credits__ = ['Pierre Chaumeil']
__license__ = 'GPL3'
__version__ = '0.0.1'
__maintainer__ = 'Pierre Chaumeil'
__email__ = 'uqpchaum@uq.edu.au'
__status__ = 'Development'

import argparse
import sys
import logging

import pandas as pd
import multiprocessing as mp

from tqdm import tqdm


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

class FeatureGenerator(object):

    def __init__(self,kmer_file,outfile,encoder=False,cpus=1):
        # load the information profile of the model
        self.profile_kmer = pd.read_csv(kmer_file)
        logger.info('Loaded kmer profile:')
        logger.info(self.profile_kmer.head())
        self.encoder = encoder
        self.cpus = cpus
        self.outf = outfile
        self.codetta_justification = ['F100','G100','S100','F95','G95','S95','N+C']

    def check_file_header(self,file_path, required_columns):
        with open(file_path, 'r') as file:
            header = file.readline().strip().split('\t')
            missing_columns = [col for col in required_columns if col not in header]
            if missing_columns:
                logger.error(f"Missing columns in file {file_path}: {missing_columns}")
                return False
        return True

    def run(self, ground_truth_file,taxonomy_file):
        """Run the feature generation process."""
        genome_infos = self.parse_ground_truth(ground_truth_file)
        logger.info('Parsed ground truth file')
        genome_infos = self.add_gc_content(taxonomy_file, genome_infos)
        logger.info('Added GC content')

        # we create the output file
        with open(self.outf, 'w') as outf:
            outf.write('Genome\tTT\tGround_truth\tGC\tCoding_density_4\tCoding_density_11\tcd11_cd4_delta')
            # we add the column with the profile information
            column_names = list(self.profile_kmer.columns)
            for col in column_names:
                if col != 'name':
                    outf.write(f'\t{col}')
            if self.encoder:
                outf.write('\ttt_label')
            outf.write('\n')

            genomes_to_process = [(genome, info, True) for genome, info in genome_infos.items()]

            # lets multiprocess this
            with mp.Pool(processes=40) as pool:
                lines_to_write = list(tqdm(pool.imap_unordered(self.generate_line, genomes_to_process),
                                           total=len(genomes_to_process), unit='genome', ncols=100))

            for line in lines_to_write:
                outf.write('\t'.join(map(str, line)) + '\n')

        print('Done')

    def parse_ground_truth(self, ground_truth_file):
        """Parse the ground truth file to extract relevant information."""
        genome_info = {}
        if self.check_file_header(ground_truth_file, ['Genome', 'GTDB_TT', 'Codetta_columns_used', 'Ground_truth',
                                                      'Ground_truth_justification', 'Coding_density_4', 'Coding_density_11']):
            logger.info('Ground truth file header is correct.')
        else:
            raise ValueError("Ground truth file header is incorrect.")
        with open(ground_truth_file) as codettaf:
            header = codettaf.readline().strip().split('\t')
            indices = {col: header.index(col) for col in header}
            for line in codettaf:
                save_genome = False
                line = line.strip().split('\t')
                genome = line[indices['Genome']]
                ground_truth_justification = line[indices['Ground_truth_justification']]
                ground_truth_justification_list = ground_truth_justification.split(',')

                if "AGREE_GNC" in ground_truth_justification or "AGREE_GC" in ground_truth_justification:
                    save_genome = True
                # if columns_used > 34 and any of the codetta_justification is in the ground_truth_list
                elif int(line[indices['Codetta_columns_used']]) > 34 and any(
                        [just in ground_truth_justification_list for just in self.codetta_justification]):
                    save_genome = True
                elif "S5" in ground_truth_justification_list or "N+G" in ground_truth_justification_list:
                    save_genome = True
                elif ground_truth_justification == 'C8: N+C+S100':
                    save_genome = True
                if save_genome:
                    genome_info[genome] = {
                        'tt': line[indices['GTDB_TT']],
                        'coding_density_4': line[indices['Coding_density_4']],
                        'ground_truth': line[indices['Ground_truth']],
                        'coding_density_11': line[indices['Coding_density_11']],
                        'cd11-cd4': float(line[indices['Coding_density_11']]) - float(line[indices['Coding_density_4']])
                    }
        return genome_info

    def generate_line(self,job):
        genome, info = job

        profile = self.profile_kmer[self.profile_kmer['name'] == genome].drop(columns=['name'], errors='ignore')

        tt_label = ''
        if self.encoder:
            if info['ground_truth'] == '11':
                tt_label = 1
            elif info['ground_truth'] in ['4', '25']:
                tt_label = 0

        return "\t".join(map(str, [
            genome, info['tt'], info['ground_truth'], info.get('gc', 'NA'),
            info['coding_density_4'], info['coding_density_11'], info['cd11_cd4']
        ] + profile.values.flatten().tolist())) + (f'\t{tt_label}' if self.encoder else '') + '\n'

    def add_gc_content(self,metadata_file, genome_infos):
        with open(metadata_file) as f:
            header = f.readline().strip().split('\t')
            indices = {col: header.index(col) for col in header}

            for line in f:
                values = line.strip().split('\t')
                genome_id = values[indices['accession']].replace('RS_', '').replace('GB_', '')
                if genome_id in genome_infos:
                    genome_infos[genome_id]['gc'] = values[indices['gc_percent']]
        return genome_infos


if __name__ == "__main__":
    print(__prog_name__ + ' v' + __version__ + ': ' + __prog_desc__)
    print('  by ' + __author__ + ' (' + __email__ + ')' + '\n')

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-k','--kmer_file', dest="kmerf",
                        required=True, help='Path the tsv file containing the kmer profile for the genomes.')
    parser.add_argument('-g','--ground_truth', dest="gtf",
                        required=True, help='Path to the ground truth file.')
    parser.add_argument('-m','metadata_file', dest="metaf",
                        required=True, help='Path to the taxonomy file.')
    parser.add_argument('-c','--cpus', dest="cpus",
                        default=1, help='Number of cpus to use.')
    parser.add_argument('-e','--encoder', dest="encoder", store=True,
                        help='Convert the Ground_truth to a binary format.')
    parser.add_argument('-o','--output_directory', dest="outd",
                        required=True, help='Path to generate genomes.')

    args = parser.parse_args()

    try:
        featgenerator = FeatureGenerator(args.kmer_file, args.output_file,args.encoder, args.cpus)
        featgenerator.run(args.ground_truth, args.metadata_file)
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
        sys.exit(1)







