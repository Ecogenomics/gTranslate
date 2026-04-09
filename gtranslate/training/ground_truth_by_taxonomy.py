#! /usr/bin/env python3

# Determine the ground truth for genomes based on their taxonomic classification.
#
# Genomes are assumed to be classified according to the GTDB taxonomic framework with the
# exception of Hodgkinia cicadicola, Nasuia deltocephalinicola, and Stammera capleta which 
# have highly reduced genomes and are not included in GTDB as they lack sufficient phylogenetically 
# informative genes to be robustly placed in the reference trees used by this taxonomic resource. 
#
# This function requires a taxonomy file that indicates the taxonomic assignment of each genome. This
# can either be a 2 column TSV file with the headers "Genome ID" and "Taxonomy", or a 3 column TSV
# file with the headers "Genome ID", "GTDB taxonomy", and "NCBI taxonomy". Taxonomy strings must be
# in Greengenes-style and indicate all ranks, e.g. 
#   d__Bacteria;p__Bacillota;c__Bacilli;o__Bacillales;f__Bacillaceae;g__Bacillus;s__Bacillus subtilis

__prog_name__ = 'ground_truth_by_taxonomy.py'
__prog_desc__ = 'Determine the ground truth for genomes based on their taxonomic classification.'

__author__ = 'Donovan Parks'
__copyright__ = 'Copyright 2026'
__credits__ = ['Donovan Parks']
__license__ = 'GPL3'
__version__ = '0.1.1'
__maintainer__ = 'Donovan Parks'
__email__ = 'donovan.parks@gmail.com'
__status__ = 'Development'


import logging
import argparse
import gzip
from collections import defaultdict

from gtranslate.biolib_lite.logger import logger_setup


class GroundTruthByTaxonomy(object):
    """Determine the ground truth for genomes based on their taxonomic classification."""

    def __init__(self):
        """Initialize."""

        # Ground truth from GTDB classifications
        self.GTDB_TT25 = set(['c__JAEDAM01'])
        self.GTDB_TT4 = set(['o__Mycoplasmatales', 's__Zinderia insecticola'])

        # Eggerthellacea genera using table 4; will need to be updated to names in Parks et al., 2026
        # once these appear in GTDB
        self.GTDB_TT4.update(set(['g__CAVGFB01', 'g__JAUNQF01']))

        # Minisyncoccia family identified in gTranslate manuscript that uses table 4. The majority, but not
        # all genomes in g__GCA-2747955 were also identified as using table 4. Currently, this is handled by
        # explicitly indicating the species in this genus identified as using table 4.
        self.GTDB_TT4.update(set(['f__JAKLIH01', 's__GCA-2747955 sp027024305', 's__GCA-2747955 sp027039745', 's__GCA-2747955 sp947311625']))

        # Must include the Fastidiosibacteraceae XS4 species cluster once (if) this genome appears in GTDB:
        #  - https://www.ncbi.nlm.nih.gov/nuccore/AP038919.1
        #  - https://pmc.ncbi.nlm.nih.gov/articles/PMC12213064

        # Ground truth from NCBI classifications
        self.NCBI_TT4 = set(['s__Candidatus Hodgkinia cicadicola', 's__Candidatus Nasuia deltocephalinicola', 's__Candidatus Stammera capleta'])
        self.NCBI_TT4.update(set(['s__Hodgkinia cicadicola', 's__Nasuia deltocephalinicola', 's__Stammera capleta']))

        # These species clusters have an unclear ground truth, see https://doi.org/10.1093/gbe/evad164
        self.GTDB_UNRESOLVED = set(['s__Providencia_A siddallii', 's__Providencia_A siddallii_A'])

        self.log = logging.getLogger('timestamp')

    def parse_manual_ground_truth_file(self, manual_gt_file: str) -> dict:
        """Parse manual ground truth file."""

        manual_ground_truth = {}
        open_file = gzip.open if manual_gt_file.endswith('.gz') else open 
        with open_file(manual_gt_file, 'rt') as f:
            header = f.readline().strip().split('\t')
            gid_idx = header.index("Genome ID")
            gt_idx = header.index("Translation table")

            for line in f:
                tokens = line.strip().split('\t')
                manual_ground_truth[tokens[gid_idx]] = tokens[gt_idx]

        return manual_ground_truth

    def run(self, taxonomy_file: str, manual_gt_file: str, out_file: str) -> None:
        """Determine the ground truth for genomes based on their taxonomic classification."""

        # read files with manually specific ground truth
        manual_ground_truth = {}
        if manual_gt_file:
            self.log.info('Parsing manual ground truth file:')
            manual_ground_truth = self.parse_manual_ground_truth_file(manual_gt_file)
            self.log.info(f' - identified manual ground truth for {len(manual_ground_truth):,} genomes')

        # determine ground truth for genomes based on their taxonomic classification
        self.log.info('Determining ground truth for genomes:')
        total_genomes = 0
        gt_table_count = defaultdict(int)
        num_by_manual_gt = 0

        open_file = gzip.open if taxonomy_file.endswith('.gz') else open 
        with open_file(taxonomy_file, 'rt') as f:
            header = f.readline().strip().split('\t')

            if 'Genome ID' in header:
                gid_idx = header.index("Genome ID")
            else:
                self.log.error("Taxonomy file must have a 'Genome ID' column.")

            fout = open(out_file, 'w')
            fout.write('Genome ID\tGround truth table')

            taxonomy_idx = None
            if 'Taxonomy' in header:
                taxonomy_idx = header.index("Taxonomy")
                fout.write('\tTaxonomy')

            gtdb_taxonomy_idx = None
            if 'GTDB taxonomy' in header:
                gtdb_taxonomy_idx = header.index('GTDB taxonomy')
                fout.write('\tGTDB taxonomy')

            ncbi_taxonomy_idx = None
            if 'NCBI taxonomy' in header:
                ncbi_taxonomy_idx = header.index('NCBI taxonomy')
                fout.write('\tNCBI taxonomy')

            if taxonomy_idx is None and gtdb_taxonomy_idx is None and ncbi_taxonomy_idx is None:
                self.log.error("Taxonomy file must have a 'Taxonomy' column or a 'GTDB taxonomy' and 'NCBI taxonomy' columns.")

            fout.write('\n')

            for line in f:
                tokens = line.strip().split('\t')

                total_genomes += 1

                gid = tokens[gid_idx]

                if gid in manual_ground_truth:
                    ground_truth_tt = manual_ground_truth[gid]
                    num_by_manual_gt += 1
                else:
                    # determine ground truth translation table based on GTDB
                    # or NCBI taxonomic classification of genome
                    taxa = set()
                    if taxonomy_idx:
                        taxa.update(set(tokens[taxonomy_idx].split(';')))

                    gtdb_taxa = set()
                    if gtdb_taxonomy_idx:
                        gtdb_taxa.update(set(tokens[gtdb_taxonomy_idx].split(';')))

                    ncbi_taxa = set()
                    if ncbi_taxonomy_idx:
                        ncbi_taxa.update(set(tokens[ncbi_taxonomy_idx].split(';')))

                    if taxa.intersection(self.GTDB_TT25) or gtdb_taxa.intersection(self.GTDB_TT25):
                        ground_truth_tt = '25'
                    elif taxa.intersection(self.GTDB_TT4) or gtdb_taxa.intersection(self.GTDB_TT4):
                        ground_truth_tt = '4'
                    elif taxa.intersection(self.NCBI_TT4) or ncbi_taxa.intersection(self.NCBI_TT4):
                        ground_truth_tt = '4'
                    elif taxa.intersection(self.GTDB_UNRESOLVED) or gtdb_taxa.intersection(self.GTDB_UNRESOLVED):
                        ground_truth_tt = 'UNRESOLVED'
                    else:
                        ground_truth_tt = '11'

                gt_table_count[ground_truth_tt] += 1

                # write out ground truth results
                fout.write(f'{gid}\t{ground_truth_tt}')

                if taxonomy_idx:
                    fout.write(f'\t{tokens[taxonomy_idx]}')

                if gtdb_taxonomy_idx:
                    fout.write(f'\t{tokens[gtdb_taxonomy_idx]}')

                if ncbi_taxonomy_idx:
                    fout.write(f'\t{tokens[ncbi_taxonomy_idx]}')

                fout.write('\n')

        fout.close()

        # write out number of genomes assigned to each translation table
        self.log.info(f' - determined ground truth for {total_genomes:,} genomes')
        if manual_gt_file:
            self.log.info(f' - ground truth set manually for {num_by_manual_gt:,} genomes')    

        for tran_table, genome_count in sorted(gt_table_count.items()):
            self.log.info(f'Table {tran_table}: {genome_count:,} ({100*genome_count/total_genomes:.2f}%)')


def main():
    print(__prog_name__ + ' v' + __version__ + ': ' + __prog_desc__)
    print('  by ' + __author__ + ' (' + __email__ + ')' + '\n')

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--taxonomy_file', required=True, help='File indicating taxonomic classification of each genome.')
    parser.add_argument('--out_file', required=True,  help='Output file to write ground truth translation table.')
    parser.add_argument('--manual_gt_file', help='File indicating manually specific ground truth for select genomes.')

    args = parser.parse_args()

    # setup logger
    logger_setup(None, "ground_truth_by_taxonomy.log", "gTranslate", __version__, False, False)

    # run program
    p = GroundTruthByTaxonomy()
    p.run(args.taxonomy_file, args.manual_gt_file, args.out_file)


if __name__ == '__main__':
    main()
