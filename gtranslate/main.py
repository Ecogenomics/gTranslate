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
import argparse
import logging
import ntpath
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta
from typing import Dict, Tuple

from tqdm import tqdm

from gtranslate.biolib_lite.common import remove_extension, check_dir_exists, check_file_exists, make_sure_path_exists
from gtranslate.exceptions import GTranslateExit
from gtranslate.files.batchfile import Batchfile


class OptionsParser(object):

    def __init__(self, version,output_dir=None):
        """Initialization.

        Parameters
        ----------
        version : str
            The current version number (e.g. 0.2.2).
        """
        self.logger = logging.getLogger('timestamp')
        self.warnings = logging.getLogger('warnings')
        self.version = version
        self._check_package_compatibility()

        self.genomes_to_process = None

        #Setup the Stage Logger File
        if output_dir is not None:
            base_name = ntpath.basename(sys.argv[0])
            if base_name == '__main__.py':
                prog_name = __name__.split('.')[0]
            else:
                prog_name = base_name
            #timestamp_logger.info(f'{prog_name} {" ".join(sys.argv[1:])}')

    @staticmethod
    def _verify_file_path(file_path: str) -> bool:
        if ' ' in file_path:
            raise GTranslateExit(f'The genome path contains a space, this is '
                             f'unsupported by downstream applications: {file_path}')
        return True

    def _genomes_to_process(self, genome_dir, batchfile, extension):
        """Get genomes to process.

        Parameters
        ----------
        genome_dir : str
            Directory containing genomes.
        batchfile : str
            File describing genomes.
        extension : str
            Extension of files to process.

        Returns
        -------
        genomic_files : d[genome_id] -> FASTA file
            Map of genomes to their genomic FASTA files.
        """

        genomic_files, tln_tables = dict(), dict()
        if genome_dir:
            for f in os.listdir(genome_dir):
                if f.endswith(extension):
                    genome_id = remove_extension(f, extension)
                    genomic_files[genome_id] = os.path.join(genome_dir, f)

        elif batchfile:
            batchfile_fh = Batchfile(batchfile)
            genomic_files, tln_tables = batchfile_fh.genome_path, batchfile_fh.genome_tln

        # Check that all of the genome IDs are valid.
        for genome_key in genomic_files:
            self._verify_genome_id(genome_key)

        # Check that there are no illegal characters in the file path
        for file_path in genomic_files.values():
            self._verify_file_path(file_path)

        # Check that the prefix is valid and the path exists
        invalid_paths = list()
        for genome_key, genome_path in genomic_files.items():

            if not os.path.isfile(genome_path):
                invalid_paths.append((genome_key, genome_path))

        # Report on any invalid paths
        if len(invalid_paths) > 0:
            self.warnings.info(f'Reading from batchfile: {batchfile}')
            self.warnings.error(f'The following {len(invalid_paths)} genomes '
                                f'have invalid paths specified in the batchfile:')
            for g_path, g_gid in invalid_paths:
                self.warnings.info(f'{g_gid}\t{g_path}')
            raise GTranslateExit(f'There are {len(invalid_paths)} paths in the '
                             f'batchfile which do not exist, see gtranslate.warnings.log')

        if len(genomic_files) == 0:
            if genome_dir:
                self.logger.error('No genomes found in directory: %s. Check '
                                  'the --extension flag used to identify '
                                  'genomes.' % genome_dir)
            else:
                self.logger.error('No genomes found in batch file: %s. Please '
                                  'check the format of this file.' % batchfile)
            raise GTranslateExit

        return genomic_files, tln_tables

    def detect_table(self, options ):
        """Detect the genetic translation table (GTT) used in prokaryotic organisms.

        Parameters
        ----------
        options : argparse.Namespace
            The CLI arguments input by the user.
        """
        self.logger.info('Running detect_table')

        if options.genome_dir:
            check_dir_exists(options.genome_dir)

        if options.batchfile:
            check_file_exists(options.batchfile)

        make_sure_path_exists(options.out_dir)

        genomes, tln_tables = self._genomes_to_process(options.genome_dir,
                                                       options.batchfile,
                                                       options.extension)

        self.logger.info('Done.')





    def parse_options(self, options):
        """Parse user options and call the correct pipeline(s)

        Parameters
        ----------
        options : argparse.Namespace
            The CLI arguments input by the user.
        """

        # Stop processing if python 2 is being used.
        if sys.version_info.major < 3:
            raise GTranslateExit('Python 2 is no longer supported.')

        # Correct user paths
        if hasattr(options, 'out_dir') and options.out_dir:
            options.out_dir = os.path.expanduser(options.out_dir)

        # Assert that the number of CPUs is a positive integer.
        if hasattr(options, 'cpus') and options.cpus < 1:
            self.logger.warning(
                'You cannot use less than 1 CPU, defaulting to 1.')
            options.cpus = 1

        if options.subparser_name == 'detect_table':
            self.detect_table(options)
        else:
            self.logger.error('Unknown GTranslate command: "' +
                              options.subparser_name + '"\n')
            sys.exit(1)

        return 0
