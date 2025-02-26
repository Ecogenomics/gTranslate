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

import logging
import multiprocessing as mp
import os
import shutil
import subprocess
from pathlib import Path

class Jellyfish(object):
    """Perform ab initio gene prediction using Prodigal."""

    def __init__(self,
                 threads=1,
                 mer_len=4,
                 hash_size='100M',
                 canonical=True):
        """Initialize."""

        self.threads = threads
        self.mer_len = mer_len
        self.hash_size = hash_size
        self.canonical = canonical

        self.version = self._get_version()

    def _get_version(self):
        try:
            env = os.environ.copy()
            proc = subprocess.Popen(['jellyfish', '-V'], stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, env=env, encoding='utf-8')

            output, error = proc.communicate()

            return output.split(' ')[1]
        except:
            return "(version unavailable)"

    def run(self, genome_file):
        """Run Jellyfish."""

        # We run jellyfish count first
        jellyfish_count_file = self._run_jellyfish_count(genome_file)
        kmer_dict = self._run_jellyfish_dump(jellyfish_count_file)
        return kmer_dict

    def _run_jellyfish_count(self, genome_file):
        # we add the extension .jf to the genome file
        jellyfish_count_file = Path(genome_file).stem + '.jf'

        args = ''
        if self.canonical:
            args = '-C'

        cmd =  ['jellyfish', 'count', '-m', str(self.mer_len),
                '-s', self.hash_size ,'-t', str(self.threads),
                genome_file ,args, '-o', jellyfish_count_file]


        process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        process.wait()

        return jellyfish_count_file

    def _run_jellyfish_dump(self, jellyfish_count_file):

        # we dump the jellyfish file
        cmd = ['jellyfish', 'dump', jellyfish_count_file]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8')

        stdout, stderr = process.communicate()

        kmer_dict = {}
        for kmerline in stdout.splitlines():
            # if line starts with >, we store the number of kmers
            if kmerline.startswith('>'):
                num_kmers = int(kmerline[1:])
            # otherwise, we store the kmer string
            else:
                kmer = kmerline.strip()
                # we store the kmer count in the dictionary
                kmer_dict[kmer] = num_kmers
        return kmer_dict




