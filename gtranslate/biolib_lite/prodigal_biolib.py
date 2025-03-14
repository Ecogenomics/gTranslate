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

__author__ = 'Donovan Parks'
__copyright__ = 'Copyright 2014'
__credits__ = ['Donovan Parks']
__license__ = 'GPL3'
__maintainer__ = 'Donovan Parks'
__email__ = 'donovan.parks@gmail.com'

import itertools
import logging
import ntpath
import os
import shutil
import subprocess
import tempfile
from collections import defaultdict, namedtuple
from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd

from . import seq_tk
from .common import remove_extension, make_sure_path_exists, check_file_exists
from .execute import check_on_path
from .parallel import Parallel
from .seq_io import read_fasta, write_fasta
from .seq_tk import gc, N50
from ..classifiers.table_classifiers import Classifier_4_11, Classifier_25
from gtranslate.config.common import CONFIG
from ..external.jellyfish import Jellyfish

@dataclass
class ConsumerData:
    aa_gene_file: str
    nt_gene_file: str
    gff_file: str
    is_empty: bool
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        for key, value in self.metadata.items():
            setattr(self, key, value) # Dynamically add metadata keys as attributes
        if 'probability_4_25' not in self.metadata:
            self.probability_4_25 = 'N/A'




class Prodigal(object):
    """Wrapper for running Prodigal in parallel."""

    def __init__(self, cpus, verbose=True,cl11=None,
            scale11=None,
            cl25=None,
            scale25=None,):
        """Initialization.

        Parameters
        ----------
        cpus : int
            Number of cpus to use.
        verbose : boolean
            Flag indicating if progress should be reported.
        """

        self.logger = logging.getLogger('timestamp')

        check_on_path('prodigal')

        self.cpus = cpus
        self.verbose = verbose
        self.cl11 = cl11
        self.scale11 = scale11
        self.cl25 = cl25
        self.scale25 = scale25

    def _producer(self, genome_file):
        """Apply prodigal to genome with most suitable translation table.

        Parameters
        ----------
        genome_file : str
            Fasta file for genome.
        """

        genome_id = remove_extension(genome_file)

        aa_gene_file = os.path.join(self.output_dir, genome_id + '_genes.faa')
        nt_gene_file = os.path.join(self.output_dir, genome_id + '_genes.fna')
        gff_file = os.path.join(self.output_dir, genome_id + '.gff')

        best_translation_table = -1
        table_coding_density = {4: -1, 11: -1}
        kmer_gene_signature = defaultdict(dict)
        aa_frequency = defaultdict(dict)
        if self.called_genes:
            os.system('cp %s %s' %
                      (os.path.abspath(genome_file), aa_gene_file))
        else:
            seqs = read_fasta(genome_file)

            if len(seqs) == 0:
                self.logger.warning('Cannot call Prodigal on an empty genome. '
                                    'Skipped: {}'.format(genome_file))
                return (
                genome_id, aa_gene_file, nt_gene_file, gff_file, best_translation_table, table_coding_density[4],
                table_coding_density[11], True)

            ksignature = GenomicSignatures(4, threads=1)
            with tempfile.TemporaryDirectory('gtranslate_prodigal_tmp_') as tmp_dir:

                # determine number of bases
                total_bases = 0
                for seq in seqs.values():
                    total_bases += len(seq)

                # call genes under different translation tables
                translation_tables = [4, 11]

                for translation_table in translation_tables:
                    os.makedirs(os.path.join(tmp_dir, str(translation_table)))
                    aa_gene_file_tmp, nt_gene_file_tmp, gff_file_tmp,processed_prodigal_input = self.run_prodigal_command(translation_table,
                                                                                                 tmp_dir, genome_id,
                                                                                                 genome_file, seqs,
                                                                                                 total_bases)
                    # determine coding density
                    prodigalParser = ProdigalGeneFeatureParser(gff_file_tmp)

                    codingBases = 0
                    for seq_id, _seq in seqs.items():
                        codingBases += prodigalParser.coding_bases(seq_id)



                    codingDensity = float(codingBases) / total_bases
                    table_coding_density[translation_table] = codingDensity * 100
                    aa_frequency[translation_table] = ksignature.calculate_aa_frequency(aa_gene_file_tmp,translation_table)
                    # calculate kmer signature for the genes
                    kmer_gene_signature[translation_table] = ksignature.calculate(nt_gene_file_tmp,translation_table)
                    # for GC percentage we only count the ATCG bases and not the N

                genome_metadata_dict = {}
                genome_metadata_dict['gc_percent'] = ksignature.calculate_gc_content(seqs)
                genome_metadata_dict['n50'] = N50(seqs)
                genome_metadata_dict['genome_size'] = sum([len(seq) for seq in seqs.values()])
                genome_metadata_dict['contig_count'] = len(seqs)

                # we store the coding density in the genome metadata dictionary
                genome_metadata_dict['coding_density_4'] = table_coding_density[4]
                genome_metadata_dict['coding_density_11'] = table_coding_density[11]

                kmer_signature_genome = ksignature.calculate_kmer_full_genome(processed_prodigal_input)

                # create a dataframe to store the genome information with columns in the same order as the classifiers
                # get columns from the scaler
                temp_df = pd.DataFrame( columns = ['GC', 'Coding_density_4','Coding_density_11','cd11_cd4_delta'])
                temp_df['Coding_density_4'] = [table_coding_density[4]]
                temp_df['Coding_density_11'] = [table_coding_density[11]]
                temp_df['cd11_cd4_delta'] = [table_coding_density[11] - table_coding_density[4]]
                temp_df['GC'] = [genome_metadata_dict['gc_percent']]
                #we add the difference tetra
                temp_df = pd.concat([temp_df, pd.DataFrame(kmer_signature_genome, index=[0])], axis=1)
                #we add the difference aa

                classifier_4_11 = Classifier_4_11(self.cl11,self.scale11)
                best_translation_table,probability_4_11 = classifier_4_11.predict_translation_table(temp_df)
                genome_metadata_dict['probability_4_11'] = probability_4_11

                # if the best translation table is 4, we need to check if its 4 or actually 25
                if best_translation_table == 4:
                    # we generate the temp files for the classifier 25
                    translation_table = 25
                    os.makedirs(os.path.join(tmp_dir, str(translation_table)))
                    aa_gene_file_tmp, nt_gene_file_tmp, gff_file_tmp,processed_prodigal_input = self.run_prodigal_command(translation_table,
                                                                                                 tmp_dir, genome_id,
                                                                                                 genome_file, seqs,
                                                                                                 total_bases)

                    # determine aa frequency
                    aa_frequency[translation_table] = ksignature.calculate_aa_frequency(aa_gene_file_tmp,translation_table)

                    # convert aa frequency and kmer signature to a dataframe and no other information
                    merge_dictionary = {**aa_frequency[4], **kmer_gene_signature[4], **aa_frequency[25]}
                    temp_df = pd.DataFrame([merge_dictionary])
                    # print all the elements of the dataframe


                    classifier_25 = Classifier_25(self.cl25,self.scale25)
                    best_translation_table,probability_4_25 = classifier_25.predict_translation_table(temp_df)
                    genome_metadata_dict['probability_4_25'] = probability_4_25
                genome_metadata_dict['best_tln_table'] = best_translation_table

                shutil.copyfile(os.path.join(tmp_dir, str(best_translation_table),
                                             genome_id + '_genes.faa'), aa_gene_file)
                shutil.copyfile(os.path.join(tmp_dir, str(best_translation_table),
                                             genome_id + '_genes.fna'), nt_gene_file)
                shutil.copyfile(os.path.join(tmp_dir, str(best_translation_table),
                                             genome_id + '.gff'), gff_file)


        return (genome_id, aa_gene_file, nt_gene_file, gff_file,genome_metadata_dict,False)

    def _consumer(self, produced_data, consumer_data):
        """Consume results from producer processes.

         Parameters
        ----------
        produced_data : tuple
            Summary statistics for called genes for a specific genome.
        consumer_data : list
            Summary statistics of called genes for each genome.

        Returns
        -------
        consumer_data: d[genome_id] -> namedtuple(aa_gene_file,
                                                    nt_gene_file,
                                                    gff_file,
                                                    best_translation_table,
                                                    coding_density_4,
                                                    coding_density_11)
            Summary statistics of called genes for each genome.
        """

        if consumer_data is None:
            consumer_data = {}

        genome_id, aa_gene_file, nt_gene_file, gff_file, metadata_dict, is_empty = produced_data

        consumer_data[genome_id] = ConsumerData(aa_gene_file, nt_gene_file, gff_file, is_empty,metadata_dict)
        return consumer_data

    def _progress(self, processed_items, total_items):
        """Report progress of consumer processes.

        Parameters
        ----------
        processed_items : int
            Number of genomes processed.
        total_items : int
            Total number of genomes to process.

        Returns
        -------
        str
            String indicating progress of data processing.
        """

        return self.progress_str % (processed_items, total_items, float(processed_items) * 100 / total_items)

    def run(self,
            genome_files,
            output_dir,
            called_genes=False,
            translation_table=None,
            meta=False,
            closed_ends=False):
        """Call genes with Prodigal.

        Call genes with prodigal and store the results in the
        specified output directory. For convenience, the
        called_gene flag can be used to indicate genes have
        previously been called and simply need to be copied
        to the specified output directory.

        Parameters
        ----------
        genome_files : list of str
            Nucleotide fasta files to call genes on.
        called_genes : boolean
            Flag indicating if genes are already called.
        translation_table : int
            Specifies desired translation table, use None to automatically
            select between tables 4 and 11.
        meta : boolean
            Flag indicating if prodigal should call genes with the metagenomics procedure.
        closed_ends : boolean
            If True, do not allow genes to run off edges (throws -c flag).
        output_dir : str
            Directory to store called genes.

        Returns
        -------
        d[genome_id] -> namedtuple(best_translation_table
                                            coding_density_4
                                            coding_density_11)
            Summary statistics of called genes for each genome.
        """

        self.called_genes = called_genes
        self.translation_table = translation_table
        self.meta = meta
        self.closed_ends = closed_ends
        self.output_dir = output_dir

        make_sure_path_exists(self.output_dir)

        progress_func = None
        if self.verbose:
            file_type = 'genomes'
            self.progress_str = '  Finished processing %d of %d (%.2f%%) genomes.'
            if meta:
                file_type = 'scaffolds'
                if len(genome_files):
                    file_type = ntpath.basename(genome_files[0])

                self.progress_str = '  Finished processing %d of %d (%.2f%%) files.'

            self.logger.info('Identifying genes within %s: ' % file_type)
            progress_func = self._progress

        parallel = Parallel(self.cpus)
        summary_stats = parallel.run(
            self._producer, self._consumer, genome_files, progress_func)

        # An error was encountered during Prodigal processing, clean up.
        if not summary_stats:
            shutil.rmtree(self.output_dir)

        return summary_stats

    def run_prodigal_command(self, translation_table, tmp_dir, genome_id, genome_file, seqs,total_bases):
        aa_gene_file_tmp = os.path.join(tmp_dir, str(
            translation_table), genome_id + '_genes.faa')
        nt_gene_file_tmp = os.path.join(tmp_dir, str(
            translation_table), genome_id + '_genes.fna')
        gff_file_tmp = os.path.join(tmp_dir, str(
            translation_table), genome_id + '.gff')

        # check if there is sufficient bases to calculate prodigal
        # parameters
        if total_bases < 100000 or self.meta:
            proc_str = 'meta'  # use best pre-calculated parameters
        else:
            proc_str = 'single'  # estimate parameters from data

        # If this is a gzipped genome, re-write the uncompressed genome
        # file to disk
        prodigal_input = genome_file
        if genome_file.endswith('.gz'):
            prodigal_input = os.path.join(
                tmp_dir, os.path.basename(genome_file[0:-3]) + '.fna')
            write_fasta(seqs, prodigal_input)

        # there may be ^M character in the input file,
        # the following code is similar to dos2unix command to remove
        # those special characters.
        with open(prodigal_input, 'r') as fh:
            text = fh.read().replace('\r\n', '\n')
        processed_prodigal_input = os.path.join(
            tmp_dir, os.path.basename(prodigal_input))
        with open(processed_prodigal_input, 'w') as fh:
            fh.write(text)

        args = '-m'
        if self.closed_ends:
            args += ' -c'

        cmd = ['prodigal', args, '-p', proc_str, '-q',
               '-f', 'gff', '-g', str(translation_table),
               '-a', aa_gene_file_tmp, '-d', nt_gene_file_tmp,
               '-i', processed_prodigal_input, '-o', gff_file_tmp]


        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, encoding='utf-8')

        stdout, stderr = proc.communicate()
        # This extra step has been added for the issue 451 where Prodigal can return a free pointer error
        if proc.returncode != 0:
            self.logger.warning('Error running Prodigal on genome: '
                                '{}'.format(genome_file))
            self.logger.warning('Error message:')
            for line in stderr.splitlines():
                print(line)
            self.logger.warning('This genome is skipped.')
        return aa_gene_file_tmp, nt_gene_file_tmp, gff_file_tmp ,processed_prodigal_input


class ProdigalGeneFeatureParser(object):
    """Parses prodigal gene feature files (GFF) output."""

    def __init__(self, filename):
        """Initialization.

        Parameters
        ----------
        filename : str
            GFF file to parse.
        """
        check_file_exists(filename)

        self.genes = {}
        self.last_coding_base = {}

        self.__parseGFF(filename)

        self.coding_base_masks = {}
        for seq_id in self.genes:
            self.coding_base_masks[seq_id] = self.__build_coding_base_mask(
                seq_id)

    def __parseGFF(self, filename):
        """Parse genes from GFF file.

        Parameters
        ----------
        filename : str
            GFF file to parse.
        """
        bGetTranslationTable = True
        with open(filename, 'r') as fh:
            for line in fh:
                if bGetTranslationTable and line.startswith('# Model Data'):
                    data_model_info = line.split(':')[1].strip().split(';')
                    dict_data_model = {}
                    for item in data_model_info:
                        k = item.split('=')[0]
                        v = item.split('=')[1]
                        dict_data_model[k] = v

                    self.translationTable = int(
                        dict_data_model.get('transl_table'))
                    bGetTranslationTable = False

                if line[0] == '#':
                    continue

                line_split = line.split('\t')
                seq_id = line_split[0]
                if seq_id not in self.genes:
                    geneCounter = 0
                    self.genes[seq_id] = {}
                    self.last_coding_base[seq_id] = 0

                geneId = seq_id + '_' + str(geneCounter)
                geneCounter += 1
                start = int(line_split[3])
                end = int(line_split[4])

                self.genes[seq_id][geneId] = [start, end]
                self.last_coding_base[seq_id] = max(
                    self.last_coding_base[seq_id], end)

    def __build_coding_base_mask(self, seq_id):
        """Build mask indicating which bases in a sequences are coding.

        Parameters
        ----------
        seq_id : str
            Unique id of sequence.
        """

        # safe way to calculate coding bases as it accounts
        # for the potential of overlapping genes
        coding_base_mask = np.zeros(self.last_coding_base[seq_id], dtype=bool)
        for pos in self.genes[seq_id].values():
            coding_base_mask[pos[0]:pos[1] + 1] = True

        return coding_base_mask

    def coding_bases(self, seq_id, start=0, end=None):
        """Calculate number of coding bases in sequence between [start, end).

        To process the entire sequence set start to 0, and
        end to None.

        Parameters
        ----------
        seq_id : str
            Unique id of sequence.
        start : int
            Start calculation at this position in sequence.
        end : int
            End calculation just before this position in the sequence.
        """

        # check if sequence has any genes
        if seq_id not in self.genes:
            return 0

        # set end to last coding base if not specified
        if end is None:
            end = self.last_coding_base[seq_id]

        return np.sum(self.coding_base_masks[seq_id][start:end])

class GenomicSignatures(object):
    def __init__(self, K, threads =1):
        self.K = K
        self.kmerCols, self.kmerToCanonicalIndex = self.makeKmerColNames()
        self.totalThreads = threads

    def makeKmerColNames(self):
        """Work out unique kmers."""

        # determine all mers of a given length
        baseWords = ("A", "C", "G", "T")
        mers = ["A", "C", "G", "T"]
        for _ in range(1, self.K):
            workingList = []
            for mer in mers:
                for char in baseWords:
                    workingList.append(mer + char)
            mers = workingList

        # pare down kmers based on lexicographical ordering
        retList = []
        for mer in mers:
            if mer not in retList:
                retList.append(mer)

        sorted(retList)

        # create mapping from kmers to their canonical order position
        kmerToCanonicalIndex = {}
        for index, kmer in enumerate(retList):
            kmerToCanonicalIndex[kmer] = index

        return retList, kmerToCanonicalIndex

    def calculate(self, seqFile,translation_table):
        """Calculate genomic signature of each sequence."""

        # process each sequence in parallel
        list_jobs = []

        seqs = read_fasta(seqFile)
        codon_usage = self.seqSignature(seqs,translation_table)
        return codon_usage

    def calculate_aa_frequency(self, seqFile,translation_table):

        # Calculating amino acid frequency in gene calling files
        seqs = read_fasta(seqFile)
        aa_freq = self.aa_frequency(seqs,translation_table)
        return aa_freq


    def seqSignature(self, seqs,translation_table):
        sig = [0] * len(self.kmerCols)

        for seqid,seq in seqs.items():
            tmp_seq = seq.upper()

            numMers = len(tmp_seq) - self.K + 1
            for i in range(0, numMers):
                try:
                    kmerIndex = self.kmerToCanonicalIndex[tmp_seq[i:i + self.K]]
                    sig[kmerIndex] += 1  # Note: a numpy array would be slow here due to this single element increment
                except KeyError:
                    # unknown kmer (e.g., contains a N)
                    pass
        # normalize
        sig = np.array(sig, dtype=float)
        if np.sum(sig) != 0 :
            sig /= np.sum(sig)
        # convert to list
        sig = list(sig)
        # multiply by 100
        sig = [i*100 for i in sig]
        # convert sig to dictionary
        sig = dict(zip(self.kmerCols, sig))
        #we add the suffix of the translation table to the dictionary
        sig = {k+f'_{translation_table}': v for k, v in sig.items()}

        return sig

    def aa_frequency(self, seqs,tt):
        standard_amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
        aa_freq = {aa: 0 for aa in standard_amino_acids}
        for seqid, seq in seqs.items():
            for aa in seq:
                if aa in aa_freq:
                    aa_freq[aa] += 1
        # normalize
        total_aa = sum(aa_freq.values())
        if total_aa != 0 :
            aa_freq = {aa: freq/total_aa for aa, freq in aa_freq.items()}
        # multiply by 100 and round to 2 decimal places
        aa_freq = {aa+f'_{tt}': round(freq*100, 4) for aa, freq in aa_freq.items()}
        return aa_freq

    def calculate_gc_content(self,seq_dict):
        """Calculate GC of sequences.

        GC is calculated as (G+C)/(A+C+G+T), where
        each of these terms represents the number
        of nucleotides within the sequence. Ambiguous
        and degenerate bases are ignored. Uracil (U)
        is treated as a thymine (T).

        Parameters
        ----------
        seqs : dict[seq_id] -> seq
            Sequences indexed by sequence ids.

        Returns
        -------
        float
            GC content of sequences.
        """

        A = 0
        C = 0
        G = 0
        T = 0
        for seq in seq_dict.values():
            a, c, g, t = seq_tk.count_nt(seq)

            A += a
            C += c
            G += g
            T += t

        return (float(G + C)*100) / (A + C + G + T)

    def calculate_kmer_full_genome(self, processed_prodigal_input):
        jellyfish = Jellyfish()
        dict_kmers = jellyfish.run(processed_prodigal_input)

        # we convert the kmers to frequencies by dividing by the total number of kmers
        total_kmers = sum(dict_kmers.values())
        for kmer in dict_kmers:
            dict_kmers[kmer] = (100*dict_kmers[kmer]) / total_kmers

        # we make sure that all kmers from Jellyfish are in dict_kmers and we add the missing ones
        for kmer in CONFIG.JELLYFISH_4MERS:
            if kmer not in dict_kmers:
                dict_kmers[kmer] = 0
        return dict_kmers
