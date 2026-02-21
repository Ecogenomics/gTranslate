import os
from typing import Optional

from gtranslate.config.output import TRANSLATION_TABLE_SUFFIX
from gtranslate.exceptions import GTranslateExit


class TlnTableFile(object):
    """
    A class to handle the translation table summary file for a genome. This class is used for reading, writing, and managing the translation table file
    associated with a genome, which includes various properties like coding densities, GC content, N50 value, genome size, etc.

    Attributes
    ----------
    path : str
        Path to the translation table summary file.
    best_tln_table : int, optional
        Best translation table for the genome.
    coding_density_4 : float, optional
        Coding density for translation table 4.
    coding_density_11 : float, optional
        Coding density for translation table 11.
    gc_percent : float, optional
        GC content percentage of the genome.
    n50 : int, optional
        N50 value of the genome.
    genome_size : int, optional
        Size of the genome.
    contig_count : int, optional
        Count of contigs in the genome.
    """

    def __init__(self, out_dir: str, gid: str,
                 best_tln_table: Optional[int] = None,
                 coding_density_4: Optional[float] = None,
                 coding_density_11: Optional[float] = None,
                 gc_percentage: Optional[float] = None,
                 n50_value: Optional[int] = None,
                 genome_length: Optional[int] = None,
                 contig_count: Optional[int] = None):
        self.path = self.get_path(out_dir, gid)
        self._best_tln_table = best_tln_table
        self._coding_density_4 = coding_density_4
        self._coding_density_11 = coding_density_11
        self._gc_percent = gc_percentage
        self._n50 = n50_value
        self._genome_size = genome_length
        self._contig_count = contig_count

    def _validate_and_set(self, attribute, value, expected_type):
        # 1. Handle missing/empty data gracefully
        if value is None or str(value).strip().upper() in ['N/A', 'NONE', 'NAN', '']:
            setattr(self, f'_{attribute}', None)
            return

        try:
            # If expected_type is int, casting a string like '4.0' directly
            # to int crashes. Casting to float first safely strips the decimal.
            if expected_type is int:
                clean_value = int(float(value))
            else:
                clean_value = expected_type(value)

            setattr(self, f'_{attribute}', clean_value)

        except ValueError:
            raise GTranslateExit(f'Invalid {attribute} value: {value} for {self.path}')

    @property
    def best_tln_table(self):
        return self._best_tln_table

    @best_tln_table.setter
    def best_tln_table(self, v):
        self._validate_and_set('best_tln_table', v, int)

    @property
    def coding_density_4(self):
        return self._coding_density_4

    @coding_density_4.setter
    def coding_density_4(self, v):
        self._validate_and_set('coding_density_4', v, float)

    @property
    def coding_density_11(self):
        return self._coding_density_11

    @coding_density_11.setter
    def coding_density_11(self, v):
        self._validate_and_set('coding_density_11', v, float)

    @property
    def gc_percent(self):
        return self._gc_percent

    @gc_percent.setter
    def gc_percent(self, v):
        self._validate_and_set('gc_percent', v, float)

    @property
    def n50(self):
        return self._n50

    @n50.setter
    def n50(self, v):
        self._validate_and_set('n50', v, int)

    @property
    def genome_size(self):
        return self._genome_size

    @genome_size.setter
    def genome_size(self, v):
        self._validate_and_set('genome_size', v, int)

    @property
    def contig_count(self):
        return self._contig_count

    @contig_count.setter
    def contig_count(self, v):
        self._validate_and_set('contig_count', v, int)


    @staticmethod
    def get_path(out_dir: str, gid: str):
        """
        Construct the path for the translation table summary file based on the genome ID and output directory.

        Parameters
        ----------
        out_dir : str
            Output directory where the file will be stored.
        gid : str
            Genome ID to be used in the file name.

        Returns
        -------
        str
            Path to the translation table summary file.
        """
        return os.path.join(out_dir, f'{gid}{TRANSLATION_TABLE_SUFFIX}')


    def read(self):
        try:
            with open(self.path, 'r') as fh:
                for line in fh.readlines():
                    idx, val = line.strip().split('\t')
                    if idx == 'best_translation_table':
                        self.best_tln_table = val
                    elif idx == 'coding_density_4':
                        self.coding_density_4 = val
                    elif idx == 'coding_density_11':
                        self.coding_density_11 = val
                    elif idx == 'gc_percent':
                        self.gc_percent = val
                    elif idx == 'n50':
                        self.n50 = val
                    elif idx == 'genome_size':
                        self.genome_size = val
                    elif idx == 'contig_count':
                        self.contig_count = val
        except FileNotFoundError:
            raise GTranslateExit(f'Translation table summary file not found: {self.path}')
        except ValueError as e:
            raise GTranslateExit(f'Error parsing file: {self.path} - {e}')


    def write(self):
        try:
            with open(self.path, 'w') as fh:
                fh.write(f'best_translation_table\t{self.best_tln_table}\n')
                fh.write(f'coding_density_4\t{self.coding_density_4}\n')
                fh.write(f'coding_density_11\t{self.coding_density_11}\n')
                fh.write(f'gc_percent\t{self.gc_percent}\n')
                fh.write(f'n50\t{self.n50}\n')
                fh.write(f'genome_size\t{self.genome_size}\n')
                fh.write(f'contig_count\t{self.contig_count}\n')
        except Exception as e:
            raise GTranslateExit(f'Error writing file: {self.path} - {e}')

