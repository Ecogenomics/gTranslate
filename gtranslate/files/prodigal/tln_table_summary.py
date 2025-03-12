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

import os
from typing import Dict, List, Tuple, Optional, Union

from gtranslate.biolib_lite.common import make_sure_path_exists
from gtranslate.config.output import PATH_TLN_TABLE_SUMMARY
from gtranslate.exceptions import GTranslateExit

class TranslationSummaryFileRow:
    """A row contained within the ClassifySummaryFile object."""

    # db_genome_id, info.get("best_translation_table"), info.get("coding_density_4"),
    # info.get("coding_density_11"), info.get("gc_percent"), info.get("n50"),
    # info.get("genome_size"), info.get("contig_count"), info.get("probability_4_11"),
    # info.get("probability_4_25", "NA"))

    __slots__ = ('gid', 'best_tln_table', 'coding_density_4', 'coding_density_11',
                 'gc_percent', 'n50', 'genome_size', 'contig_count', 'probability_4_11',
                 'probability_4_25')

    def __init__(self):
        """Initialise the row, default all the values to None."""
        self.gid: Optional[str] = None
        self.best_tln_table: Optional[int] = None
        self.coding_density_4: Optional[float] = None
        self.coding_density_11: Optional[float] = None
        self.gc_percent: Optional[float] = None
        self.n50: Optional[int] = None
        self.genome_size: Optional[int] = None
        self.contig_count: Optional[int] = None
        self.probability_4_11: Optional[float] = None
        self.probability_4_25: Optional[str] = None



class TranslationSummaryFile(object):
    """Records the translation table for one or more genomes."""
    __slots__ = ('path', 'rows')

    def __init__(self, out_dir: str, prefix: str):
        """Configure paths and initialise storage dictionary."""
        self.path: str = os.path.join(out_dir, PATH_TLN_TABLE_SUMMARY.format(prefix=prefix))
        self.rows: Dict[str, TranslationSummaryFileRow] = dict()

    @staticmethod
    def get_col_order(row: TranslationSummaryFileRow = None) -> Tuple[List[str], List[Union[str, float, int]]]:
        """Return the column order that will be written. If a row is provided
        then format the row in that specific order."""
        if row is None:
            row = TranslationSummaryFileRow()
        mapping = [('user_genome', row.gid),
                     ('best_tln_table', row.best_tln_table),
                     ('coding_density_4', row.coding_density_4),
                     ('coding_density_11', row.coding_density_11),
                     ('gc_percent', row.gc_percent),
                     ('n50', row.n50),
                     ('genome_size', row.genome_size),
                     ('contig_count', row.contig_count),
                     ('probability_4_11', row.probability_4_11),
                     ('probability_4_25', row.probability_4_25)]
        cols, data = list(), list()
        for col_name, col_val in mapping:
            cols.append(col_name)
            data.append(col_val)
        return cols, data


    def add_row(self, row: TranslationSummaryFileRow):
        if row.gid in self.rows:
            raise GTranslateExit(f'Attempting to add duplicate row: {row.gid}')
        self.rows[row.gid] = row

    def get_row(self, gid: str) -> TranslationSummaryFileRow:
        if gid not in self.rows:
            raise GTranslateExit(f'Attempting to get non-existent row: {gid}')
        return self.rows[gid]

    def update_row(self, row: TranslationSummaryFileRow):
        if row.gid not in self.rows:
            raise GTranslateExit(f'Attempting to update non-existent row: {row.gid}')
        self.rows[row.gid] = row

    def has_row(self) -> bool:
        if self.rows.items():
            return True
        return False

    def write(self):
        """Writes the summary file to disk. None will be replaced with N/A"""
        make_sure_path_exists(os.path.dirname(self.path))
        with open(self.path, 'w') as fh:
            fh.write('\t'.join(self.get_col_order()[0]) + '\n')
            for gid, row in sorted(self.rows.items()):
                buf = list()
                for idx,data in enumerate(self.get_col_order(row)[1]):
                    # for the red_value field, we want to round the data to 5 decimals after the comma if the value is not None
                    buf.append(self.none_value if data is None else str(data))
                fh.write('\t'.join(buf) + '\n')



    def read(self):
        """Read the translation table summary file from disk."""
        if len(self.genomes) > 0:
            raise GTranslateExit(f'Warning! Attempting to override in-memory values '
                             f'for translation table summary file: {self.path}')
        with open(self.path, 'r') as fh:
            for line in fh.readlines():
                gid, tbl = line.strip().split('\t')
                self.genomes[gid] = str(tbl)

    def read(self):
        """Read the summary file from disk."""
        if not os.path.isfile(self.path):
            raise GTranslateExit(f'Error, classify summary file not found: {self.path}')
        with open(self.path) as fh:

            # Load and verify the columns match the expected order.
            cols_exp, _ = self.get_col_order()
            cols_cur = fh.readline().strip().split('\t')
            if cols_exp != cols_cur:
                raise GTranslateExit(f'The classify summary file columns are inconsistent: {cols_cur}')

            # Process the data.
            for line in fh.readlines():
                data = line.strip().split('\t')
                row = TranslationSummaryFileRow()
                row.gid = data[0]
                row.best_tln_table = data[1]
                row.coding_density_4 = data[2]
                row.coding_density_11 = data[3]
                row.gc_percent = data[4]
                row.n50 = data[5]
                row.genome_size = data[6]
                row.contig_count = data[7]
                row.probability_4_11 = data[8]
                row.probability_4_25 = data[9]
                self.add_row(row)
