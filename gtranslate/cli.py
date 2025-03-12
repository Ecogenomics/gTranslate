import argparse
import tempfile
from contextlib import contextmanager

from gtranslate.biolib_lite.custom_help_formatter import ChangeTempAction, CustomHelpFormatter


@contextmanager
def subparser(parser, name, desc):
    yield parser.add_parser(name, conflict_handler='resolve', help=desc,
                            formatter_class=CustomHelpFormatter)


@contextmanager
def mutex_group(parser, required):
    group = parser.add_argument_group(f'mutually exclusive '
                                      f'{"required" if required else "optional"} '
                                      f'arguments')
    yield group.add_mutually_exclusive_group(required=required)


@contextmanager
def arg_group(parser, name):
    yield parser.add_argument_group(name)

def __genome_dir(group):
    group.add_argument(
        '--genome_dir', help="directory containing genome files in FASTA format")

def __batchfile(group):
    group.add_argument('--batchfile', help="path to file describing genomes - tab "
                                           "separated in 2 columns (FASTA "
                                           "file, genome ID)")

def __out_dir(group, required):
    group.add_argument('--out_dir', type=str, default=None, required=required,
                       help="directory to output files")

def __cpus(group):
    group.add_argument('--cpus', default=1, type=int,
                       help='number of CPUs to use')

def __force(group):
    group.add_argument('--force', action='store_true', default=False,
                       help='continue processing if an error occurs on a single genome')

def __help(group):
    group.add_argument('-h', '--help', action="help", help="show help message")

def __keep_called_genes(group):
    group.add_argument('--keep_called_genes', default=False, action='store_true',
                       help='keep genes called with the right Translation table.')

def __extension(group):
    group.add_argument('-x', '--extension', type=str, default='fna',
                       help='extension of files to process, ``gz`` = gzipped')

def __prefix(group):
    group.add_argument('--prefix', type=str, default='gtranslate',
                       help='prefix for all output files')

def __temp_dir(group):
    group.add_argument('--tmpdir', action=ChangeTempAction, default=tempfile.gettempdir(),
                       help="specify alternative directory for temporary files")


def __cl11(group):
    group.add_argument('--cl11', help='Pickle file for Classifier 4/11')

def __scale11(group):
    group.add_argument('--scale11', help='Pickle file for Scaler 4/11')

def __cl25(group):
    group.add_argument('--cl25', help='Pickle file for Classifier 4/25')

def __scale25(group):
    group.add_argument('--scale25', help='Pickle file for Scaler 4/25')


def get_main_parser():
    # Setup the main, and sub parsers.
    main_parser = argparse.ArgumentParser(
        prog='gtranslate', add_help=False, conflict_handler='resolve')
    sub_parsers = main_parser.add_subparsers(help="--", dest='subparser_name')

    # de novo workflow.
    with subparser(sub_parsers, 'detect_table', 'Detect the genetic translation table (GTT) used '
                                                'in prokaryotic organisms.') as parser:
        with mutex_group(parser, required=True) as grp:
            __genome_dir(grp)
            __batchfile(grp)
        with arg_group(parser, 'required named arguments') as grp:
            __out_dir(grp, required=True)
        with arg_group(parser, 'optional arguments') as grp:
            __extension(grp)
            __temp_dir(grp)
            __cpus(grp)
            __keep_called_genes(grp)
            __prefix(grp)
            __force(grp)
            __cl11(grp)
            __scale11(grp)
            __cl25(grp)
            __scale25(grp)

    return main_parser