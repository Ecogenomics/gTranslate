# gTranslate

**gTranslate** is a machine learning-based command-line tool designed to automatically detect and predict the genetic translation table (GTT) used in prokaryotic genomes. 

By analyzing specific sequence features—such as coding density differences, Trp ratios, and Gly ratios—`gTranslate` utilizes advanced classifiers (including XGBoost and LightGBM) to accurately distinguish between different genetic codes, particularly focusing on the identification of recoding events associated with Translation Tables 11, 4 (UGA=Trp), and 25 (UGA=Gly).

## Features

* **Automated Table Detection:** Rapidly predict the correct genetic translation table for single genomes or large batches.
* **Interactive Visualizations:** Generate dynamic HTML dashboards to explore the feature space used by the classifiers.

## Usage

gTranslate is operated via four primary subcommands: `detect_table`, `generate_plot`, `test`, and `check_install`.

You can view the general help menu at any time:
```bash
gtranslate -h
```

### 1. `detect_table`
The core pipeline for detecting the genetic translation table in prokaryotic organisms. You must provide input genomes either via a directory or a batch file.

**Basic Usage:**
```bash
# Process a directory of FASTA files
gtranslate detect_table --genome_dir /path/to/genomes --out_dir /path/to/output

# Process genomes defined in a batch file
gtranslate detect_table --batchfile genomes.tsv --out_dir /path/to/output
```

### 2. `generate_plot`
Generates an interactive HTML dashboard to visually explore the features (e.g., Coding Density Difference, amino acid ratios) used for GTT prediction. 

**Basic Usage:**
```bash
gtranslate generate_plot --feature_file features.tsv --output_file dashboard.html
```

### 3. `test`
Runs the built-in test suite on bundled genomic data to validate that the GTT detection pipeline is functioning correctly on your system.

**Basic Usage:**
```bash
gtranslate test --cpus 4
```
*Note: If `--out_dir` is not specified, the test will securely execute in a temporary directory.*

### 4. `check_install`
Verifies your installation and ensures that all required dependencies and reference data files are correctly configured and present.

**Basic Usage:**
```bash
gtranslate check_install
```

