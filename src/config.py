# The label that uniquely identifies the top event in datasets, and in extension, graphs
import os

TOP_EVENT_LABEL = "TE"

# Directory where input files (trainingsets) are stored
INPUT_DIR = 'input'
# Directory where mutual information files (mutual information) are stored
MUTUAL_INFORMATION_DIR = os.path.join('input', 'mutual_information')

# Directory where output (graphs) are stored.
OUTPUT_DIR = 'output'

# Alpha, the learning parameter used in algorithm_3.learn_bowtie.learn_quantative_fault_tree
ALPHA = 1

# Do not calculate, but retrieve mutual information from a file on disk.
GET_MUTUAL_INFORMATION_FROM_FILE = False
