import csv
import os

from config import TRAININGSET_DIR
from trainingset import TrainingSet


def read_trainingset(filename):
    ft_trainingset = None
    with open(os.path.join(TRAININGSET_DIR, filename), 'r') as f:
        csv_file = csv.reader(f)
        ft_trainingset = TrainingSet()

        had_first_line = False
        for line in csv_file:
            # The first line are the column names
            if not had_first_line:
                ft_trainingset.event_names = [e for e in line]
                had_first_line = True
            else:
                ft_trainingset.observations.append([e == "T" for e in line])
    return ft_trainingset