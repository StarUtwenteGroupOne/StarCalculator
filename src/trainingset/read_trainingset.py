import csv
import os

from src.config import INPUT_DIR
from src.trainingset.trainingset import TrainingSet


def read_trainingset(filename):
    trainingset = None
    with open(os.path.join(INPUT_DIR, filename), 'r') as f:
        csv_file = csv.reader(f)
        trainingset = TrainingSet()

        had_first_line = False
        for line in csv_file:
            # The first line are the column names
            if not had_first_line:
                trainingset.event_names = [e for e in line]
                had_first_line = True
            else:
                trainingset.observations.append([e == "T" for e in line])
        trainingset.filename = filename
    return trainingset