import csv
from decimal import Decimal
from random import randint

from main import TRAININGSET_DIR


class TrainingSet:

    # The distinct events that are available in this class
    event_names = [1, 2]

    # The observations. A 2d table, where the lists inside are ordered by the
    # event_names above.
    observations = []

    def __init__(self, training_set=None):
        if training_set:
            assert 'event_names' in training_set
            assert 'observations' in training_set
            # print(training_set['event_names'])
            self.event_names = training_set['event_names']
            self.observations = training_set['observations']
        else:
            self.event_names = []
            self.observations = []

    def get_observations_by_event_name(self, event_name):
        if event_name not in self.event_names:
            print(event_name)
            raise AttributeError("Event not in training set!")
        else:
            return list(zip(*self.observations))[self.event_names.index(event_name)]

    def compute_single_probability(self, event_index, event_state):
        observations = Decimal(0)
        correspondingObservations = Decimal(0)
        for observation in self.observations:
            observations += 1
            if observation[event_index] == event_state:
                correspondingObservations += 1
        return correspondingObservations / observations

    def compute_combined_probability(self, event1_index, event1_state, event2_index, event2_state):
        observations = Decimal(0)
        correspondingObservations = Decimal(0)
        for observation in self.observations:
            observations += 1
            if observation[event1_index] == event1_state and observation[event2_index] == event2_state:
                correspondingObservations += 1
        return correspondingObservations / observations

    def get_events_size(self):
        return len(self.event_names)


def create_bogus_trainingset(et_or_ft):
    return TrainingSet(training_set={
        'event_names': [v.label for v in et_or_ft.vertices],
        'observations': [
            [randint(0, 1) for _ in et_or_ft.vertices]
            for _ in range(20)
        ]
    })


def read_trainingset(filename):
    ft_trainingset = None
    with open(TRAININGSET_DIR + filename, 'r') as f:
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


def get_paper_fault_tree():

    filenames = ['faulttree.csv', 'FT_other_order.csv', 'FT_and_or.csv']

    return read_trainingset(filenames[2])


def get_paper_event_tree():

    filenames = ['eventtree.csv', 'ET_other_order.csv']

    return read_trainingset(filenames[1])