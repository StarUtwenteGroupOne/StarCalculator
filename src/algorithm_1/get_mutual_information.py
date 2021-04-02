import csv
import os
from decimal import Decimal

from src.config import MUTUAL_INFORMATION_DIR, GET_MUTUAL_INFORMATION_FROM_FILE
from src.trainingset.trainingset import TrainingSet


def get_mutual_information(training_set: TrainingSet):
    events_size = training_set.get_events_size()

    if GET_MUTUAL_INFORMATION_FROM_FILE:
        weights = []
        with open(os.path.join(MUTUAL_INFORMATION_DIR, training_set.filename), 'r') as f:
            csv_file = csv.reader(f)

            had_first_line = False

            columns = 0
            rows = 0

            for line in csv_file:
                if not had_first_line:
                    if not all(map(lambda x, y: x == y, line, training_set.event_names)):
                        raise AttributeError("Columns of mutual information set do not match "
                                             "Trainingset.")
                    columns = len(line)
                    had_first_line = True
                else:
                    weights.append(line)
                    rows += 1
            if columns != rows:
                raise AttributeError("The mutual information table is not square! (columns == "
                                     "rows)")

    else:
        weights = [[0] * events_size for _ in range(events_size)]

        # calculate mutual information
        for i in range(0, events_size):
            for j in range(0, events_size):
                weights[i][j] = compute_mutual_information(training_set, i, j)
    return weights


def compute_mutual_information(training_set, event1, event2):
    weight = 0
    for event1_state in [True, False]:
        for event2_state in [True, False]:
            probability_event1 = training_set.compute_single_probability(event1, event1_state)
            probability_event2 = training_set.compute_single_probability(event2, event2_state)
            probability_event1_and_event2 = training_set.compute_combined_probability(event1,
                                                                                      event1_state,
                                                                                      event2,
                                                                                      event2_state)
            if any([i == 0 for i in [probability_event1_and_event2, probability_event1,
                                     probability_event2]]):
                weight += 0
            else:
                probability = probability_event1_and_event2 / \
                              (probability_event1 * probability_event2)
                probabilityLog = probability.log10() / Decimal(2).log10()
                weight += probability_event1_and_event2 * probabilityLog

    return weight if event1 != event2 else None