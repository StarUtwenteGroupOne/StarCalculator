
class TrainingSet:

    # The distinct events that are available in this class
    event_names = []

    # The observations. A 2d table, where the lists inside are ordered by the
    # event_names above.
    observations = []

    def __init__(self, training_set=None, event_tree=None, fault_tree=None):
        if training_set:
            observations = training_set
        elif event_tree:
            pass
        elif event_tree:
            pass
        else:
            raise AttributeError("Fill in at least one of the arguments")

    def get_observations_by_event_name(self, event_name):
        i = self.event_names.index(event_name)
        if not i:
            raise AttributeError("Event not in training set!")
        else:
            return list(zip(*self.observations))[i]
