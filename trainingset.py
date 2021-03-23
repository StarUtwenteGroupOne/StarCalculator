
class TrainingSet:

    # The distinct events that are available in this class
    event_names = [1,2]

    # The observations. A 2d table, where the lists inside are ordered by the
    # event_names above.
    observations = []

    def __init__(self, training_set=None, event_tree=None, fault_tree=None):
        if training_set:
            assert 'event_names' in training_set
            assert 'observations' in training_set
            print(training_set['event_names'])
            self.event_names = training_set['event_names']
            self.observations = training_set['observations']

            observations = training_set
        elif event_tree:
            pass
        elif event_tree:
            pass
        else:
            raise AttributeError("Fill in at least one of the arguments")

    def get_observations_by_event_name(self, event_name):
        if event_name not in self.event_names:
            print(event_name)
            raise AttributeError("Event not in training set!")
        else:
            return list(zip(*self.observations))[self.event_names.index(event_name)]


    def compute_single_probability(self, event, event_state):
        print("compute_single_probability")
        return 1


    def compute_combined_probability(self, event1, event1_state, event2, event2_state):
        print("compute_combined_probability")
        return 1

    def get_events_size(self):
        return len(self.event_names)