
class TrainingSet:
    # The distinct events that are available in this class
    event_names = {}

    # The observations. A 2d table, where the lists inside are ordered by the event_names.
    observations = []

    def __init__(self, observations=None, event_names=None):
        if observations:
            self.observations = observations
        else:
            raise AttributeError("Fill in the arguments")
        if event_names:
            self.event_names = event_names
        else:
            raise AttributeError("Fill in the arguments")

    def get_observations_by_event_name(self, event_name):
        i = self.event_names[event_name]
        if not i:
            raise AttributeError("Event not in training set!")
        else:
            return list(zip(*self.observations))[i]


    def compute_single_probability(self, event_index, event_state):
        print("compute_single_probability")
        return 1


    def compute_combined_probability(self, event1_index, event1_state, event2_index, event2_state):
        print("compute_combined_probability")
        return 1

    def get_events_size(self):
        return len(self.event_names)