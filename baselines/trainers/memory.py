class Memory(object):
    def __init__(self):
        self.data_keys = ["states", "actions", "next_states", "priorities"]

    def reset(self):
        """Method to fully reset the memory storage and related variables"""
        raise NotImplementedError("Memory is an abstract class")

    def update(self, state, action, reward, next_state, done):
        """
        Implement memory update given the full info from the latest timestep. NOTE: guard for np.nan reward and done when individual env resets.
        Return True if memory is ready to be sampled for training, False  otherwise
        """
        raise NotImplementedError("Memory is an abstract class")

    def sample(self):
        """Implement memory sampling mechanism"""
        raise NotImplementedError("Memory is an abstract class")
