class Belief:
    """
    Belief superclass
    """

    def __init__(self):
        pass

    def update(self, *args):
        """update belief using observation"""
        pass

    def sample(self, *args):
        """sample from the belief space"""
        pass


class FixedBelief(Belief):
    def __init__(self, obj):
        super().__init__()
        self.obj = obj

    def sample(self, *args):
        return self.obj
