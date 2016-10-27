class PerformanceMetrics():

    def __init__(self):

        # The smart cab arrived destination on time
        self.successes = 0

        # Last 10 successes
        self.successes_last_10 = 0

        # Sum of scores for cab iterations
        self.scores = 0

        # Sum of scores for cab iterations
        self.scores_last_10 = 0

        # Avg deadline
        self.perc_to_deadline = 0.0

        # Avg deadline last 10 iterations
        self.perc_to_deadline_last_10 = 0.0

        # Penalties (negative rewards)
        self.penalties = 0

        # Penalties (negative rewards)
        self.penalties_last_10 = 0

        # Total steps
        self.steps = 0

        # Total steps
        self.steps_last_10 = 0
