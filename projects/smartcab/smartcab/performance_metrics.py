class PerformanceMetrics():

    def __init__(self):

        # The smart cab arrived destination on time
        self.successes = 0

        # Last 10 successes
        self.last_10_successes = 0

        # Time that smart cab took to arrive destination
        self.min_epsilon_iter = 101

        # Sum of scores for cab iterations
        self.score = 0

        # Sum of scores for cab iterations
        self.score_last_10 = 0

        # Avg deadline
        self.avg_perc_to_deadline = 0.0

        # Avg deadline last 10 iterations
        self.avg_perc_to_deadline_last_10 = 0.0
