from online.online_utils import Queue


class ClassifierPostprocessing:
    ""
    def __init__(self, output_queue_size, filter_method):
        """
        :param indices_of_non_gestures: indices of classes that are not valid gestures (No_Gesture, Doing_other_things, etc.)
        :param filter_method: median, mean or exp
        """
        self.output_queue = Queue(output_queue_size)

        assert filter_method in ["median", "mean", "exp"]
        self.filter_method = filter_method

    def __call__(self, probabilities):

        self.output_queue.enqueue(probabilities)

        if self.filter_method == "median":
            return self.output_queue.median_filtering()
        elif self.filter_method == "mean":
            return self.output_queue.mean_filtering()
        else:
            return self.output_queue.exponential_average_filtering()