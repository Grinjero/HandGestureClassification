import numpy as np
import time
from online.online_utils import Queue


class ActionActivator:
    """
    Handles single time activations based on filtered/raw classifier probabilities
    """

    # def __init__(self, n_classes, output_queue_size, filter_method, contrast_class_indices, contrast_patience, average_gesture_duration):
    def __init__(self, opts, contrast_class_indices):
        """
        :param output_queue_size: filter queue size
        :param contrast_class_indices: indices of contrast classes i.e. classes that are not real gestures
        :param indices_of_non_gestures: indices of classes that are not valid gestures (No_Gesture, Doing_other_things, etc.)
        :param filter_method: median, mean or exp
        :param contrast_patience: how many times in a row can a contrast class appear before the activator becomes inactive
        :param average_gesture_duration: gesture duration in seconds
        :param n_classes
        """


        self.output_queue = Queue(opts.output_queue_size)
        assert opts.filter_method in ["median", "mean", "exp"]
        self.filter_method = opts.filter_method

        # is a gesture currently detected
        self.active = False
        self.previous_active = False
        self.early_predict = False
        self.finished_prediction = False
        # counts for how long has the activator been active
        self.active_time_start = 0
        self.previous_step_time = time.process_time()

        # how many times has a contrast class appeared in succession
        self.passive_count = 0
        self.contrast_class_indices = contrast_class_indices
        self.contrast_patience = opts.contrast_patience

        self.n_classes = opts.n_classes
        self.cum_sum = np.zeros(opts.n_classes)
        self.previous_best_idx = -1

        self.average_gesture_duration = opts.average_gesture_duration

    def _weighting_func(self, x):
        return (1 / (1 + np.exp(self.average_gesture_duration - self.active_time_start)))

    def _set_activator_state(self, activation_state):
        self.active = activation_state

        if self.active:
            # self.active_time_start = time.process_time()
            self.active_time_start = self.previous_step_time
            self.previous_step_time = self.active_time_start

    def _postprocess(self, probabilities):
        top_class_index = np.argmax(probabilities)

        if top_class_index not in self.contrast_class_indices:
            self.output_queue.enqueue(probabilities)
            self.passive_count = 0

            if self.filter_method == "median":
                probabilities = self.output_queue.median_filtering()
            elif self.filter_method == "mean":
                probabilities = self.output_queue.mean_filtering()
            else:
                probabilities = self.output_queue.exponential_average_filtering()
        else:
            probabilities = np.zeros(len(probabilities))
            self.output_queue.enqueue(probabilities)

            self.passive_count += 1


        return probabilities

    def __call__(self, probabilities):
        """
        :param probabilities: numpy array of class probabilities
        :return: detected class index if a class is detected, None otherwise
        """

        probabilities = self._postprocess(probabilities)

        if self.passive_count >= self.contrast_patience:
            self._set_activator_state(False)
        else:
            self._set_activator_state(True)

        current_step_time = time.process_time()

        activated_class = None
        if self.active:
            # cumulative sum taking into account the duration of each step
            self.cum_sum *= self.previous_step_time - self.active_time_start
            # weighting function takes into account the average duration of a gesture
            self.cum_sum += self._weighting_func(current_step_time - self.active_time_start) * probabilities
            self.cum_sum /= current_step_time - self.active_time_start

            best2_idx, best1_idx = self.cum_sum.argsort()[-2:]
            if (self.cum_sum[best1_idx] - self.cum_sum[best2_idx]) > 0.15:
                self.finished_prediction = True
                self.early_predict = True

        if not self.active and self.previous_active:
            self.finished_prediction = True
        elif self.active and not self.previous_active:
            self.finished_prediction = False

        if self.finished_prediction:
            best_idx = self.cum_sum.argmax()
            if self.cum_sum[best_idx] > 0.7:
                if self.early_predict:
                    if best_idx != self.previous_best_idx:
                        print("Early detection: class {} prob {:.5f}", best_idx, self.cum_sum[best_idx])
                        activated_class = best_idx
                else:
                    if best_idx == self.previous_best_idx:
                        print("Late detection: class {} prob {:.5f}", best_idx, self.cum_sum[best_idx])
                        activated_class = best_idx
                    else:
                        print("Late detection: class {} prob {:.5f}", best_idx, self.cum_sum[best_idx])
                        activated_class = best_idx

                self.previous_best_idx = best_idx
                self.finished_prediction = False

            self.cum_sum = np.zeros(self.n_classes)

        if not self.active and self.previous_active:
            self.early_predict = False

        self.previous_active = self.active
        self.previous_step_time = current_step_time

        return activated_class