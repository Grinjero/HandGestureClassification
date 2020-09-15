import numpy as np
import time
import torch

from online.online_utils import Queue


class ActionActivator:
    """
    Handles single time activations based on filtered/raw classifier probabilities
    """

    def __init__(self, opts, contrast_class_indices):
        """
        :param output_queue_size: filter queue size
        :param contrast_class_indices: indices of contrast classes i.e. classes that are not real gestures
        :param indices_of_non_gestures: indices of classes that are not valid gestures (No_Gesture, Doing_other_things, etc.)
        :param filter_method: median, mean or exp
        :param contrast_patience: how many times in a row can a contrast class appear before the activator becomes inactive
        :param average_gesture_duration: gesture duration; if cumulative_method is timed ->  seconds
                                                                                for step -> number of steps
        :param n_classes
        :param early_threshold
        :param late_threshold
        :param cumulative_method: timed for real time step measuring, step doesn't use real time, just counts classifier calls
        """
        self.output_queue = Queue(opts.output_queue_size)
        assert opts.filter_method in ["median", "mean", "exp"]
        self.filter_method = opts.filter_method

        if opts.cumulative_method:
            assert opts.cumulative_method in ["timed", "step"]
            self.cumulative_method = opts.cumulative_method
        else:
            self.cumulative_method = "timed"

        if self.cumulative_method == "timed":
            print("Using timed accumulation")
            self.average_gesture_duration = opts.average_gesture_duration / 4
        elif self.cumulative_method == "step":
            print("Using step accumulation")
            self.average_gesture_duration = opts.average_gesture_duration / (4 * opts.skip_frames)

        # is a gesture currently detected
        self.active = False
        self.active_index = 0
        self.previous_active = False
        self.early_predict = False
        self.finished_prediction = False
        # counts for how long has the activator been active
        self.active_time_start = 0
        self.previous_step_time = time.clock()

        # how many times has a contrast class appeared in succession
        self.passive_count = 0
        self.contrast_class_indices = contrast_class_indices
        self.contrast_patience = opts.contrast_patience

        self.n_classes = opts.n_classes
        self.cum_sum = np.zeros(opts.n_classes)
        self.previous_best_idx = -1


        self.early_threshold = opts.early_threshold
        self.late_threshold = opts.late_threshold

    def _weighting_time_func(self, current_time):
        return 1 / (1 + np.exp((self.average_gesture_duration- current_time)))

    def _weighting_step_func(self, current_step):
        return 1 / (1 + np.exp(-0.2 * (current_step - self.average_gesture_duration)))

    def _set_activator_state(self, activation_state):
        if self.active == activation_state:
            # no change around here
            return

        self.active = activation_state

        if self.active:
            print("Activator activated")
            # self.active_time_start = time.clock()
            self.active_time_start = self.previous_step_time
            self.previous_step_time = self.active_time_start
            self.previous_duration_from_action_start = 0

            if self.cumulative_method == "timed":
                self.weighted_probabilities_array = []
                self.duration_array = []
            elif self.cumulative_method == "step":
                self.action_index = 0

        else:
            print("Activator deactivated")

    def _postprocess(self, probabilities):
        top_class_index = np.argmax(probabilities)

        if top_class_index not in self.contrast_class_indices:
            self.output_queue.enqueue(probabilities)
            self.passive_count = 0

            if self.filter_method == "median":
                self.filtered_probabilities = self.output_queue.median_filtering()
            elif self.filter_method == "mean":
                self.filtered_probabilities = self.output_queue.mean_filtering()
            else:
                self.filtered_probabilities = self.output_queue.exponential_average_filtering()

            max_probability = np.max(self.filtered_probabilities)
            print(max_probability)
        else:
            self.filtered_probabilities = np.zeros(len(probabilities))
            self.output_queue.enqueue(probabilities)

            self.passive_count += 1


        return self.filtered_probabilities

    def _calculate_timed_cum_sum(self, current_step_time, probabilities):
        # cumulative sum taking into account the duration of each step
        current_duration = current_step_time - self.previous_step_time
        self.duration_array.append(current_duration)

        new_duration_from_action_start = current_step_time - self.active_time_start
        self.weighted_probability = self._weighting_time_func(new_duration_from_action_start) * probabilities
        self.weighted_probabilities_array.append(self.weighted_probability * current_duration)

        self.cum_sum = np.array(self.weighted_probabilities_array).sum(axis=0)
        self.cum_sum /= np.array(self.duration_array).sum()

    def _calculate_stepped_cum_sum(self, current_step_index, probabilities):
        print("step " + str(current_step_index))
        self.cum_sum *= current_step_index - 1
        self.weighted_probability = self._weighting_step_func(current_step_index) * probabilities
        self.cum_sum += self.weighted_probability
        self.cum_sum /= current_step_index

    def _calculate_cum_sum(self, current_step_time, probabilities):
        if self.cumulative_method == "timed":
            self._calculate_timed_cum_sum(current_step_time, probabilities)
        elif self.cumulative_method == "step":
            self._calculate_stepped_cum_sum(self.active_index, probabilities)
        else:
            raise ValueError("Shouldn't be possible")

    def __call__(self, probabilities):
        """
        :param probabilities: numpy array of class probabilities
        :return: detected class index if a class is detected, None otherwise
        """
        if isinstance(probabilities, torch.Tensor):
            if probabilities.ndim != 1:
                probabilities = torch.squeeze(probabilities)
            probabilities = probabilities.numpy()


        probabilities = self._postprocess(probabilities)

        if self.passive_count >= self.contrast_patience:
            self._set_activator_state(False)
        else:
            self._set_activator_state(True)

        current_step_time = time.clock()

        activated_class = None
        if self.active:
            self.active_index += 1
            self._calculate_cum_sum(current_step_time, probabilities)

            print("Weighted probability max {}".format(np.max(self.weighted_probability)))
            print("Cum sum max {}".format(np.max(self.cum_sum)))

            best2_idx, best1_idx = self.cum_sum.argsort()[-2:]
            if (self.cum_sum[best1_idx] - self.cum_sum[best2_idx]) > self.early_threshold:
                self.finished_prediction = True
                self.early_predict = True
        else:
            self.active_index = 0

        if not self.active and self.previous_active:
            # switched off due to appearances of contrast classes
            self.finished_prediction = True
        elif self.active and not self.previous_active:
            self.finished_prediction = False

        if self.finished_prediction:
            best_idx = self.cum_sum.argmax()

            if self.early_predict:
                if best_idx != self.previous_best_idx and self.cum_sum[best_idx] > self.late_threshold:
                    print("Early detection: class {} prob {:.5f}".format(best_idx, self.cum_sum[best_idx]))
                    activated_class = best_idx
            elif self.cum_sum[best_idx] > self.late_threshold:
                if best_idx == self.previous_best_idx:
                    print("Late detection: class {} prob {:.5f}".format(best_idx, self.cum_sum[best_idx]))
                    activated_class = best_idx
                else:
                    print("Late detection: class {} prob {:.5f}".format(best_idx, self.cum_sum[best_idx]))
                    activated_class = best_idx

                self.previous_best_idx = best_idx
                self.finished_prediction = False

            self.cum_sum = np.zeros(self.n_classes)

        if not self.active and self.previous_active:
            self.early_predict = False

        self.previous_active = self.active
        self.previous_step_time = current_step_time

        return activated_class