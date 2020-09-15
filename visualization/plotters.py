import matplotlib
from matplotlib import pyplot as plt
matplotlib.use("TkAgg")
from collections import deque
import numpy as np

class ResultPlotter:
    def __init__(self, n_classes, prediction_names=None, plot_activations=False, x_size=100, colormap="rainbow"):
        self.prediction_names = prediction_names
        self.plot_activations = plot_activations
        self.x_size = x_size
        self.color_map = plt.get_cmap(colormap)
        self.n_classes = n_classes

        # key is the name of the probability, value is a dequeu containing plot values
        self.prediction_queues = dict()

        for prediction_name in prediction_names:
            self.prediction_queues[prediction_name] = deque(maxlen=self.x_size)

        self._init_figures()
        plt.show(block=False)

    def _init_figures(self):
        self.figure = plt.figure()
        self.subplots = dict()

        num_rows = len(self.prediction_names)
        if self.plot_activations:
            num_rows += 1

        row_index = 1
        for prediction_name in self.prediction_names:
            ax = self.figure.add_subplot(num_rows, 1, row_index)

            self.subplots[prediction_name] = ax

            row_index += 1

        if self.plot_activations:
            ax = self.figure.add_subplot(num_rows, 1, row_index)
            ax.set(ylabel="activations")
            ax.set(xlabel="t")
            ax.set(xlim=(-self.x_size, 0), ylim=(0, 1))
            self.subplots["activations"] = ax

    def _get_color(self, class_id):
        """
        Get color for a class id.
        Args:
            class_id (int): class id.
        """
        rgb = self.color_map(class_id / self.n_classes)[:3]
        return rgb

    def __call__(self, prediction_arrays=None, activation_index=None):
        """
        :param prediction_arrays: dict -> key is the name of the probability, value is a tensor/array of probabilities per class
        :param activation_index: if you are plotting activations and an action has just been classified it will be displayed as an arrow
        in the color of the given class index
        """
        # self.figure.clf()
        if prediction_arrays is not None:
            for prediction_name in self.prediction_names:
                self._plot_prediction_over_times(prediction_arrays[prediction_name], prediction_name)

        if self.plot_activations and activation_index:
            self._plot_activation(activation_index)

        plt.show(block=False)
        plt.pause(0.001)

    def _plot_prediction_over_times(self, new_prediction, prediction_name):
        assert prediction_name in self.prediction_names

        ax = self.subplots[prediction_name]
        self.prediction_queues[prediction_name].append(new_prediction)
        predictions = np.stack(self.prediction_queues[prediction_name], axis=0)
        num_predictions = len(predictions)
        t_values = list(range(-num_predictions, 0, 1))
        ax.clear()
        ax.set(ylabel=prediction_name)
        ax.set(xlabel="t")
        ax.set(xlim=[-self.x_size, 0])
        ax.set(ylim=[0,1])
        ax.set_autoscaley_on(False)
        ax.set_autoscalex_on(False)
        for class_ind in range(0, self.n_classes):
            predictions_per_class = predictions[:, class_ind]
            class_color = self._get_color(class_ind)

            ax.plot(t_values, predictions_per_class, color=class_color)





    def _plot_activation(self, activation_index):
        pass

