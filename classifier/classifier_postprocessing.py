class Postprocessor:
    def __call__(self, probabilities):
        """
        :param probabilities: task containing softmaxed output of model in shape [n_classes]
        :return: task containing postprocessed output in shape [n_classes]
        """
        pass


class ClassifierPostprocessing(Postprocessor):
    ""
    def __init__(self, non_gesture_indices, classification_thresh):
        """
        :param indices_of_non_gestures: indices of classes that are not valid gestures (No_Gesture, Doing_other_things, etc.)
        """
        self.non_gesture_indices = non_gesture_indices

    def __call__(self, probabilities):
        return probabilities
