class Postprocessor:
    def __call__(self, probabilities):
        """
        :param probabilities: task containing softmaxed output of model in shape [n_classes]
        :return: task containing postprocessed output in shape [n_classes]
        """
        pass


class ClassifierPostprocessing(Postprocessor):
    ""
    def __init__(self, postprocessing_transforms):
        self.postprocessing_transforms = postprocessing_transforms

    def __call__(self, probabilities):
        return probabilities
