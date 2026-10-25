class BaseEvaluator:
    def __init__(self, pred_postprocessor=None, **kwargs):
        self.pred_postprocessor = pred_postprocessor
    def score(self, *args, **kwargs):
        raise NotImplementedError
