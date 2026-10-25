class _Registry:
    """No-op stand-in for opencompass' mmengine Registry."""
    def __init__(self, name):
        self.name = name
    def register_module(self, name=None, force=False, module=None):
        def _wrap(obj):
            return obj
        if module is not None:
            return module
        if callable(name):          # used bare: @REGISTRY.register_module
            return name
        return _wrap

ICL_EVALUATORS = _Registry('icl_evaluators')
LOAD_DATASET = _Registry('load_dataset')
TEXT_POSTPROCESSORS = _Registry('text_postprocessors')
