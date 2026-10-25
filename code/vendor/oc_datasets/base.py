class BaseDataset:
    def __init__(self, reader_cfg=None, **kwargs):
        self.reader_cfg = reader_cfg or {}
