"""Stubs: imported by modeling_sdar.py but unused on SDAR's decode path."""
def _nope(name):
    def f(*a, **k):
        raise NotImplementedError(
            f"sdar compat flash_attn shim: bert_padding.{name} is not implemented "
            f"(SDAR's decode path does not use it)."
        )
    return f

index_first_axis = _nope("index_first_axis")
pad_input = _nope("pad_input")
unpad_input = _nope("unpad_input")
