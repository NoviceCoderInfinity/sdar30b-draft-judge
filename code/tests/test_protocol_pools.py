"""CPU tests for the training-prompt pool helpers in research_protocol (no dataset access)."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import research_protocol as P


def test_last_boxed_balanced():
    assert P.last_boxed(r"so \boxed{\frac{1}{2}} and \boxed{x^{2}+1}.") == "x^{2}+1"
    assert P.last_boxed(r"\boxed{}") == ""
    assert P.last_boxed("no box") is None
    assert P.last_boxed(r"\boxed{unbalanced") is None


def test_split_base_and_slicing(monkeypatch):
    calls = []
    monkeypatch.setitem(P.POOLS, "fake_pool", lambda: (calls.append(1) or [{"idx": i} for i in range(10)]))
    assert P.split_base("fake_pool:2:5") == "fake_pool"
    assert [x["idx"] for x in P.items_for_split("fake_pool:2:5")] == [2, 3, 4]
    assert [x["idx"] for x in P.items_for_split("fake_pool::3")] == [0, 1, 2]
    assert [x["idx"] for x in P.items_for_split("fake_pool:8")] == [8, 9]
    assert len(P.items_for_split("fake_pool")) == 10
    for name in ("gsm8k_pool", "math_pool", "mbpp_pool", "math500"):
        assert P.GEN_LEN[name] in (512, 1024)
