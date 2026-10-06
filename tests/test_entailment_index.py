"""quorum.ensemble.entailment_index must pick the entailment class regardless of label order or naming."""
import pytest
from quorum.ensemble import entailment_index


@pytest.mark.parametrize("id2label,expected", [
    ({0: "entailment", 1: "not_entailment"}, 0),                        # PrismNLI / zeroshot-v2.0(-c) order
    ({0: "not_entailment", 1: "entailment"}, 1),                        # the order the old substring heuristic got wrong
    ({0: "contradiction", 1: "entailment", 2: "neutral"}, 1),           # cross-encoder/nli-deberta-v3-large style
    ({"0": "ENTAILMENT", "1": "NEUTRAL", "2": "CONTRADICTION"}, 0),     # string keys, upper case
    ({0: "entails", 1: "neutral"}, 0),                                  # prefix fallback
])
def test_picks_entailment(id2label, expected):
    assert entailment_index(id2label) == expected


@pytest.mark.parametrize("id2label", [{0: "LABEL_0", 1: "LABEL_1"}, {0: "entailment", 1: "entailment"}])
def test_ambiguous_raises(id2label):
    with pytest.raises(ValueError):
        entailment_index(id2label)
