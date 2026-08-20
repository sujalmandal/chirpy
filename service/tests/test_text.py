from app.text import complete_sentences


def test_keeps_partial_sentence() -> None:
    assert complete_sentences("First sentence. Second") == (["First sentence."], "Second")


def test_flushes_final_fragment() -> None:
    assert complete_sentences("A final thought", final=True) == (["A final thought"], "")
