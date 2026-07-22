import pytest

from video_annotator.prompts import parse_prompt


def test_category_separators_normalize_to_same_detector_prompt():
    expected = ("dog", "horse")
    for prompt in ("dog and horse", "dog, horse", "dog . horse ."):
        spec = parse_prompt(prompt)
        assert spec.targets == expected
        assert spec.detector_prompt == "dog . horse ."
        assert spec.mode == "category_union"


def test_explicit_targets_are_deduplicated():
    spec = parse_prompt("", explicit_targets=["dog", "horse", "dog"])
    assert spec.targets == ("dog", "horse")


def test_referring_expression_is_not_over_split():
    spec = parse_prompt("the man in black and white clothes", mode="referring")
    assert spec.targets == ()
    assert spec.motion_text == "the man in black and white clothes"


def test_only_and_articles_are_removed_from_simple_category():
    assert parse_prompt("the dog only").targets == ("dog",)


def test_empty_category_prompt_fails():
    with pytest.raises(ValueError):
        parse_prompt("only")
