from pathlib import Path

from ai_journey.web_extraction_config import load_extraction_policy


def test_default_config_enforces_the_day_65_hundred_page_gate():
    root = Path(__file__).resolve().parents[1]

    policy = load_extraction_policy(root / "config/html_extraction_100_pages.json")

    assert policy.minimum_pages == 100
    assert policy.minimum_text_characters == 200
    assert policy.maximum_replacement_fraction == 0.005
