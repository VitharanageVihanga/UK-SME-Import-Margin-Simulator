"""
Pytest suite for scripts/data_merge.py
Covers: helper extraction functions (unit-testable without I/O).
"""
import pytest
from scripts.data_merge import extract_country_code_from_name, extract_main_sitc_section


class TestExtractCountryCode:
    def test_standard_format(self):
        assert extract_country_code_from_name("AE United Arab Emirates") == "AE"

    def test_two_letter_only(self):
        assert extract_country_code_from_name("US") == "US"

    def test_none_input(self):
        assert extract_country_code_from_name(None) is None

    def test_lowercase_rejected(self):
        assert extract_country_code_from_name("ae Something") is None

    def test_three_letter_rejected(self):
        assert extract_country_code_from_name("USA United States") is None


class TestExtractMainSitcSection:
    def test_single_digit(self):
        assert extract_main_sitc_section("0 Food & live animals") == 0

    def test_two_digit_sub(self):
        assert extract_main_sitc_section("01 Meat & meat preparations") == 0

    def test_three_digit_sub(self):
        assert extract_main_sitc_section("792 Aircraft") == 7

    def test_section_7(self):
        assert extract_main_sitc_section("7 Machinery & transport equipment") == 7

    def test_total_skipped(self):
        assert extract_main_sitc_section("T Total") is None

    def test_none_input(self):
        assert extract_main_sitc_section(None) is None

    def test_empty_string(self):
        assert extract_main_sitc_section("") is None
