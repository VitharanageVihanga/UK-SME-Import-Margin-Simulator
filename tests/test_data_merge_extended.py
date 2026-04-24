"""
Extended tests for scripts/data_merge.py
Targets missed lines: normalize_columns, prepare_hmrc, prepare_ons_commodity,
compute_ons_coverage_by_sitc, create_hs2_coverage_from_sitc,
merge_hmrc_ons_totals, save_coverage_output, HS2_TO_SITC mapping.
"""
import os
import pytest
import pandas as pd
import numpy as np

from scripts.data_merge import (
    normalize_columns,
    prepare_hmrc,
    prepare_ons_commodity,
    compute_ons_coverage_by_sitc,
    create_hs2_coverage_from_sitc,
    merge_hmrc_ons_totals,
    save_coverage_output,
    HS2_TO_SITC,
    SITC_NAMES,
)


# ── shared fixtures ───────────────────────────────────────────────────────────

@pytest.fixture
def raw_hmrc():
    return pd.DataFrame({
        "partner_country": ["DE", "FR", "YY", "XX", "DE", "CN", "ZZ", ""],
        "commodity":       [8471,  8471,  8471,  8471, 170100, 2701,  84,  85],
        "year":            [2020,  2020,  2020,  2020,  2021,  2021, 2021, 2021],
        "value":           [100,   200,   300,   400,   500,   600,  700,  800],
        "net_mass":        [10,    20,    30,    40,    50,    60,   70,   80],
    })


@pytest.fixture
def raw_ons_commodity():
    return pd.DataFrame({
        "country_name": [
            "DE Germany",
            "FR France",
            "invalid_no_code",
            "T Total",
            "AE United Arab Emirates",
            "CN China",
        ],
        "commodity": [
            "7 Machinery & transport equipment",
            "7 Machinery & transport equipment",
            "7 Machinery & transport equipment",
            "T Total",
            "0 Food & live animals",
            "5 Chemicals",
        ],
        "year":   [2020, 2020, 2020, 2020, 2021, 2021],
        "import_value_million_gbp": [1000.0, 500.0, 100.0, 9999.0, 250.0, 400.0],
    })


@pytest.fixture
def prepared_hmrc(raw_hmrc):
    return prepare_hmrc(raw_hmrc)


@pytest.fixture
def prepared_ons(raw_ons_commodity):
    return prepare_ons_commodity(raw_ons_commodity)


@pytest.fixture
def sitc_coverage_10yr():
    return pd.DataFrame({
        "sitc_section":   [0,   5,   7],
        "sitc_name":      ["0 Food & live animals", "5 Chemicals", "7 Machinery & transport equipment"],
        "years_with_data": [5,  3,   7],
        "total_value":    [5000.0, 1500.0, 9000.0],
        "total_years":    [10, 10,  10],
        "coverage_pct":   [50.0, 30.0, 70.0],
    })


@pytest.fixture
def raw_ons_totals():
    return pd.DataFrame({
        "country_code": ["DE", "FR", "DE"],
        "year": [2020, 2020, 2021],
        "import_value_million_gbp": [5000.0, 3000.0, 5500.0],
    })


# ── normalize_columns (lines 116-123) ────────────────────────────────────────

class TestNormalizeColumns:
    def test_lowercases(self):
        df = pd.DataFrame({"UPPER": [1], "MixedCase": [2]})
        result = normalize_columns(df)
        assert "upper" in result.columns
        assert "mixedcase" in result.columns

    def test_strips_whitespace(self):
        df = pd.DataFrame({" Name ": [1], "Value ": [2]})
        result = normalize_columns(df)
        assert "name" in result.columns
        assert "value" in result.columns

    def test_replaces_spaces_with_underscore(self):
        df = pd.DataFrame({"First Name": [1], "Last Name": [2]})
        result = normalize_columns(df)
        assert "first_name" in result.columns
        assert "last_name" in result.columns

    def test_replaces_hyphens(self):
        df = pd.DataFrame({"import-value": [1], "net-mass": [2]})
        result = normalize_columns(df)
        assert "import_value" in result.columns
        assert "net_mass" in result.columns

    def test_returns_same_dataframe(self):
        df = pd.DataFrame({"col_a": [1, 2]})
        result = normalize_columns(df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2


# ── prepare_hmrc (lines 149-184) ─────────────────────────────────────────────

class TestPrepareHmrc:
    def test_renames_partner_country_to_country_code(self, raw_hmrc):
        result = prepare_hmrc(raw_hmrc)
        assert "country_code" in result.columns
        assert "partner_country" not in result.columns

    def test_removes_yy_code(self, raw_hmrc):
        result = prepare_hmrc(raw_hmrc)
        assert "YY" not in result["country_code"].values

    def test_removes_zz_code(self, raw_hmrc):
        result = prepare_hmrc(raw_hmrc)
        assert "ZZ" not in result["country_code"].values

    def test_removes_xx_code(self, raw_hmrc):
        result = prepare_hmrc(raw_hmrc)
        assert "XX" not in result["country_code"].values

    def test_removes_empty_string_code(self, raw_hmrc):
        result = prepare_hmrc(raw_hmrc)
        assert "" not in result["country_code"].fillna("").values

    def test_hs2_chapter_four_digit_commodity(self, raw_hmrc):
        result = prepare_hmrc(raw_hmrc)
        assert 84 in result["hs2_chapter"].values

    def test_hs2_chapter_six_digit_commodity(self, raw_hmrc):
        result = prepare_hmrc(raw_hmrc)
        assert 17 in result["hs2_chapter"].values

    def test_hs2_chapter_two_digit_commodity(self, raw_hmrc):
        result = prepare_hmrc(raw_hmrc)
        assert 27 in result["hs2_chapter"].values

    def test_sitc_section_84_maps_to_7(self, prepared_hmrc):
        row = prepared_hmrc[prepared_hmrc["hs2_chapter"] == 84]
        assert not row.empty
        assert row["sitc_section"].iloc[0] == 7

    def test_sitc_section_27_maps_to_3(self, prepared_hmrc):
        row = prepared_hmrc[prepared_hmrc["hs2_chapter"] == 27]
        assert not row.empty
        assert row["sitc_section"].iloc[0] == 3

    def test_sitc_name_populated(self, prepared_hmrc):
        valid = prepared_hmrc.dropna(subset=["sitc_section"])
        assert valid["sitc_name"].notna().all()

    def test_invalid_commodity_gives_null_chapter(self):
        df = pd.DataFrame({
            "partner_country": ["DE"],
            "commodity":       ["BAD"],
            "year":            [2020],
            "value":           [100],
            "net_mass":        [10],
        })
        result = prepare_hmrc(df)
        assert result["hs2_chapter"].iloc[0] is None

    def test_notna_filter_removes_null_country(self):
        df = pd.DataFrame({
            "partner_country": ["DE", None, np.nan],
            "commodity":       [8471, 8471, 8471],
            "year":            [2020, 2020, 2020],
            "value":           [100,  200,  300],
            "net_mass":        [10,   20,   30],
        })
        result = prepare_hmrc(df)
        assert result["country_code"].notna().all()

    def test_returns_dataframe(self, raw_hmrc):
        result = prepare_hmrc(raw_hmrc)
        assert isinstance(result, pd.DataFrame)


# ── prepare_ons_commodity (lines 189-223) ────────────────────────────────────

class TestPrepareOnsCommodity:
    def test_country_code_extracted(self, raw_ons_commodity):
        result = prepare_ons_commodity(raw_ons_commodity)
        assert "DE" in result["country_code"].values
        assert "FR" in result["country_code"].values
        assert "AE" in result["country_code"].values

    def test_invalid_country_rows_dropped(self, raw_ons_commodity):
        result = prepare_ons_commodity(raw_ons_commodity)
        assert result["country_code"].notna().all()

    def test_total_row_dropped(self, raw_ons_commodity):
        result = prepare_ons_commodity(raw_ons_commodity)
        assert result["sitc_section"].notna().all()

    def test_commodity_renamed_to_sitc_name_raw(self, raw_ons_commodity):
        result = prepare_ons_commodity(raw_ons_commodity)
        assert "sitc_name_raw" in result.columns
        assert "commodity" not in result.columns

    def test_sitc_section_column_present(self, raw_ons_commodity):
        result = prepare_ons_commodity(raw_ons_commodity)
        assert "sitc_section" in result.columns

    def test_sitc_name_column_present(self, raw_ons_commodity):
        result = prepare_ons_commodity(raw_ons_commodity)
        assert "sitc_name" in result.columns

    def test_section_7_correctly_mapped(self, raw_ons_commodity):
        result = prepare_ons_commodity(raw_ons_commodity)
        row = result[result["sitc_section"] == 7]
        assert not row.empty

    def test_section_0_correctly_mapped(self, raw_ons_commodity):
        result = prepare_ons_commodity(raw_ons_commodity)
        row = result[result["sitc_section"] == 0]
        assert not row.empty

    def test_returns_dataframe(self, raw_ons_commodity):
        result = prepare_ons_commodity(raw_ons_commodity)
        assert isinstance(result, pd.DataFrame)


# ── compute_ons_coverage_by_sitc (lines 231-270) ─────────────────────────────

class TestComputeOnsCoverageBySitc:
    def test_returns_tuple_of_two(self, prepared_ons):
        result = compute_ons_coverage_by_sitc(prepared_ons)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_first_element_is_dataframe(self, prepared_ons):
        sitc_cov, _ = compute_ons_coverage_by_sitc(prepared_ons)
        assert isinstance(sitc_cov, pd.DataFrame)

    def test_second_element_is_int(self, prepared_ons):
        _, total_years = compute_ons_coverage_by_sitc(prepared_ons)
        assert isinstance(total_years, int)

    def test_total_years_matches_unique_years(self, prepared_ons):
        _, total_years = compute_ons_coverage_by_sitc(prepared_ons)
        assert total_years == prepared_ons["year"].nunique()

    def test_expected_columns_present(self, prepared_ons):
        sitc_cov, _ = compute_ons_coverage_by_sitc(prepared_ons)
        for col in ("sitc_section", "sitc_name", "years_with_data", "coverage_pct"):
            assert col in sitc_cov.columns

    def test_coverage_pct_between_0_and_100(self, prepared_ons):
        sitc_cov, _ = compute_ons_coverage_by_sitc(prepared_ons)
        assert (sitc_cov["coverage_pct"] >= 0).all()
        assert (sitc_cov["coverage_pct"] <= 100).all()

    def test_has_data_flag_positive_for_nonzero_value(self, prepared_ons):
        sitc_cov, _ = compute_ons_coverage_by_sitc(prepared_ons)
        assert (sitc_cov["years_with_data"] >= 0).all()


# ── create_hs2_coverage_from_sitc (lines 278-330) ────────────────────────────

class TestCreateHs2CoverageFromSitc:
    def test_returns_dataframe(self, sitc_coverage_10yr):
        result = create_hs2_coverage_from_sitc(sitc_coverage_10yr, total_years=10)
        assert isinstance(result, pd.DataFrame)

    def test_has_required_columns(self, sitc_coverage_10yr):
        result = create_hs2_coverage_from_sitc(sitc_coverage_10yr, total_years=10)
        for col in ("commodity", "hs2_chapter", "sitc_section", "coverage_class",
                    "ons_coverage_pct", "ons_covered_years"):
            assert col in result.columns

    def test_all_hs2_chapters_covered(self, sitc_coverage_10yr):
        result = create_hs2_coverage_from_sitc(sitc_coverage_10yr, total_years=10)
        expected = set(HS2_TO_SITC.keys())
        actual = set(result["hs2_chapter"].values)
        assert expected == actual

    def test_coverage_classes_are_valid(self, sitc_coverage_10yr):
        result = create_hs2_coverage_from_sitc(sitc_coverage_10yr, total_years=10)
        valid = {"High coverage", "Partial coverage", "Low coverage", "No coverage"}
        assert set(result["coverage_class"].unique()).issubset(valid)

    def test_high_coverage_threshold(self):
        sitc = pd.DataFrame({
            "sitc_section": [7],
            "sitc_name": ["7 Machinery & transport equipment"],
            "years_with_data": [10],
            "total_value": [9000.0],
            "total_years": [10],
            "coverage_pct": [100.0],
        })
        result = create_hs2_coverage_from_sitc(sitc, total_years=10)
        sitc7 = result[result["sitc_section"] == 7]
        assert (sitc7["coverage_class"] == "High coverage").all()

    def test_partial_coverage_threshold(self):
        sitc = pd.DataFrame({
            "sitc_section": [5],
            "sitc_name": ["5 Chemicals"],
            "years_with_data": [6],
            "total_value": [1000.0],
            "total_years": [10],
            "coverage_pct": [60.0],
        })
        result = create_hs2_coverage_from_sitc(sitc, total_years=10)
        sitc5 = result[result["sitc_section"] == 5]
        assert (sitc5["coverage_class"] == "Partial coverage").all()

    def test_low_coverage_threshold(self):
        sitc = pd.DataFrame({
            "sitc_section": [0],
            "sitc_name": ["0 Food & live animals"],
            "years_with_data": [3],
            "total_value": [500.0],
            "total_years": [10],
            "coverage_pct": [30.0],
        })
        result = create_hs2_coverage_from_sitc(sitc, total_years=10)
        sitc0 = result[result["sitc_section"] == 0]
        assert (sitc0["coverage_class"] == "Low coverage").all()

    def test_no_coverage_when_sitc_missing(self):
        empty_sitc = pd.DataFrame(
            columns=["sitc_section", "sitc_name", "years_with_data",
                     "total_value", "total_years", "coverage_pct"]
        )
        result = create_hs2_coverage_from_sitc(empty_sitc, total_years=10)
        assert (result["coverage_class"] == "No coverage").all()

    def test_sorted_by_commodity(self, sitc_coverage_10yr):
        result = create_hs2_coverage_from_sitc(sitc_coverage_10yr, total_years=10)
        commodities = result["commodity"].tolist()
        assert commodities == sorted(commodities)


# ── merge_hmrc_ons_totals (lines 335-360) ────────────────────────────────────

class TestMergeHmrcOnsTotals:
    def test_returns_dataframe(self, prepared_hmrc, raw_ons_totals):
        result = merge_hmrc_ons_totals(prepared_hmrc, raw_ons_totals)
        assert isinstance(result, pd.DataFrame)

    def test_has_country_and_year_cols(self, prepared_hmrc, raw_ons_totals):
        result = merge_hmrc_ons_totals(prepared_hmrc, raw_ons_totals)
        assert "country_code" in result.columns
        assert "year" in result.columns

    def test_hmrc_total_value_aggregated(self, prepared_hmrc, raw_ons_totals):
        result = merge_hmrc_ons_totals(prepared_hmrc, raw_ons_totals)
        assert "hmrc_total_value" in result.columns

    def test_output_file_saved(self, prepared_hmrc, raw_ons_totals):
        merge_hmrc_ons_totals(prepared_hmrc, raw_ons_totals)
        assert os.path.exists("data/output/merged_hmrc_ons_totals.csv")

    def test_left_join_preserves_hmrc_rows(self, prepared_hmrc, raw_ons_totals):
        result = merge_hmrc_ons_totals(prepared_hmrc, raw_ons_totals)
        hmrc_agg_len = prepared_hmrc.groupby(["country_code", "year"]).ngroups
        assert len(result) == hmrc_agg_len


# ── save_coverage_output (lines 365-376) ─────────────────────────────────────

class TestSaveCoverageOutput:
    def test_saves_classified_file(self, sitc_coverage_10yr):
        hs2_cov = create_hs2_coverage_from_sitc(sitc_coverage_10yr, total_years=10)
        save_coverage_output(hs2_cov)
        assert os.path.exists("data/output/ons_coverage_by_commodity_classified.csv")

    def test_saves_aggregated_file(self, sitc_coverage_10yr):
        hs2_cov = create_hs2_coverage_from_sitc(sitc_coverage_10yr, total_years=10)
        save_coverage_output(hs2_cov)
        assert os.path.exists("data/output/ons_coverage_by_commodity_aggregated.csv")

    def test_saved_file_readable(self, sitc_coverage_10yr):
        hs2_cov = create_hs2_coverage_from_sitc(sitc_coverage_10yr, total_years=10)
        save_coverage_output(hs2_cov)
        loaded = pd.read_csv("data/output/ons_coverage_by_commodity_classified.csv")
        assert len(loaded) == len(hs2_cov)


# ── HS2_TO_SITC mapping sanity (no new coverage but validates the constant) ──

class TestHs2SitcMapping:
    def test_all_sitc_values_in_0_to_9(self):
        assert set(HS2_TO_SITC.values()).issubset(set(range(10)))

    def test_all_sitc_sections_have_names(self):
        for section in HS2_TO_SITC.values():
            assert section in SITC_NAMES

    def test_hs2_84_to_sitc_7(self):
        assert HS2_TO_SITC[84] == 7

    def test_hs2_27_to_sitc_3(self):
        assert HS2_TO_SITC[27] == 3

    def test_hs2_28_to_sitc_5(self):
        assert HS2_TO_SITC[28] == 5

    def test_hs2_1_to_sitc_0(self):
        assert HS2_TO_SITC[1] == 0
