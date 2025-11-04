# utils_align.py
from typing import Dict, List, Set, Tuple

import pandas as pd
from utils_country import standardize_country_column


def standardize_and_keep_common(
    dfs: List[pd.DataFrame], col: str = "Country", with_report: bool = True
) -> Tuple[List[pd.DataFrame], Set[str], Dict[str, list]]:
    """
    1) Standardize country names in each dataframe.
    2) Keep only countries that appear in ALL dataframes.

    Returns:
        filtered_dfs: list of filtered DFs (same order as input)
        common: set of common country names after standardization
        report: dict showing dropped countries per DF
    """
    if not dfs:
        return [], set(), {}

    # 1. Standardize all country columns
    standardized = []
    for i, df in enumerate(dfs):
        if col not in df.columns:
            raise KeyError(f"'{col}' column not found in df index {i}")
        standardized.append(standardize_country_column(df, col=col))

    # 2. Find intersection of all country sets
    common = set(standardized[0][col].dropna())
    for df in standardized[1:]:
        common &= set(df[col].dropna())

    # 3. Filter each dataframe to the common set
    filtered = [df[df[col].isin(common)].copy() for df in standardized]

    # 4. Optional report (countries that got dropped)
    report = {}
    if with_report:
        for i, df in enumerate(standardized, start=1):
            dropped = sorted(set(df[col].dropna()) - common)
            report[f"df{i}"] = dropped

    return filtered, common, report
