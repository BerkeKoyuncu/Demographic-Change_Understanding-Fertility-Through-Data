# utils_country.py
# Country name standardization for cross-dataset consistency
# Output target: "Country" column with canonical English names

import re
import unicodedata

import pandas as pd

# --- 1) Text normalization helpers ---

_PUNCT_RX = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WS_RX = re.compile(r"\s+")


def strip_accents(s: str) -> str:
    """Remove accents/diacritics without changing letters."""
    if s is None:
        return s
    s = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in s if not unicodedata.combining(ch))


def _post_token_rules(s: str) -> str:
    """
    Extra WB/UN harmonization rules applied after basic normalization.
    Handle common abbreviations and patterns (no punctuation at this point).
    """
    # unify connectors
    s = s.replace("&", "and")

    # expand common abbreviations (with or without dots initially removed)
    # e.g., "Dem. People’s Rep. of Korea" -> "democratic peoples republic of korea"
    s = re.sub(r"\bdem\b", "democratic", s)
    s = re.sub(r"\brep\b", "republic", s)
    s = re.sub(r"\brep of\b", "republic of", s)
    s = re.sub(r"\barab rep\b", "arab republic", s)
    s = re.sub(r"\bislamic rep\b", "islamic republic", s)

    # DPRK variants after punctuation removal
    s = s.replace("dem peoples", "democratic peoples")
    s = s.replace("dem people s", "democratic peoples")

    # Micronesia (WB short) → Federated States
    s = re.sub(r"\bfed\s*sts\b", "federated states", s)
    s = re.sub(r"\bfed\s*states\b", "federated states", s)
    s = re.sub(r"\bfed\b", "federal", s)

    # collapse spaces
    s = _WS_RX.sub(" ", s)
    return s.strip()


def normalize_token(s: str) -> str:
    """Lowercase, strip accents, remove punctuation, collapse whitespace, then WB/UN rules."""
    s = (s or "").strip()
    s = strip_accents(s)
    s = s.lower()
    s = _PUNCT_RX.sub(" ", s)  # remove punctuation/symbols (., ’, , ... & etc.)
    s = _WS_RX.sub(" ", s)  # collapse spaces
    s = s.strip()
    s = _post_token_rules(s)
    return s


# --- 2) Canonical name mapping ---
# Left side: human-friendly aliases (any case/punct allowed)
# Right side: canonical "Country" (English)
CANONICAL = {
    # Turkey
    "Türkiye": "Turkey",
    "Türkiye Cumhuriyeti": "Turkey",
    "turkiye": "Turkey",
    "republic of turkey": "Turkey",
    "turkey": "Turkey",
    # Czechia
    "Czech Republic": "Czechia",
    "czechia": "Czechia",
    # Russia
    "Russian Federation": "Russia",
    "russia": "Russia",
    # Vietnam
    "Viet Nam": "Vietnam",
    "vietnam": "Vietnam",
    # Syria
    "Syrian Arab Republic": "Syria",
    "syria": "Syria",
    # Iran
    "Iran, Islamic Rep.": "Iran",
    "Iran (Islamic Republic of)": "Iran",
    "Islamic Republic of Iran": "Iran",
    "iran islamic rep": "Iran",
    "iran": "Iran",
    # Laos
    "Lao PDR": "Laos",
    "Lao People's Democratic Republic": "Laos",
    "lao people s democratic republic": "Laos",
    "laos": "Laos",
    # Gambia
    "Gambia, The": "Gambia",
    "The Gambia": "Gambia",
    "gambia": "Gambia",
    # Bahamas
    "Bahamas, The": "Bahamas",
    "The Bahamas": "Bahamas",
    "bahamas": "Bahamas",
    # Slovakia
    "Slovak Republic": "Slovakia",
    "slovakia": "Slovakia",
    # Somalia
    "Somalia, Fed. Rep.": "Somalia",
    "Somalia Federal Republic": "Somalia",
    "somalia fed rep": "Somalia",
    "somalia": "Somalia",
    # St. Kitts and Nevis
    "St. Kitts and Nevis": "St. Kitts and Nevis",
    "Saint Kitts and Nevis": "St. Kitts and Nevis",
    "st kitts and nevis": "St. Kitts and Nevis",
    "saint kitts and nevis": "St. Kitts and Nevis",
    # St. Lucia
    "St. Lucia": "St. Lucia",
    "Saint Lucia": "St. Lucia",
    "st lucia": "St. Lucia",
    "saint lucia": "St. Lucia",
    # St. Vincent and the Grenadines
    "St. Vincent and the Grenadines": "St. Vincent and the Grenadines",
    "Saint Vincent and the Grenadines": "St. Vincent and the Grenadines",
    "st vincent and the grenadines": "St. Vincent and the Grenadines",
    "saint vincent and the grenadines": "St. Vincent and the Grenadines",
    # Cabo Verde
    "Cabo Verde": "Cabo Verde",
    "Cape Verde": "Cabo Verde",
    # Côte d’Ivoire
    "Cote d'Ivoire": "Côte d’Ivoire",
    "Cote d Ivoire": "Côte d’Ivoire",
    "Ivory Coast": "Côte d’Ivoire",
    "Ivory Coast (Cote d'Ivoire)": "Côte d’Ivoire",
    # Eswatini
    "Eswatini": "Eswatini",
    "Swaziland": "Eswatini",
    # Myanmar
    "Myanmar": "Myanmar",
    "Burma": "Myanmar",
    # Timor-Leste
    "Timor-Leste": "Timor-Leste",
    "East Timor": "Timor-Leste",
    # Brunei
    "Brunei Darussalam": "Brunei",
    "Brunei": "Brunei",
    # Congo (DRC)
    "Democratic Republic of the Congo": "Congo (Democratic Republic of the)",
    "Congo, Democratic Republic of the": "Congo (Democratic Republic of the)",
    "Congo, Dem. Rep.": "Congo (Democratic Republic of the)",
    "DR Congo": "Congo (Democratic Republic of the)",
    "DRC": "Congo (Democratic Republic of the)",
    # Congo (Republic)
    "Republic of the Congo": "Congo",
    "Congo, Rep.": "Congo",
    "Congo": "Congo",
    # Korea (South)
    "Korea, Rep.": "Korea, Republic of",
    "Republic of Korea": "Korea, Republic of",
    "South Korea": "Korea, Republic of",
    "Korea Republic of": "Korea, Republic of",
    # Korea (North)
    "Korea, Dem. People's Rep.": "Korea, Democratic People’s Republic of",
    "Korea, Democratic People's Republic of": "Korea, Democratic People’s Republic of",
    "Democratic People's Republic of Korea": "Korea, Democratic People’s Republic of",
    "Dem. People's Republic of Korea": "Korea, Democratic People’s Republic of",
    "DPR Korea": "Korea, Democratic People’s Republic of",
    "DPRK": "Korea, Democratic People’s Republic of",
    "North Korea": "Korea, Democratic People’s Republic of",
    # Hong Kong / Macao
    "China, Hong Kong SAR": "Hong Kong SAR, China",
    "Hong Kong SAR, China": "Hong Kong SAR, China",
    "Hong Kong": "Hong Kong SAR, China",
    "China, Macao SAR": "Macao SAR, China",
    "Macao SAR, China": "Macao SAR, China",
    "Macau": "Macao SAR, China",
    "Macao": "Macao SAR, China",
    # Moldova
    "Republic of Moldova": "Moldova",
    "Moldova": "Moldova",
    # United States
    "United States": "United States of America",
    "United States of America": "United States of America",
    "USA": "United States of America",
    "U S A": "United States of America",
    "U S": "United States of America",
    # Venezuela
    "Venezuela (Bolivarian Republic of)": "Venezuela (Bolivarian Republic of)",
    "Venezuela, RB": "Venezuela (Bolivarian Republic of)",
    "Venezuela": "Venezuela (Bolivarian Republic of)",
    # United States Virgin Islands
    "Virgin Islands (U.S.)": "United States Virgin Islands",
    "United States Virgin Islands": "United States Virgin Islands",
    "virgin islands us": "United States Virgin Islands",
    # Yemen
    "Yemen, Rep.": "Yemen",
    "Republic of Yemen": "Yemen",
    "Yemen": "Yemen",
    # Australia (quirky aggregates sometimes)
    "Australia/New Zealand": "Australia",
    # United Kingdom
    "United Kingdom": "United Kingdom",
    "UK": "United Kingdom",
    "U K": "United Kingdom",
    "Great Britain": "United Kingdom",
    "Britain": "United Kingdom",
    # Palestine
    "State of Palestine": "Palestine",
    "West Bank and Gaza": "Palestine",
    "Palestine": "Palestine",
    # Territories / special cases
    "Cocos (Keeling) Islands": "Cocos (Keeling) Islands",
    "Micronesia, Federated States of": "Micronesia, Federated States of",
    "Micronesia, Fed. Sts.": "Micronesia, Federated States of",
    # North Macedonia
    "TFYR Macedonia": "North Macedonia",
    "North Macedonia": "North Macedonia",
    "Macedonia, the former Yugoslav Republic of": "North Macedonia",
    "Macedonia (the former Yugoslav Republic of)": "North Macedonia",
    # Bolivia, Tanzania (UN names)
    "Bolivia (Plurinational State of)": "Bolivia (Plurinational State of)",
    "Bolivia": "Bolivia (Plurinational State of)",
    "Tanzania, United Republic of": "Tanzania, United Republic of",
    "United Republic of Tanzania": "Tanzania, United Republic of",
    "Tanzania": "Tanzania, United Republic of",
    # Bahrain (TR spelling)
    "Bahrein": "Bahrain",
    "Bahrain": "Bahrain",
    # WB ↔ UN harmonization
    "Egypt, Arab Rep.": "Egypt",
    "Egypt (Arab Republic of)": "Egypt",
    "Arab Republic of Egypt": "Egypt",
    "Egypt": "Egypt",
    "Curacao": "Curaçao",
    "Curaçao": "Curaçao",
    "faroe islands": "Faroe Islands",
    "Faeroe Islands": "Faroe Islands",
    "Kyrgyz Republic": "Kyrgyzstan",
    "Kyrgyzstan": "Kyrgyzstan",
    "Puerto Rico": "Puerto Rico",
    "Puerto Rico (US)": "Puerto Rico",
    "puerto rico us": "Puerto Rico",
    # A few UN territories often appearing
    "Falkland Islands (Malvinas)": "Falkland Islands (Malvinas)",
    "Holy See": "Holy See",
    "Guadeloupe": "Guadeloupe",
    "Martinique": "Martinique",
    "Mayotte": "Mayotte",
    "French Guiana": "French Guiana",
    "Montserrat": "Montserrat",
    "Melanesia": "Melanesia",
}

# Build a normalized alias map once, so lookups always work
ALIAS = {normalize_token(k): v for k, v in CANONICAL.items()}

# Region/aggregate hints (drop_non_countries already handles these, but we avoid remapping)
AGGREGATE_HINTS = {
    "world": "World",
    "euro area": "Euro area",
    "europe": "Europe",
    "sub saharan": "Sub-Saharan Africa",
    "latin america": "Latin America & Caribbean",
    "east asia": "East Asia & Pacific",
    "south asia": "South Asia",
    "middle east": "Middle East & North Africa",
    "ibrd": "IBRD",
    "ida": "IDA",
    "oecd": "OECD",
    "income": "income",
    "demographic dividend": "demographic dividend",
}


def canonical_country(name: str) -> str:
    """Map input country name to canonical form if possible; otherwise return trimmed original."""
    if pd.isna(name):
        return name
    raw = str(name).strip()
    key = normalize_token(raw)

    # Handle "the ..." patterns (e.g., 'the bahamas', 'the gambia')
    if key.startswith("the "):
        key = key[4:]

    # Direct alias hit (normalized)
    if key in ALIAS:
        return ALIAS[key]

    # Looks like an aggregate? keep original (drop step removes)
    for agg in AGGREGATE_HINTS:
        if agg in key:
            return raw

    # Default: return original trimmed
    return raw


def standardize_country_column(df: pd.DataFrame, col: str = "Country") -> pd.DataFrame:
    """Return a copy where df[col] is canonicalized."""
    if col not in df.columns:
        return df
    out = df.copy()
    out[col] = out[col].map(canonical_country)
    return out


def report_unmapped(df: pd.DataFrame, col: str = "Country", sample=30):
    """Quick check for values that might still need mapping."""
    ser = df[col].astype(str)
    pairs = pd.DataFrame(
        {
            "original": ser,
            "normalized": ser.map(normalize_token),
            "canonical": ser.map(lambda x: ALIAS.get(normalize_token(x), None)),
        }
    )
    unmapped = pairs[pairs["canonical"].isna()]
    # Ignore aggregates heuristically
    mask_agg = unmapped["normalized"].str.contains(
        r"world|income|area|region|europe|asia|africa|america|caribbean|sub saharan"
        r"|middle east|north africa"
        r"|east asia|south asia|pacific|latin america|oecd|ibrd|ida|demographic dividend",
        case=False,
        regex=True,
    )
    unmapped = unmapped.loc[~mask_agg, ["original"]].drop_duplicates().head(sample)
    return unmapped


if __name__ == "__main__":
    # quick self-test
    tests = [
        "Korea, Democratic People's Republic of",
        "Dem. People's Republic of Korea",
        "DPRK",
        "DPR Korea",
        "Micronesia, Fed. Sts.",
        "Egypt, Arab Rep.",
        "Curaçao",
        "Curacao",
        "Faeroe Islands",
        "Faroe Islands",
        "Saint Lucia",
        "St. Lucia",
        "Gambia, The",
        "The Bahamas",
    ]
    df = pd.DataFrame({"Country": tests})
    print(standardize_country_column(df))
