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
    s = re.sub(r"\bfed\.?\s*sts\b", "federated states", s)
    s = re.sub(r"\bfed\.?\s*states\b", "federated states", s)
    s = re.sub(r"\bfed\b", "federal", s)

    # collapse spaces
    s = _WS_RX.sub(" ", s)
    return s.strip()


def normalize_token(s: str) -> str:
    """Lowercase, strip accents, remove punctuation, collapse whitespace, then WB/UN rules."""
    s = (s or "").strip()
    s = strip_accents(s)
    s = s.lower()
    s = _PUNCT_RX.sub(" ", s)  # remove punctuation/symbols (., ’, , ...)
    s = _WS_RX.sub(" ", s)  # collapse spaces
    s = s.strip()
    s = _post_token_rules(s)
    return s


# --- 2) Canonical name mapping ---
# Left side: *normalized tokens* (via normalize_token)
# Right side: canonical "Country" (English)

CANONICAL = {
    # Turkey
    "turkiye": "Turkey",
    "turkiye cumhuriyeti": "Turkey",
    "türkiye": "Turkey",
    "republic of turkey": "Turkey",
    "turkey": "Turkey",
    # Czechia
    "czech republic": "Czechia",
    "czechia": "Czechia",
    # Russia
    "russian federation": "Russia",
    "russia": "Russia",
    # Vietnam
    "viet nam": "Vietnam",
    "vietnam": "Vietnam",
    # Syria
    "syrian arab republic": "Syria",
    "syria": "Syria",
    # Iran
    "iran islamic republic of": "Iran",
    "islamic republic of iran": "Iran",
    "iran islamic rep": "Iran",
    "iran": "Iran",
    # Laos
    "lao pdr": "Laos",
    "lao people s democratic republic": "Laos",
    "laos": "Laos",
    # Gambia
    "gambia the": "Gambia",
    "the gambia": "Gambia",
    "gambia": "Gambia",
    # Bahamas
    "bahamas the": "Bahamas",
    "the bahamas": "Bahamas",
    "bahamas": "Bahamas",
    # Cabo Verde
    "cabo verde": "Cabo Verde",
    "cape verde": "Cabo Verde",
    # Côte d’Ivoire
    "cote d ivoire": "Côte d’Ivoire",
    "cote d'ivoire": "Côte d’Ivoire",
    "ivory coast": "Côte d’Ivoire",
    "ivory coast cote d ivoire": "Côte d’Ivoire",
    # Eswatini
    "eswatini": "Eswatini",
    "swaziland": "Eswatini",
    # Myanmar
    "myanmar": "Myanmar",
    "burma": "Myanmar",
    # Timor-Leste
    "timor leste": "Timor-Leste",
    "east timor": "Timor-Leste",
    # Brunei
    "brunei darussalam": "Brunei",
    "brunei": "Brunei",
    # Congo (DRC)
    "democratic republic of the congo": "Congo (Democratic Republic of the)",
    "congo democratic republic of the": "Congo (Democratic Republic of the)",
    "congo dem rep": "Congo (Democratic Republic of the)",
    "dr congo": "Congo (Democratic Republic of the)",
    "drc": "Congo (Democratic Republic of the)",
    # Congo (Republic)
    "republic of the congo": "Congo",
    "congo rep": "Congo",
    "congo": "Congo",
    # Korea (South)
    "korea rep": "Korea, Republic of",
    "republic of korea": "Korea, Republic of",
    "south korea": "Korea, Republic of",
    "korea republic of": "Korea, Republic of",
    # Korea (North) — UN & WB & short forms
    "korea democratic people s republic of": "Korea, Democratic People’s Republic of",
    "korea democratic peoples republic of": "Korea, Democratic People’s Republic of",
    "democratic people s republic of korea": "Korea, Democratic People’s Republic of",
    "democratic peoples republic of korea": "Korea, Democratic People’s Republic of",
    "dem peoples republic of korea": "Korea, Democratic People’s Republic of",
    "dem people s republic of korea": "Korea, Democratic People’s Republic of",
    "Dem. People's Republic of Korea": "Korea, Democratic People’s Republic of",
    "korea dem people s rep": "Korea, Democratic People’s Republic of",
    "north korea": "Korea, Democratic People’s Republic of",
    "dpr korea": "Korea, Democratic People’s Republic of",
    "dprk": "Korea, Democratic People’s Republic of",
    # Hong Kong / Macao
    "china hong kong sar": "Hong Kong SAR, China",
    "hong kong sar china": "Hong Kong SAR, China",
    "hong kong": "Hong Kong SAR, China",
    "china macao sar": "Macao SAR, China",
    "macao sar china": "Macao SAR, China",
    "macau": "Macao SAR, China",
    "macao": "Macao SAR, China",
    # Moldova
    "moldova": "Moldova",
    "republic of moldova": "Moldova",
    # United States
    "united states": "United States",
    "united states of america": "United States",
    "usa": "United States",
    "u s a": "United States",
    "u s": "United States",
    # United Kingdom
    "united kingdom": "United Kingdom",
    "uk": "United Kingdom",
    "u k": "United Kingdom",
    "great britain": "United Kingdom",
    "britain": "United Kingdom",
    # Palestine
    "state of palestine": "Palestine",
    "palestine": "Palestine",
    "west bank and gaza": "Palestine",
    # Territories / special cases
    "cocos keeling islands": "Cocos (Keeling) Islands",
    "micronesia federated states of": "Micronesia, Federated States of",
    "micronesia federal states of": "Micronesia, Federated States of",
    "micronesia federated states": "Micronesia, Federated States of",
    "micronesia fed states of": "Micronesia, Federated States of",
    "micronesia fed states": "Micronesia, Federated States of",
    "micronesia fed sts": "Micronesia, Federated States of",
    # Bolivia, Tanzania, Venezuela (UN names)
    "bolivia plurinational state of": "Bolivia (Plurinational State of)",
    "bolivia": "Bolivia (Plurinational State of)",
    "tanzania united republic of": "Tanzania, United Republic of",
    "united republic of tanzania": "Tanzania, United Republic of",
    "tanzania": "Tanzania, United Republic of",
    "venezuela bolivarian republic of": "Venezuela (Bolivarian Republic of)",
    "venezuela": "Venezuela (Bolivarian Republic of)",
    # Bahrain (TR spelling)
    "bahrein": "Bahrain",
    "bahrain": "Bahrain",
    # WB ↔ UN harmonization we saw in your data
    "egypt arab republic": "Egypt",
    "egypt arab rep": "Egypt",
    "egypt": "Egypt",
    "curacao": "Curaçao",
    "curaçao": "Curaçao",
    "faroe islands": "Faroe Islands",
    "faeroe islands": "Faroe Islands",
    "slovak republic": "Slovakia",
    "slovakia": "Slovakia",
    "kyrgyz republic": "Kyrgyzstan",
    "kyrgyzstan": "Kyrgyzstan",
    "north macedonia": "North Macedonia",
    "macedonia the former yugoslav republic of": "North Macedonia",
    "macedonia former yugoslav republic of": "North Macedonia",
    "macedonia fyrom": "North Macedonia",
    "somalia fed rep": "Somalia",
    "somalia federal republic": "Somalia",
    "somalia": "Somalia",
    "puerto rico": "Puerto Rico",
    "puerto rico us": "Puerto Rico",
    # A few UN territories often appearing
    "falkland islands malvinas": "Falkland Islands (Malvinas)",
    "holy see": "Holy See",
    "guadeloupe": "Guadeloupe",
    "martinique": "Martinique",
    "mayotte": "Mayotte",
    "french guiana": "French Guiana",
    "montserrat": "Montserrat",
    "melanesia": "Melanesia",
    "micronesia": "Micronesia",
}

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

    # Direct alias hit
    if key in CANONICAL:
        return CANONICAL[key]

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
            "canonical": ser.map(lambda x: CANONICAL.get(normalize_token(x), None)),
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
    ]
    df = pd.DataFrame({"Country": tests})
    print(standardize_country_column(df))
