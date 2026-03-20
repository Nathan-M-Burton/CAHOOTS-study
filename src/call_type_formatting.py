"""Call-type name normalization for the OMD analysis."""
import re
import unicodedata

import pandas as pd

from config import CALLTYPE_RENAME_MAP


def normalize_call_type(x):
    """Strip invisible characters, normalize dashes and whitespace."""
    if pd.isna(x):
        return ""
    s = str(x)
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\u00A0", " ")
    s = re.sub(r"[\u200B-\u200D\uFEFF]", "", s)
    s = re.sub(r"[\u2010\u2011\u2012\u2013\u2014\u2212]", "-", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def format_call_type_name(x):
    """Title-case a call type and apply the project rename map."""
    s = normalize_call_type(x)
    if s == "":
        return ""
    s = s.title()
    return CALLTYPE_RENAME_MAP.get(s, s)
