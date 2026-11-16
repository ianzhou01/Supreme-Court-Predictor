"""
Data loading and cleaning for the SCDB case-centered dataset.
"""

from os import PathLike
from typing import TypeAlias

import pandas as pd


Path: TypeAlias = str | PathLike[str]

ID_COLUMNS = (
    "caseId",
    "docketId",
    "caseIssuesId",
    "voteId",
    "usCite",
    "sctCite",
    "ledCite",
    "lexisCite",
    "docket",
    "caseName",
)

POST_DECISION_COLUMNS = (
    "dateDecision",
    "decisionType",
    "declarationUncon",
    "caseDisposition",
    "caseDispositionUnusual",
    "partyWinning",
    "precedentAlteration",
    "voteUnclear",
    "decisionDirection",
    "decisionDirectionDissent",
    "authorityDecision1",
    "authorityDecision2",
    "majOpinWriter",
    "majOpinAssigner",
    "splitVote",
    "majVotes",
    "minVotes",
)

AGREED_DROP_COLUMNS = (
    "petitionerState",
    "respondentState",
    "caseSourceState",
    "caseOriginState",
    "adminActionState",
    "lawMinor",
    "lawType",
    "lawSupp",
    "chief",
)

CIRCUIT_CODE_MEANINGS = '''
Circuit-group output codes used by Katz et al. (2017):

    0      State court, no applicable federal circuit, missing, or unknown
    1-11   First through Eleventh U.S. Courts of Appeals
    12     U.S. Court of Appeals for the District of Columbia Circuit
    13     Federal Circuit and related specialized federal courts
    14     Military appellate courts
    22     U.S. Tax Court and bankruptcy courts
    99     Territorial, emergency, consular, or other historical special courts

The dictionary
    - keys are raw SCDB caseSource/caseOrigin court codes.
    - values are the broader circuit-group codes described above.

For example
    - raw code 21 (First Circuit) maps to 1
    - raw code 48 (Central District of California) maps to 9
    - state-court code 300 maps to 0.

Mapping source:
https://github.com/mjbommar/scotus-predict-v2/blob/master/src/legacy_model_data.py
'''


COURT_CIRCUIT_MAP = {
    1: 13, 2: 13, 3: 13, 4: 14, 5: 14, 6: 13, 7: 13, 8: 13,
    9: 22, 10: 99, 12: 9, 13: 99, 14: 13, 15: 99, 16: 99, 17: 99,
    18: 99, 19: 0, 20: 22, 21: 1, 22: 2, 23: 3, 24: 4, 25: 5,
    26: 6, 27: 7, 28: 8, 29: 9, 30: 10, 31: 11, 32: 12, 41: 11,
    42: 11, 43: 11, 44: 9, 45: 9, 46: 8, 47: 8, 48: 9, 49: 9,
    50: 9, 51: 9, 52: 10, 53: 2, 54: 3, 55: 12, 56: 11, 57: 11,
    58: 11, 59: 11, 60: 11, 61: 11, 62: 9, 63: 9, 64: 9, 65: 7,
    66: 7, 67: 7, 68: 7, 69: 7, 70: 8, 71: 8, 72: 10, 73: 6,
    74: 6, 75: 5, 76: 5, 77: 5, 78: 1, 79: 4, 80: 1, 81: 6,
    82: 6, 83: 8, 84: 5, 85: 5, 86: 8, 87: 8, 88: 9, 89: 8,
    90: 9, 91: 1, 92: 3, 93: 10, 94: 2, 95: 2, 96: 2, 97: 2,
    98: 4, 99: 4, 100: 4, 101: 8, 102: 9, 103: 6, 104: 6, 105: 10,
    106: 10, 107: 10, 108: 9, 109: 3, 110: 3, 111: 3, 112: 1,
    113: 1, 114: 4, 115: 8, 116: 6, 117: 6, 118: 6, 119: 5, 120: 5,
    121: 5, 122: 5, 123: 10, 124: 2, 125: 3, 126: 4, 127: 4,
    128: 9, 129: 9, 130: 4, 131: 4, 132: 7, 133: 7, 134: 10,
    150: 5, 151: 9, 152: 4, 153: 7, 155: 4, 160: 4, 162: 11,
    163: 5, 164: 11, 165: 7, 166: 7, 167: 8, 168: 6, 169: 5,
    170: 8, 171: 3, 172: 3, 173: 2, 174: 4, 175: 6, 176: 3,
    177: 3, 178: 5, 179: 4, 180: 4, 181: 7, 182: 6, 183: 3,
    184: 9, 185: 11, 186: 8, 187: 5, 300: 0, 301: 0, 302: 0,
    400: 99, 401: 99, 402: 99, 403: 11, 404: 8, 405: 9, 406: 2,
    407: 3, 408: 11, 409: 11, 410: 7, 411: 7, 412: 8, 413: 10,
    414: 6, 415: 5, 416: 1, 417: 4, 418: 1, 419: 6, 420: 8,
    421: 5, 422: 8, 423: 9, 424: 1, 425: 3, 426: 2, 427: 4,
    428: 6, 429: 9, 430: 3, 431: 1, 432: 4, 433: 6, 434: 5,
    435: 2, 436: 4, 437: 4, 438: 7, 439: 10, 440: 12, 441: 8,
    442: 10, 443: 9,
}


def load_data(path: Path) -> pd.DataFrame:
    """
    Load SCDB cases that satisfy the team's row-eligibility rules.

    Only decision types 1, 6, and 7 and rows with a recorded binary
    ``partyWinning`` value are kept.
    ``partyWinning`` is not the prediction target;
    the project predicts affirm versus overturn.
    """
    data = pd.read_csv(path, encoding="latin-1")
    print(f"Number of rows loaded: {len(data):,}")

    data = data.loc[data["decisionType"].isin([1, 6, 7])]
    print(f"Number of rows after decisionType filter: {len(data):,}")

    data = data.loc[data["partyWinning"].isin([0, 1])]
    print(f"Number of rows after partyWinning filter: {len(data):,}")

    return data.reset_index(drop=True)


def handle_missing(data: pd.DataFrame) -> pd.DataFrame:
    """
    Remove excluded features and encode missing categories.

    Callers must extract the affirm/overturn target before this function drops
    post-decision columns such as ``caseDisposition``.
    The input DataFrame is not modified.
    """
    excluded_columns = [
        *ID_COLUMNS,
        *POST_DECISION_COLUMNS,
        *AGREED_DROP_COLUMNS,
    ]
    cleaned = data.drop(columns=excluded_columns, errors="ignore").copy()

    if "threeJudgeFdc" in cleaned.columns:
        cleaned["threeJudgeFdc"] = cleaned["threeJudgeFdc"].fillna(0)

    return cleaned.fillna(-1)


def _map_circuit(value: object) -> int:
    """
    Map one SCDB court code to Katz's circuit grouping.
    """
    if pd.isna(value):
        return 0

    try:
        return COURT_CIRCUIT_MAP.get(int(value), 0)
    except (TypeError, ValueError, OverflowError):
        return 0


def add_circuit_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Add circuit features for case source and origin court codes.
    """
    required_columns = {"caseSource", "caseOrigin"}
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        raise KeyError(f"Missing required court columns: {sorted(missing_columns)}")

    featured = data.copy()
    featured["caseSourceCircuit"] = featured["caseSource"].map(_map_circuit)
    featured["caseOriginCircuit"] = featured["caseOrigin"].map(_map_circuit)
    return featured
