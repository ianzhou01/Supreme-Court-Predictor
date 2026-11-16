RANDOM_STATE = 42
RARE_CATEGORY_THRESHOLD = 10

TRAIN_END_TERM = 2015
VALIDATION_START_TERM = 2016
VALIDATION_END_TERM = 2020
TEST_START_TERM = 2021
TEST_END_TERM = 2025

TARGET_COLUMN = "partyWinning"
FILTER_COLUMN = "decisionType"
ID_COLUMNS = [
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
]

POST_DECISION_COLUMNS = [
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
]

OTHER_DROP_COLUMNS = [
    "petitionerState",
    "respondentState",
    "caseSourceState",
    "caseOriginState",
    "adminActionState",
    "lawMinor",
    "lawType",
    "lawSupp",
    "chief",
]

CONVERT_COLUMNS = [
    "dateArgument",
    "dateRearg",
]

CATEGORICAL_FEATURES = [
    "naturalCourt",
    "petitioner",
    "respondent",
    "jurisdiction",
    "adminAction",
    "threeJudgeFdc",
    "caseOrigin",
    "caseSource",
    "lcDisagreement",
    "certReason",
    "lcDisposition",
    "lcDispositionDirection",
    "issue",
    "issueArea",
]

NUMERIC_FEATURES = [
    "term",
]

DERIVED_CATEGORICAL_FEATURES = []
DERIVED_NUMERIC_FEATURES = []
RAW_KEEP_COLUMNS = (CATEGORICAL_FEATURES + NUMERIC_FEATURES)
DROP_COLUMNS = (ID_COLUMNS + POST_DECISION_COLUMNS + OTHER_DROP_COLUMNS + CONVERT_COLUMNS)
FEATURE_COLUMNS = (RAW_KEEP_COLUMNS + DERIVED_CATEGORICAL_FEATURES + DERIVED_NUMERIC_FEATURES)
LEAKAGE_REVIEW_COLUMNS = [
    "lcDispositionDirection",
    "issue",
    "issueArea",
]

COLUMN_REASONS = {
    #identifiers
    "caseId": "Drop: database identifier.",
    "docketId": "Drop: database identifier.",
    "caseIssuesId": "Drop: issue-level identifier.",
    "voteId": "Drop: vote-level identifier.",
    "usCite": "Drop: U.S. Reports citation.",
    "sctCite": "Drop: Supreme Court Reporter citation.",
    "ledCite": "Drop: Lawyers' Edition citation.",
    "lexisCite": "Drop: Lexis citation.",
    "docket": "Drop: case docket number.",
    "caseName": "Drop: case name.",

    #post-decision information
    "dateDecision": "Drop: date of decision.",
    "decisionType": "Filter only: excluded from model inputs.",
    "declarationUncon": "Drop: outcome of constitutional review.",
    "caseDisposition": "Drop: Supreme Court disposition.",
    "caseDispositionUnusual": "Drop: final disposition details.",
    "partyWinning": "Drop: outcome being predicted, excluded from input features.",
    "precedentAlteration": "Drop: precedent changes after ruling.",
    "voteUnclear": "Drop: final vote information.",
    "decisionDirection": "Drop: direction of decision.",
    "decisionDirectionDissent": "Drop: dissenting opinion direction.",
    "authorityDecision1": "Drop: authority cited in ruling.",
    "authorityDecision2": "Drop: additional authority cited in ruling.",
    "majOpinWriter": "Drop: majority opinion author.",
    "majOpinAssigner": "Drop: majority opinion assigner.",
    "splitVote": "Drop: final vote structure.",
    "majVotes": "Drop: number of majority votes.",
    "minVotes": "Drop: number of minority votes.",

    #other dropped columns
    "petitionerState": "Drop: high percentage of missing values.",
    "respondentState": "Drop: high percentage of missing values.",
    "caseSourceState": "Drop: high percentage of missing values.",
    "caseOriginState": "Drop: high percentage of missing values.",
    "adminActionState": "Drop: high percentage of missing values.",
    "lawMinor": "Drop: mostly missing free-text data.",
    "lawType": "Drop: coded using the final opinion.",
    "lawSupp": "Drop: coded using the final opinion.",
    "chief": "Drop: redundant with naturalCourt.",

    #converted columns
    "dateArgument": "Convert: extract argument month.",
    "dateRearg": "Convert: create reargument indicator.",

    #retained columns
    "term": "Retain: term of the case.",
    "naturalCourt": "Retain: Court membership period.",
    "petitioner": "Retain: petitioner party.",
    "respondent": "Retain: respondent party.",
    "jurisdiction": "Retain: jurisdiction of the case.",
    "adminAction": "Retain: administrative action taken.",
    "threeJudgeFdc": "Retain: three-judge panel indicator.",
    "caseOrigin": "Retain: origin of the case.",
    "caseSource": "Retain: source of the case.",
    "lcDisagreement": "Retain: lower court disagreement indicator.",
    "certReason": "Retain: reason for certiorari.",
    "lcDisposition": "Retain: lower court disposition.",
    "lcDispositionDirection": "Retain: direction of lower court disposition.",
    "issue": "Retain: issue of the case.",
    "issueArea": "Retain: area of law for the case.",
}

def validate_columns(columns_or_dataframe):
    """
    Check that every original CSV column is classified exactly once and has a documented reason.
    """
    if hasattr(columns_or_dataframe, 'columns'):
        columns = list(columns_or_dataframe.columns)
    else:
        columns = list(columns_or_dataframe)
    groups = [
        ID_COLUMNS,
        POST_DECISION_COLUMNS,
        OTHER_DROP_COLUMNS,
        CONVERT_COLUMNS,
        CATEGORICAL_FEATURES,
        NUMERIC_FEATURES,
    ]
    classified = [
        name
        for group in groups
        for name in group
    ]

    duplicates = sorted({name for name in classified if classified.count(name) > 1})
    raw_duplicates = sorted({name for name in columns if columns.count(name) > 1})
    missing = sorted(set(columns) - set(classified))
    unclassified = sorted(set(classified) - set(columns))
    reasons_missing = sorted(set(columns) - set(COLUMN_REASONS))
    reasons_extra = sorted(set(COLUMN_REASONS) - set(columns))
    invalid_leakage_columns = sorted(set(LEAKAGE_REVIEW_COLUMNS) - set(RAW_KEEP_COLUMNS))
    errors = []
    checks = [
        ("Columns classified more than once", duplicates),
        ("Duplicate CSV headers", raw_duplicates),
        ("Columns missing from classification", missing),
        ("Columns classified but not present in data", unclassified),
        ("Columns missing documented reason", reasons_missing),
        ("Columns with documented reason but not present in data", reasons_extra),
        ("Leakage review columns not retained", invalid_leakage_columns),
    ]

    for label, values in checks:
        if values:
            errors.append(f"{label}: {values}")
    if errors:
        raise ValueError("; ".join(errors))
    return True