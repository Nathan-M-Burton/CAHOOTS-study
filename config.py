"""Central configuration for CAHOOTS study analyses."""
import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_RAW_PATH = os.path.join(BASE_DIR, "data", "raw", "CAD_data_clean.csv")
DATA_PROCESSED_PATH = os.path.join(BASE_DIR, "data", "processed", "cad_cleaned.csv")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")

# Column names
TYPE_COL = "InitialIncidentTypeDescription"
CALL_CREATED_COL = "Call_Created_Time"
DISPATCH_COL = "Unit_Dispatched_Time"
ONSCENE_COL = "Unit_OnScene_Time"
PRIMARY_UNIT_COL = "PrimaryUnitCallSign"
RESPONDING_UNIT_COL = "RespondingUnitCallSign"

# OMD analysis
OM_THRESHOLD = 0.85
OM_LOWER = 1 - OM_THRESHOLD
OM_UPPER = OM_THRESHOLD
START_YEAR = "2016"
END_YEAR = "2021"
DISPATCHED = True
ARRIVED = True
HEATMAP_MODE = "proportions"  # or "counts"

# SDR/PDR analysis
CUTOFF_YEAR = 2017
CUTOFF_MONTH = 1
PARTIAL_YEAR = 2016
PARTIAL_MONTH = 11
TREATMENT_HOURS = (3, 6)  # inclusive range for early_morning indicator
MIN_CALLS = 0
MIN_TOTAL_CALLS = 1000
P_VALUE = 0.05
ROLLING_WINDOW = 2

# Call type formatting
CALLTYPE_RENAME_MAP = {
    "Assist Fire Department": "Assist FD",
    "Police Officer Hold": "Officer Hold",
    "Officer Safety Info": "Safety Info",
}

# Treemap
TOP_N = 14
GROUP_OTHERS = True

# Composition analysis
KEY_TYPES = ["TRANSPORT", "ASSIST PUBLIC", "DISPUTE", "SUSPICIOUS SUBJECT", "ASSAULT"]
KEY_TYPE_LABELS = ["Transport", "Assist Public", "Dispute", "Susp Subject", "Assault"]

# Figures
FIGURE_DPI = 300
