# ============================================================
# mappings.py
# Ordinal/label encoding maps that exactly match the
# training notebook (Insurance_Fraud.ipynb).
# ============================================================

# --- Binary LabelEncoder results (fit_transform order) ---
# LabelEncoder sorts classes alphabetically, so:
#   AccidentArea : Rural=0, Urban=1
#   AgentType    : External=0, Internal=1
#   Fault        : Policy Holder=0, Third Party=1
#   PoliceReportFiled : No=0, Yes=1
#   Sex          : Female=0, Male=1
#   WitnessPresent : No=0, Yes=1

BINARY_COLS = [
    "Sex",
    "AccidentArea",
    "PoliceReportFiled",
    "WitnessPresent",
    "Fault",
    "AgentType",
]

# Maps for binary LabelEncoded columns (alphabetical order → 0/1)
SEX_MAP = {"Female": 0, "Male": 1}

ACCIDENT_AREA_MAP = {"Rural": 0, "Urban": 1}

POLICE_REPORT_FILED_MAP = {"No": 0, "Yes": 1}

WITNESS_PRESENT_MAP = {"No": 0, "Yes": 1}

FAULT_MAP = {"Policy Holder": 0, "Third Party": 1}

AGENT_TYPE_MAP = {"External": 0, "Internal": 1}

# Convenience dict: column name → encoding map
BINARY_ENCODING_MAP = {
    "Sex": SEX_MAP,
    "AccidentArea": ACCIDENT_AREA_MAP,
    "PoliceReportFiled": POLICE_REPORT_FILED_MAP,
    "WitnessPresent": WITNESS_PRESENT_MAP,
    "Fault": FAULT_MAP,
    "AgentType": AGENT_TYPE_MAP,
}

# --- Ordinal encoding maps (from notebook cells 71-82) ---

VEHICLE_PRICE_MAP = {
    "less than 20000": 0,
    "20000 to 29000": 1,
    "30000 to 39000": 2,
    "40000 to 59000": 3,
    "60000 to 69000": 4,
    "more than 69000": 5,
}

PAST_NUMBER_OF_CLAIMS_MAP = {
    "none": 0,
    "1": 1,
    "2 to 4": 2,
    "more than 4": 3,
}

AGE_OF_POLICY_HOLDER_MAP = {
    "16 to 17": 0,
    "18 to 20": 1,
    "21 to 25": 2,
    "26 to 30": 3,
    "31 to 35": 4,
    "36 to 40": 5,
    "41 to 50": 6,
    "51 to 65": 7,
    "over 65": 8,
}

NUMBER_OF_CARS_MAP = {
    "1 vehicle": 1,
    "2 vehicles": 2,
    "3 to 4": 3,
    "5 to 8": 4,
    "more than 8": 5,
}

AGE_OF_VEHICLE_MAP = {
    "new": 0,
    "2 years": 1,
    "3 years": 2,
    "4 years": 3,
    "5 years": 4,
    "6 years": 5,
    "7 years": 6,
    "more than 7": 7,
}

NUMBER_OF_SUPPLIMENTS_MAP = {
    "none": 0,
    "1 to 2": 1,
    "3 to 5": 2,
    "more than 5": 3,
}

# Convenience dict: column name → ordinal encoding map
ORDINAL_ENCODING_MAP = {
    "VehiclePrice": VEHICLE_PRICE_MAP,
    "PastNumberOfClaims": PAST_NUMBER_OF_CLAIMS_MAP,
    "AgeOfPolicyHolder": AGE_OF_POLICY_HOLDER_MAP,
    "NumberOfCars": NUMBER_OF_CARS_MAP,
    "AgeOfVehicle": AGE_OF_VEHICLE_MAP,
    "NumberOfSuppliments": NUMBER_OF_SUPPLIMENTS_MAP,
}

# --- One-hot encoded columns (pd.get_dummies with drop_first=True) ---
# The dropped (reference) category for each column is the first
# alphabetically / by appearance that pandas drops.
# Remaining dummies are exactly the columns in feature_names.pkl.

ONE_HOT_COLS = [
    "Month",
    "DayOfWeek",
    "Make",
    "DayOfWeekClaimed",
    "MonthClaimed",
    "MaritalStatus",
    "PolicyType",
    "VehicleCategory",
    "Days_Policy_Accident",
    "Days_Policy_Claim",
    "AddressChange_Claim",
    "BasePolicy",
]