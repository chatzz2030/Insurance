# ============================================================
# utils.py
# Shared validation helpers for the FastAPI backend.
# ============================================================

from mappings import (
    BINARY_ENCODING_MAP,
    ORDINAL_ENCODING_MAP,
    ONE_HOT_COLS,
    VEHICLE_PRICE_MAP,
    PAST_NUMBER_OF_CLAIMS_MAP,
    AGE_OF_POLICY_HOLDER_MAP,
    NUMBER_OF_CARS_MAP,
    AGE_OF_VEHICLE_MAP,
    NUMBER_OF_SUPPLIMENTS_MAP,
)

# ---------------------------------------------------------------------------
# Valid values for every field that the API accepts.
# Used to generate clear validation error messages.
# ---------------------------------------------------------------------------

VALID_VALUES: dict[str, list] = {
    # Binary encoded columns
    "Sex": list(BINARY_ENCODING_MAP["Sex"].keys()),
    "AccidentArea": list(BINARY_ENCODING_MAP["AccidentArea"].keys()),
    "PoliceReportFiled": list(BINARY_ENCODING_MAP["PoliceReportFiled"].keys()),
    "WitnessPresent": list(BINARY_ENCODING_MAP["WitnessPresent"].keys()),
    "Fault": list(BINARY_ENCODING_MAP["Fault"].keys()),
    "AgentType": list(BINARY_ENCODING_MAP["AgentType"].keys()),

    # Ordinal encoded columns
    "VehiclePrice": list(VEHICLE_PRICE_MAP.keys()),
    "PastNumberOfClaims": list(PAST_NUMBER_OF_CLAIMS_MAP.keys()),
    "AgeOfPolicyHolder": list(AGE_OF_POLICY_HOLDER_MAP.keys()),
    "NumberOfCars": list(NUMBER_OF_CARS_MAP.keys()),
    "AgeOfVehicle": list(AGE_OF_VEHICLE_MAP.keys()),
    "NumberOfSuppliments": list(NUMBER_OF_SUPPLIMENTS_MAP.keys()),

    # One-hot encoded columns (valid raw string values)
    "Month": [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ],
    "DayOfWeek": [
        "Monday", "Tuesday", "Wednesday",
        "Thursday", "Friday", "Saturday", "Sunday",
    ],
    "Make": [
        "Honda", "Toyota", "Ford", "Mazda", "Chevrolet", "Pontiac",
        "Accura", "Dodge", "Mercury", "Jaguar", "Nisson", "VW",
        "Saab", "Saturn", "Porche", "BMW", "Mecedes", "Ferrari", "Lexus",
    ],
    "DayOfWeekClaimed": [
        "Monday", "Tuesday", "Wednesday",
        "Thursday", "Friday", "Saturday", "Sunday",
    ],
    "MonthClaimed": [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ],
    "MaritalStatus": ["Single", "Married", "Widow", "Divorced"],
    "PolicyType": [
        "Sport - Liability", "Sport - Collision", "Sport - All Perils",
        "Sedan - Liability", "Sedan - Collision", "Sedan - All Perils",
        "Utility - Liability", "Utility - Collision", "Utility - All Perils",
    ],
    "VehicleCategory": ["Sport", "Utility", "Sedan"],
    "Days_Policy_Accident": [
        "none", "1 to 7", "8 to 15", "15 to 30", "more than 30",
    ],
    "Days_Policy_Claim": [
        "none", "8 to 15", "15 to 30", "more than 30",
    ],
    "AddressChange_Claim": [
        "no change", "under 6 months", "1 year",
        "2 to 3 years", "4 to 8 years",
    ],
    "BasePolicy": ["Collision", "Liability", "All Perils"],
}


def validate_categorical_field(field_name: str, value: str) -> None:
    """
    Raise a ValueError if *value* is not in the known valid set for
    *field_name*.

    Parameters
    ----------
    field_name : str
    value      : str

    Raises
    ------
    ValueError
        With a human-readable message listing accepted values.
    """
    valid = VALID_VALUES.get(field_name)
    if valid is None:
        return  # no validation rule → skip
    if value not in valid:
        raise ValueError(
            f"Invalid value '{value}' for field '{field_name}'. "
            f"Accepted values: {valid}"
        )


def validate_numeric_range(field_name: str, value, min_val=None, max_val=None) -> None:
    """
    Raise a ValueError if *value* is outside [min_val, max_val].

    Parameters
    ----------
    field_name : str
    value      : numeric
    min_val    : optional lower bound (inclusive)
    max_val    : optional upper bound (inclusive)

    Raises
    ------
    ValueError
    """
    if min_val is not None and value < min_val:
        raise ValueError(
            f"Field '{field_name}' must be >= {min_val}, got {value}."
        )
    if max_val is not None and value > max_val:
        raise ValueError(
            f"Field '{field_name}' must be <= {max_val}, got {value}."
        )
