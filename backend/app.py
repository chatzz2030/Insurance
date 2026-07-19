# ============================================================
# app.py
# FastAPI application for the Insurance Fraud Detection API.
#
# Startup: model, scaler, and feature_names are loaded once
#          via model_loader.py (module-level imports).
#
# Endpoint: POST /predict
#   - Accepts JSON with all claim fields
#   - Returns: prediction, probability, confidence
# ============================================================
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Optional

# Import model artifacts at startup (loaded once in model_loader)
import model_loader  # noqa: F401  — triggers module-level load

from predict import run_prediction
from utils import validate_categorical_field, validate_numeric_range

# ---------------------------------------------------------------------------
# App definition
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Insurance Fraud Detection API",
    description=(
        "Predicts whether an insurance claim is fraudulent using "
        "a trained XGBoost classifier."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Request Schema
# ---------------------------------------------------------------------------

class ClaimInput(BaseModel):
    """All fields required by the model's feature set."""

    # Numeric fields
    WeekOfMonth: int = Field(..., ge=1, le=5, description="Week of the month (1-5)")
    WeekOfMonthClaimed: int = Field(..., ge=1, le=5, description="Week of month claimed (1-5)")
    Age: int = Field(..., ge=1, le=100, description="Claimant age (1-100)")
    Deductible: int = Field(..., ge=300, le=700, description="Deductible amount (300/400/500/700)")
    DriverRating: int = Field(..., ge=1, le=4, description="Driver rating (1-4)")
    Year: int = Field(..., ge=1994, le=1996, description="Policy year (1994-1996)")

    # Binary encoded columns (string input)
    Sex: str = Field(..., description="Female or Male")
    AccidentArea: str = Field(..., description="Rural or Urban")
    PoliceReportFiled: str = Field(..., description="No or Yes")
    WitnessPresent: str = Field(..., description="No or Yes")
    Fault: str = Field(..., description="Policy Holder or Third Party")
    AgentType: str = Field(..., description="External or Internal")

    # Ordinal encoded columns (string input)
    VehiclePrice: str = Field(..., description="Vehicle price range")
    PastNumberOfClaims: str = Field(..., description="Past number of claims")
    AgeOfPolicyHolder: str = Field(..., description="Age bracket of policy holder")
    NumberOfCars: str = Field(..., description="Number of cars")
    AgeOfVehicle: str = Field(..., description="Age of vehicle")
    NumberOfSuppliments: str = Field(..., description="Number of supplements")

    # One-hot encoded columns (string input)
    Month: str = Field(..., description="Month of accident (Jan-Dec)")
    DayOfWeek: str = Field(..., description="Day of accident (Monday-Sunday)")
    Make: str = Field(..., description="Car make/brand")
    DayOfWeekClaimed: str = Field(..., description="Day claim was filed (Monday-Sunday)")
    MonthClaimed: str = Field(..., description="Month claim was filed (Jan-Dec)")
    MaritalStatus: str = Field(..., description="Single/Married/Widow/Divorced")
    PolicyType: str = Field(..., description="Vehicle type + base policy combination")
    VehicleCategory: str = Field(..., description="Sport/Utility/Sedan")
    Days_Policy_Accident: str = Field(..., description="Days between policy and accident")
    Days_Policy_Claim: str = Field(..., description="Days between policy and claim")
    AddressChange_Claim: str = Field(..., description="Address change before claim")
    BasePolicy: str = Field(..., description="Collision/Liability/All Perils")

    # Validators for categorical fields
    @field_validator(
        "Sex", "AccidentArea", "PoliceReportFiled", "WitnessPresent",
        "Fault", "AgentType", "VehiclePrice", "PastNumberOfClaims",
        "AgeOfPolicyHolder", "NumberOfCars", "AgeOfVehicle", "NumberOfSuppliments",
        "Month", "DayOfWeek", "Make", "DayOfWeekClaimed", "MonthClaimed",
        "MaritalStatus", "PolicyType", "VehicleCategory",
        "Days_Policy_Accident", "Days_Policy_Claim", "AddressChange_Claim",
        "BasePolicy",
        mode="before",
    )
    @classmethod
    def validate_string_fields(cls, v, info):
        """Validate string fields against known valid values."""
        if not isinstance(v, str):
            raise ValueError(f"Field '{info.field_name}' must be a string.")
        v = v.strip()
        if not v:
            raise ValueError(f"Field '{info.field_name}' cannot be empty.")
        validate_categorical_field(info.field_name, v)
        return v


# ---------------------------------------------------------------------------
# Response Schema
# ---------------------------------------------------------------------------

class ShapExplanation(BaseModel):
    feature: str
    value: str
    shap_value: float
    impact: str

class PredictionResponse(BaseModel):
    """Prediction result returned by POST /predict."""
    prediction: str = Field(..., description="'Fraud' or 'Not Fraud'")
    probability: float = Field(..., description="Fraud probability (0.0 – 1.0)")
    confidence: str = Field(..., description="Fraud probability as percentage, e.g. '91%'")
    shap_explanation: list[ShapExplanation] = Field(..., description="SHAP feature impacts")
    explanation: list[str] = Field(..., description="Human-readable explanation lines")


# ---------------------------------------------------------------------------
# Global error handlers
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """Catch-all for unexpected server errors."""
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"},
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", tags=["Health"])
def root():
    """Health check endpoint."""
    return {"status": "ok", "message": "Insurance Fraud Detection API is running."}


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Prediction"],
    summary="Predict insurance fraud",
    description=(
        "Submit a claim record and receive a fraud prediction, "
        "probability score, and confidence level."
    ),
)
def predict(claim: ClaimInput):
    """
    Run the fraud prediction pipeline on the submitted claim.

    - **prediction**: 'Fraud' or 'Not Fraud'
    - **probability**: float in [0, 1] — fraud probability
    - **confidence**: human-readable percentage, e.g. '91%'
    """
    
    try:
        # Convert Pydantic model to plain dict for the pipeline
        data = claim.model_dump()

        # Auto-fill default values for fields hidden in the frontend
        data.update({
            "WeekOfMonth": data.get("WeekOfMonth", 1),
            "WeekOfMonthClaimed": data.get("WeekOfMonthClaimed", 1),
            "DriverRating": data.get("DriverRating", 2),
            "Year": data.get("Year", 1994),
            "AgeOfPolicyHolder": data.get("AgeOfPolicyHolder", "31 to 35"),
            "NumberOfCars": data.get("NumberOfCars", "1 vehicle"),
            "Month": data.get("Month", "Jan"),
            "DayOfWeek": data.get("DayOfWeek", "Monday"),
            "DayOfWeekClaimed": data.get("DayOfWeekClaimed", "Monday"),
            "MonthClaimed": data.get("MonthClaimed", "Jan"),
            "Days_Policy_Accident": data.get("Days_Policy_Accident", "more than 30"),
            "Days_Policy_Claim": data.get("Days_Policy_Claim", "more than 30"),
        })

        result = run_prediction(data)
        return result

    except KeyError as e:
        raise HTTPException(
            status_code=422,
            detail=f"Encoding error — unknown value for field: {e}",
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}",
        )
