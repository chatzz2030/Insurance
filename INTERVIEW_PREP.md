# 🎙️ FraudShield AI — Interview Preparation Guide

> Everything you need to revise before your interview. Covers the project pitch, step-by-step what you built, the dataset, all files, the `.pkl` model files, and every type of question an interviewer might ask.

---

## 🚀 PROJECT PITCH (Your Intro — Memorize This)

> *"I built an end-to-end AI-powered Insurance Fraud Detection web application called **FraudShield AI**. The system takes a live insurance claim submitted through a web form, sends it to a FastAPI backend, runs it through a pre-trained **XGBoost classifier**, and instantly predicts whether the claim is fraudulent or genuine — along with a fraud probability score and an **explainable AI breakdown** using SHAP values that shows exactly which factors drove the model's decision. The entire pipeline — from data preprocessing in Jupyter Notebook, model training, saving trained artifacts as `.pkl` files, building a REST API, to deploying a premium responsive frontend — was built by me from scratch."*

**Tech Stack in one line:**
`Python + XGBoost + SHAP + FastAPI + Pydantic + Joblib + Vanilla HTML/CSS/JS`

**What makes it unique:**
- Not just a prediction — gives a **human-readable explanation** (Explainable AI / XAI)
- **SHAP values** are visualized with animated bar charts on the UI
- Two-server architecture: FastAPI backend (port 8000) + Python HTTP server for frontend (port 8080)
- Input validation at **two layers**: Pydantic schema in API + JS validation in the browser

---

## 📂 PROJECT STRUCTURE

```
insurance/
├── fraud_oracle.csv              ← Dataset (raw data, ~3.6 MB)
├── notebooks/
│   └── Insurance_Fraud.ipynb    ← Full EDA + model training notebook
├── models/
│   ├── insurance_fraud_xgboost.pkl   ← Trained XGBoost model (~764 KB)
│   ├── scaler.pkl                    ← StandardScaler artifact (~5 KB)
│   └── feature_names.pkl             ← Ordered list of features (~2 KB)
├── backend/
│   ├── app.py               ← FastAPI app + API endpoints
│   ├── predict.py           ← Core prediction + SHAP logic
│   ├── preprocessing.py     ← Transforms raw input into model-ready features
│   ├── mappings.py          ← All encoding maps (binary, ordinal, one-hot)
│   ├── model_loader.py      ← Loads .pkl files at startup (once)
│   ├── utils.py             ← Validation helpers (valid values per field)
│   └── requirements.txt     ← All Python dependencies
├── frontend/
│   ├── index.html           ← Full UI (form + results panel)
│   ├── style.css            ← Premium dark/glassmorphism design (~20 KB)
│   └── script.js            ← API calls, SHAP visualization, form logic
└── run_app.bat              ← One-click startup script (Windows)
```

---

## 📊 THE DATASET — `fraud_oracle.csv`

**Source:** Oracle Insurance Fraud dataset (publicly available)
**Size:** ~3.6 MB
**Target column:** `FraudFound_P` → 0 = Not Fraud, 1 = Fraud

### All 33 Columns:

| Column | Type | Description |
|---|---|---|
| `PolicyNumber` | ID | Dropped — not a feature |
| `RepNumber` | ID | Dropped — not a feature |
| `Month` | Categorical | Month of the accident (Jan–Dec) |
| `WeekOfMonth` | Numeric | Week within the month (1–5) |
| `DayOfWeek` | Categorical | Day of accident (Monday–Sunday) |
| `Make` | Categorical | Car brand (Honda, Toyota, BMW, etc.) |
| `AccidentArea` | Binary | Rural or Urban |
| `DayOfWeekClaimed` | Categorical | Day the claim was filed |
| `MonthClaimed` | Categorical | Month the claim was filed |
| `WeekOfMonthClaimed` | Numeric | Week of month claimed (1–5) |
| `Sex` | Binary | Female or Male |
| `MaritalStatus` | Categorical | Single / Married / Widow / Divorced |
| `Age` | Numeric | Claimant's age |
| `Fault` | Binary | Policy Holder or Third Party |
| `PolicyType` | Categorical | Vehicle type + coverage (e.g. Sedan - Collision) |
| `VehicleCategory` | Categorical | Sport / Utility / Sedan |
| `VehiclePrice` | Ordinal | Price range of vehicle |
| `FraudFound_P` | **Target** | **0 = Genuine, 1 = Fraud** |
| `Deductible` | Numeric | 300 / 400 / 500 / 700 |
| `DriverRating` | Numeric | 1–4 |
| `Days_Policy_Accident` | Categorical | Days between policy start and accident |
| `Days_Policy_Claim` | Categorical | Days between policy start and claim |
| `PastNumberOfClaims` | Ordinal | none / 1 / 2 to 4 / more than 4 |
| `AgeOfVehicle` | Ordinal | new / 2 years / ... / more than 7 |
| `AgeOfPolicyHolder` | Ordinal | Age brackets (16-17 → over 65) |
| `PoliceReportFiled` | Binary | No or Yes |
| `WitnessPresent` | Binary | No or Yes |
| `AgentType` | Binary | External or Internal |
| `NumberOfSuppliments` | Ordinal | Extra damage claims added |
| `AddressChange_Claim` | Categorical | How recently claimant moved |
| `NumberOfCars` | Ordinal | 1 vehicle → more than 8 |
| `Year` | Numeric | Policy year (1994–1996) |
| `BasePolicy` | Categorical | Collision / Liability / All Perils |

### Data Cleaning Done in Notebook:
- **Dropped** `PolicyNumber` and `RepNumber` (non-predictive IDs)
- **Age == 0** → replaced with **median age**
- **MonthClaimed == '0'** → replaced with **mode**
- **DayOfWeekClaimed == '0'** → replaced with **mode**
- **Class imbalance** noted: fraud cases are a minority (~15–25% of records)

---

## 🔢 ENCODING STRATEGY (3 Different Methods)

### 1. Binary Encoding (Label Encoding — alphabetical order)
Applied to 6 columns with exactly 2 values:
```
Sex              : Female=0, Male=1
AccidentArea     : Rural=0, Urban=1
PoliceReportFiled: No=0, Yes=1
WitnessPresent   : No=0, Yes=1
Fault            : Policy Holder=0, Third Party=1
AgentType        : External=0, Internal=1
```
**Why?** LabelEncoder assigns 0/1 alphabetically. Replicating this exactly prevents train/serve skew.

### 2. Ordinal Encoding (Manual mapping — natural order)
Applied to 6 columns with a meaningful rank order:
```
VehiclePrice        : less than 20000=0 → more than 69000=5
PastNumberOfClaims  : none=0, 1=1, 2 to 4=2, more than 4=3
AgeOfPolicyHolder   : 16 to 17=0 → over 65=8
NumberOfCars        : 1 vehicle=1 → more than 8=5
AgeOfVehicle        : new=0 → more than 7=7
NumberOfSuppliments : none=0 → more than 5=3
```
**Why?** These have a meaningful natural order — encoding numerically preserves that order for the tree model.

### 3. One-Hot Encoding (`pd.get_dummies` with `drop_first=True`)
Applied to 12 remaining categorical columns:
```
Month, DayOfWeek, Make, DayOfWeekClaimed, MonthClaimed,
MaritalStatus, PolicyType, VehicleCategory,
Days_Policy_Accident, Days_Policy_Claim,
AddressChange_Claim, BasePolicy
```
**Why `drop_first=True`?** Removes one dummy per column to avoid multicollinearity (the dummy variable trap).

---

## 🧠 THE MODEL — XGBoost

**Algorithm:** XGBoost (Extreme Gradient Boosting)
**Type:** Ensemble of Gradient Boosted Decision Trees

**Why XGBoost?**
- Handles mixed data types (numeric + encoded categoricals) natively
- Built-in regularization (L1/L2) reduces overfitting
- Feature importance is built-in
- Works extremely well on tabular data
- Industry standard for fraud detection problems

**Model accuracy on test set: ~94.6%**

### The Notebook Pipeline:
```
Raw CSV Data
  ↓ Drop PolicyNumber, RepNumber
  ↓ Impute: Age median, MonthClaimed/DayOfWeekClaimed mode
  ↓ Binary Label Encode (6 cols)
  ↓ Ordinal Encode (6 cols)
  ↓ One-Hot Encode: get_dummies drop_first=True (12 cols)
  ↓ Train/Test Split (e.g. 80/20)
  ↓ StandardScaler (fit on train only, transform both)
  ↓ XGBoost Classifier (fit on train)
  ↓ Save: model.pkl + scaler.pkl + feature_names.pkl
```

---

## 📦 THE `.pkl` FILES — `models/` folder

These 3 files are the **serialized artifacts** saved from the Jupyter notebook using `joblib.dump()`.

| File | Size | What's inside |
|---|---|---|
| `insurance_fraud_xgboost.pkl` | ~764 KB | The trained XGBoost model — all trees, weights, hyperparameters |
| `scaler.pkl` | ~5 KB | StandardScaler fitted on training data — stores mean & std for every feature |
| `feature_names.pkl` | ~2 KB | Exact ordered list of feature column names after all encoding |

### Why `feature_names.pkl` is Critical
After one-hot encoding, `pd.get_dummies` creates many new columns (e.g., `Make_Honda`, `Make_Toyota`, etc.). The **exact column order** must match what the model was trained on. `feature_names.pkl` stores this list so the preprocessing pipeline can:
1. **Add** any missing columns (set to 0 — e.g., `Make_BMW` when user entered `Honda`)
2. **Drop** any extra columns
3. **Reorder** columns to exactly match training

### Why `joblib` and not `pickle`?
- `joblib` is optimized for large NumPy arrays (used inside sklearn/XGBoost models)
- Faster serialization/deserialization for ML artifacts
- Recommended by scikit-learn documentation

### How they're loaded — `model_loader.py`:
```python
import joblib, os
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR  = os.path.join(BASE_DIR, "models")

model         = joblib.load(os.path.join(MODEL_DIR, "insurance_fraud_xgboost.pkl"))
scaler        = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
feature_names = joblib.load(os.path.join(MODEL_DIR, "feature_names.pkl"))
```
**Loaded once at startup** (module-level import) — not per-request. This is the efficient, production-correct approach.

---

## ⚙️ BACKEND — `backend/` folder (file by file)

### `app.py` — FastAPI Application
- Creates the FastAPI app with **CORS enabled** (allows any origin — for dev)
- Defines `ClaimInput` — a **Pydantic BaseModel** with all 28 fields
  - Numeric fields use `Field(..., ge=..., le=...)` for range validation
  - String fields use `@field_validator` which calls `validate_categorical_field()`
- Defines `PredictionResponse` model for the response schema
- **POST `/predict`** endpoint:
  1. Receives and auto-validates JSON via Pydantic
  2. Converts to dict with `claim.model_dump()`
  3. Auto-fills 12 hidden fields with sensible defaults
  4. Calls `run_prediction(data)` → returns structured result
  5. FastAPI serializes and returns JSON
- **GET `/`** → health check endpoint
- Global `@app.exception_handler(Exception)` → catches unhandled errors → 500 JSON

### `preprocessing.py` — Feature Engineering Pipeline
Converts raw user input dict → scaled DataFrame, step by step:
1. Shallow copy of input dict
2. Apply `BINARY_ENCODING_MAP` → convert strings (Male/Female) to 0/1
3. Apply `ORDINAL_ENCODING_MAP` → convert strings (new/2 years/...) to integers
4. Convert to `pd.DataFrame`
5. Apply `pd.get_dummies(drop_first=True)` for one-hot columns only
6. **Column alignment:**
   - Add any missing column from `feature_names` → set value = 0
   - Drop extra columns not in `feature_names`
   - Reorder to exact `feature_names` order
7. Apply `scaler.transform()` → StandardScaler normalization
8. Return scaled DataFrame (1 row)

### `predict.py` — Prediction + SHAP Logic
1. Call `preprocess_input()` → get scaled feature matrix
2. `model.predict(processed)` → returns `[0]` or `[1]`
3. `model.predict_proba(processed)[0][1]` → fraud probability float (0.0–1.0)
4. `explainer.shap_values(processed)[0]` → SHAP value per feature column
5. **Aggregate SHAP**: loop through encoded columns, match back to original feature name (e.g., `Make_Honda` → `Make`), sum SHAP values per original feature
6. Sort: if Fraud → highest positive SHAP first; if Genuine → most negative SHAP first
7. Build `explanation` list — human-readable top-5 reasons
8. Return: `{ prediction, probability, confidence, shap_explanation, explanation }`

### `mappings.py` — Encoding Maps
**Single source of truth** for all encoding rules. Exactly mirrors what was done in the training notebook. Three sections:
- `BINARY_ENCODING_MAP` — dict of `{column: {value: int}}`
- `ORDINAL_ENCODING_MAP` — dict of `{column: {value: int}}`
- `ONE_HOT_COLS` — list of column names for `pd.get_dummies`

### `utils.py` — Validation Helpers
- `VALID_VALUES` dict → every string field maps to its list of accepted values
- `validate_categorical_field(field_name, value)` → raises `ValueError` with helpful message if value not in accepted set
- `validate_numeric_range(field_name, value, min, max)` → raises `ValueError` for out-of-range numbers
- Both functions called by Pydantic validators in `app.py`

### `model_loader.py` — Artifact Loader
- Module-level (global) imports → **executes once when Uvicorn starts**
- Makes `model`, `scaler`, `feature_names` importable by other modules
- Uses `os.path` to build absolute paths → works regardless of working directory

---

## 🌐 FRONTEND — `frontend/` folder (file by file)

### `index.html` — The Single-Page Web App
A complete single-page application with these sections:
1. **Navbar** — Logo (FraudShield AI + shield icon), navigation links, mobile hamburger
2. **Hero Section** — "Intelligent Insurance Claim Risk Detection" title, subtitle, CTA, hero image
3. **Statistics Bar** — 4 stat cards: `94.6% Accuracy`, `<1s Prediction`, `XGBoost`, `XAI`
4. **Security Section** — Bank-grade security, SOC2 compliance, End-to-end encryption
5. **Features Grid** — 6 feature cards: AI Detection, Machine Learning, Real-Time, Explainable AI, FastAPI Backend, Secure Processing
6. **Predict Section** — Main form (5 cards) + Results sidebar (3 dynamic states)
7. **Footer** — Logo, quick links, contact

**Form cards (5 sections covering 18 visible fields):**

| Card | Fields |
|---|---|
| Policy Information | PolicyType, BasePolicy, Deductible |
| Customer Information | Age, Sex, MaritalStatus, PastNumberOfClaims |
| Vehicle Information | Make, VehicleCategory, VehiclePrice, AgeOfVehicle |
| Claim Information | AccidentArea, Fault, PoliceReportFiled, WitnessPresent |
| Additional Information | AgentType, AddressChange_Claim, NumberOfSuppliments |

**Results sidebar (3 dynamic states toggled by JS):**
- `#stateWaiting` — shown initially ("Fill out claim details")
- `#stateFraud` — shown if Fraud (red, alert-triangle icon, fraud score meter, "Escalate to SIU")
- `#stateGenuine` — shown if Not Fraud (green, check icon, low risk)
- `#stateExplanation` — always shown after any prediction (SHAP bar charts)

**External libraries (CDN, no install needed):**
- **Google Fonts** — Inter (body) + Plus Jakarta Sans (headings)
- **Lucide Icons** — modern SVG icon library (`lucide.createIcons()`)

### `style.css` — Premium Design (~20 KB)
- **CSS custom properties** (variables) for entire design system (`--primary`, `--accent`, `--danger`, `--success`, etc.)
- **Dark glassmorphism** aesthetic with background blobs
- **Animated blobs** using CSS keyframe animations
- **Hover micro-animations** on form cards (`translateY(-2px)` + shadow)
- **Animated progress meter** bars (width transitions)
- **SHAP bar charts** — red `.up` class for positive, green `.down` class for negative
- **Fully responsive** — mobile hamburger menu, stacked form on small screens

### `script.js` — Frontend Logic (~14 KB)
Key elements:
- **`requiredFields`** — array of 18 backend field names the user must fill
- **`idMap`** — maps HTML element IDs (camelCase) to backend field names (PascalCase/snake_case)
- **Click handler on `#predictBtn`:**
  1. Collects all 18 field values using `idMap`
  2. Validates: Age (18–100), Deductible (must be 300/400/500/700), no empty fields
  3. Auto-fills 12 hidden backend fields (WeekOfMonth=1, Year=1994, etc.)
  4. `fetch()` POST to `http://127.0.0.1:8000/predict` with JSON payload
  5. Renders result: shows correct state card (fraud/genuine), updates meter width
  6. Renders SHAP: creates animated bar divs for each factor, red/green colored
  7. Shows loading spinner (Lucide `loader-2` + spin CSS) while waiting
  8. Handles errors with `alert()` showing the backend error message

---

## 🔄 HOW TO RUN THE APP

**One click:** Double-click `run_app.bat`

**What the bat file does internally:**
```batch
start "FastAPI Backend" cmd /k "cd backend && ..\venv\Scripts\uvicorn.exe app:app --host 127.0.0.1 --port 8000 --reload"
start "Frontend Web Server" cmd /k "cd frontend && ..\venv\Scripts\python.exe -m http.server 8080"
```
Opens 2 separate CMD windows — one for each server.

**Access:**
| URL | Purpose |
|---|---|
| `http://localhost:8080` | Frontend web app |
| `http://127.0.0.1:8000` | Backend health check |
| `http://127.0.0.1:8000/docs` | **Auto-generated Swagger API docs** |
| `http://127.0.0.1:8000/redoc` | ReDoc API docs |

---

## 🎯 STEP-BY-STEP: WHAT YOU DID (Tell This in Interviews)

### Phase 1: Data Exploration (Jupyter Notebook)
1. Loaded `fraud_oracle.csv` into Pandas
2. Explored shape, dtypes, missing values, class distribution (`FraudFound_P`)
3. Identified target column and feature types
4. Dropped non-predictive IDs: `PolicyNumber`, `RepNumber`
5. Fixed data quality issues: `Age==0` → median, `'0'` values → mode

### Phase 2: Feature Engineering
6. Applied **LabelEncoder** to 6 binary columns (alphabetical 0/1)
7. Applied **manual ordinal encoding** to 6 ordinal columns
8. Applied **`pd.get_dummies(drop_first=True)`** to 12 categorical columns
9. Applied **StandardScaler** — fit on train set only, transform both splits

### Phase 3: Model Training & Evaluation
10. Train/Test split (e.g. 80/20)
11. Trained **XGBoostClassifier**
12. Evaluated: accuracy ~94.6%, precision, recall, F1-score, confusion matrix
13. Saved 3 artifacts with `joblib.dump()`:
    - `insurance_fraud_xgboost.pkl` — the model
    - `scaler.pkl` — the fitted scaler
    - `feature_names.pkl` — the exact column list

### Phase 4: Backend API (FastAPI)
14. Created `model_loader.py` — loads .pkl at startup
15. Created `mappings.py` — hardcoded all encoding maps matching notebook
16. Created `preprocessing.py` — replicates notebook pipeline for single input
17. Created `predict.py` — runs model + SHAP explanation
18. Created `utils.py` — all valid value rules for 28 fields
19. Created `app.py` — FastAPI app, Pydantic schema, `/predict` endpoint

### Phase 5: Frontend
20. Designed `index.html` with 5 form cards + dynamic results sidebar
21. Built `style.css` — glassmorphism dark theme + CSS animations
22. Wrote `script.js` — form validation, `fetch()` API call, SHAP bar chart rendering

### Phase 6: Packaging
23. Created `run_app.bat` for one-click startup
24. Tested the full end-to-end flow: form submit → API → prediction → UI render

---

## ❓ INTERVIEW QUESTIONS & MODEL ANSWERS

---

### 🔵 PROJECT & CONCEPT

**Q: Tell me about your project.**
> FraudShield AI is an end-to-end insurance fraud detection system. Users fill a web form, the data is sent to a FastAPI backend, preprocessed exactly like the training data, and an XGBoost model predicts fraud probability. SHAP values explain which factors drove the decision. I built everything — from the Jupyter notebook to the frontend UI.

**Q: What is the target variable in your dataset?**
> `FraudFound_P` — binary column: 0 = genuine claim, 1 = fraudulent claim.

**Q: Was the dataset balanced?**
> No. Fraud cases are a minority class (roughly 15–25%). This is expected in real fraud data because most claims are genuine. This makes recall (catching actual fraud) more important than raw accuracy.

**Q: How did you handle class imbalance?**
> XGBoost's `scale_pos_weight` parameter weights the minority class. Alternatives include SMOTE (synthetic oversampling) or adjusting the classification threshold. For evaluation, I focused on F1-score and recall on the fraud class rather than just overall accuracy.

---

### 🔵 MACHINE LEARNING

**Q: Why XGBoost?**
> XGBoost builds trees sequentially — each tree corrects the errors of the previous ones (gradient boosting). It has built-in regularization (L1/L2) to prevent overfitting, handles mixed feature types well, and is the industry standard for tabular fraud detection. Random Forest (bagging) builds trees in parallel and averages — generally slightly less accurate on this type of problem. Logistic Regression is linear and can't capture complex feature interactions.

**Q: What is gradient boosting?**
> An ensemble technique where each new model (decision tree) is trained to minimize the residual errors (gradient of the loss function) of the combined previous models. Each tree is added sequentially, shrinking errors iteratively. XGBoost adds regularization terms to the objective function to control tree complexity.

**Q: What is StandardScaler and why was it used?**
> StandardScaler normalizes each feature to mean=0, std=1 using `z = (x - mean) / std`. While tree models don't strictly require scaling, it was part of the training pipeline in the notebook. The key rule: **fit only on training data, then transform both train and test** — to avoid data leakage.

**Q: What is data leakage?**
> When information from the test set is used during training. Example: fitting the scaler on all data before splitting means test set statistics influence training. This makes model performance look better than it actually is in production.

**Q: What metrics did you use for evaluation?**
> Accuracy (~94.6%), Precision, Recall, F1-Score (especially for the fraud class — class 1), and a Confusion Matrix. False Negatives (missed fraud) are the most costly error in fraud detection, so recall on the fraud class is the primary concern.

**Q: What is precision vs recall?**
> - **Precision** = TP / (TP + FP) — of all predicted frauds, how many were actually fraud
> - **Recall** = TP / (TP + FN) — of all actual frauds, how many did the model catch
> High recall is critical in fraud detection — we'd rather flag a genuine claim for review than miss actual fraud.

---

### 🔵 ENCODING

**Q: Why three different encoding methods?**
> Each method suits the data type:
> - **Label/Binary**: 2 categories, no order needed → simple 0/1
> - **Ordinal**: categories have a natural order (vehicle age, price ranges) → integer ranks preserve that order for tree splits
> - **One-Hot**: multiple categories with no order → prevents the model from assuming false ordering

**Q: Why `drop_first=True` in get_dummies?**
> To avoid the dummy variable trap — perfect multicollinearity. If you have 3 categories (A, B, C) you only need 2 dummies. A is implied when both are 0. `drop_first=True` drops the first alphabetical category.

**Q: Why did you hardcode the encoding maps instead of saving encoder objects?**
> The encoding rules are simple, fixed, and well-defined. Hardcoding in `mappings.py` makes the code explicit and verifiable — you can read exactly what encoding is applied. Saving sklearn encoder objects adds complexity for what is essentially a lookup table.

**Q: What is `feature_names.pkl` and why is it needed?**
> After one-hot encoding, `pd.get_dummies` creates columns like `Make_Honda`, `Make_BMW`, etc. The model was trained on a specific ordered set of ~60+ columns. At prediction time, a user submitting `Make=Honda` only produces `Make_Honda=1` — all other make columns must be added as 0 and the whole list must be reordered. `feature_names.pkl` stores this exact ordered column list for alignment.

---

### 🔵 BACKEND / API

**Q: Why FastAPI and not Flask?**
> FastAPI is async-ready (ASGI), auto-generates Swagger/OpenAPI docs at `/docs`, has first-class Pydantic integration for validation, and is significantly faster than Flask (WSGI). For a prediction API, these advantages are meaningful.

**Q: What is Pydantic?**
> A Python data validation library. You define a class inheriting from `BaseModel` and declare fields with types and constraints. FastAPI uses Pydantic to automatically parse, validate, and serialize JSON. Invalid input automatically returns a 422 Unprocessable Entity error with details.

**Q: What does `@field_validator` do?**
> It's a Pydantic v2 decorator for custom field-level validation logic. In this project, it fires for every string field and calls `validate_categorical_field()` which checks the value against a pre-defined accepted list. If invalid, it raises `ValueError` → Pydantic converts it to a 422 HTTP error.

**Q: What is CORS and why did you enable it?**
> CORS (Cross-Origin Resource Sharing) is a browser security mechanism that blocks requests to a different origin (port counts). Since frontend is on `localhost:8080` and backend is on `localhost:8000`, the browser would block the API call without CORS headers. `CORSMiddleware` with `allow_origins=["*"]` allows all origins — fine for development.

**Q: What is Uvicorn?**
> An ASGI (Asynchronous Server Gateway Interface) web server for Python. It runs FastAPI applications. `--reload` enables hot-reloading — the server restarts automatically when code changes.

**Q: Why are some fields hidden from the UI but still sent to the API?**
> To keep the UI user-friendly we hide low-impact fields (WeekOfMonth, Year, DriverRating, etc.) that have obvious defaults. The model still needs all features it was trained on, so they're auto-filled in `app.py` before calling the prediction function.

**Q: What HTTP status codes does your API return?**
> `200` — successful prediction; `422` — validation error (Pydantic caught bad input); `500` — unexpected server error (caught by global exception handler).

---

### 🔵 SHAP / EXPLAINABLE AI

**Q: What is SHAP?**
> SHAP (SHapley Additive exPlanations) explains individual model predictions by assigning each feature a value indicating its contribution. It's based on Shapley values from cooperative game theory — each feature gets credit for the portion of the prediction it "caused." Positive SHAP = pushed prediction toward Fraud; Negative SHAP = pushed toward Genuine.

**Q: What is TreeExplainer?**
> `shap.TreeExplainer` is a fast, exact SHAP implementation specifically for tree-based models like XGBoost, LightGBM, and Random Forest. It's much faster than the general KernelSHAP (which treats the model as a black box) because it exploits the tree structure directly.

**Q: Why initialize `explainer = shap.TreeExplainer(model)` globally?**
> Initializing the TreeExplainer processes the model's tree structure — it's computationally expensive. Creating it once at server startup and reusing it for every prediction request avoids this overhead per-request.

**Q: How do you aggregate SHAP values for one-hot encoded features?**
> After one-hot encoding, `Make` becomes `Make_Honda`, `Make_Toyota`, etc. In `predict.py`, we loop through every processed column name and match it back to the original feature using `col.startswith(f"{key}_")`. We accumulate (sum) all SHAP values for the same original feature name, giving one SHAP value per original feature.

**Q: What does positive vs negative SHAP mean in your project?**
> - **Positive SHAP** → this feature pushed the fraud probability higher for this specific claim
> - **Negative SHAP** → this feature pushed the fraud probability lower (evidence of genuineness)

**Q: How is SHAP shown on the frontend?**
> The backend returns `shap_explanation` (list of `{feature, value, shap_value, impact}`). The JS builds animated horizontal bar charts — red bars labeled "High Risk Factors" for positive SHAP values and green bars labeled "Low Risk Factors" for negative SHAP values. Bars animate from 0% to their percentage (relative to max SHAP) using CSS transitions.

---

### 🔵 FRONTEND

**Q: Why Vanilla JS and not React?**
> This is a single page with one form and a results panel — no complex state management. Vanilla JS keeps it lightweight (no build process, no npm, instant load). It also demonstrates core JavaScript skills without framework dependency.

**Q: How does the form submit to the API?**
> The predict button has a `click` listener. It collects values from all 18 form elements into a `payload` object using `idMap`, validates them, auto-fills 12 hidden fields, then uses the browser's `fetch()` API to POST JSON to the backend. The response is rendered dynamically into the results sidebar.

**Q: What is the `idMap` in script.js?**
> A mapping object from HTML element IDs (camelCase, like `pastClaims`) to backend field names (like `PastNumberOfClaims`). This bridges the gap between UI-friendly HTML IDs and the exact field names the FastAPI endpoint expects.

**Q: How does the SHAP animation work?**
> Each bar div starts with `width: 0%` in CSS. After a 100ms `setTimeout`, the width is set to the calculated percentage. CSS `transition: width 0.6s ease` on the bar class creates the smooth fill animation.

---

### 🔵 GENERAL / ADVANCED

**Q: Difference between `predict()` and `predict_proba()`?**
> `predict()` returns the class label (0 or 1). `predict_proba()` returns probabilities for each class — e.g., `[0.08, 0.92]` means 8% genuine, 92% fraud. We use `predict_proba()[0][1]` to get the fraud probability score displayed to the user.

**Q: If you had more time, what would you improve?**
> - Add a database to log all predictions for auditing
> - Add user authentication (API keys / OAuth2)
> - Tackle class imbalance with SMOTE or cost-sensitive training
> - Use MLflow for experiment tracking during training
> - Deploy to cloud (AWS/GCP) with Docker containers
> - Add model monitoring to detect data drift in production

**Q: What is `joblib` vs `pickle`?**
> `joblib` is optimized for serializing large NumPy arrays — the core of sklearn/XGBoost models. It uses memory mapping and compression more efficiently. Both work, but `joblib` is the scikit-learn/XGBoost recommended approach for ML artifacts.

**Q: How do you prevent train-test skew at prediction time?**
> The encoding maps in `mappings.py` exactly mirror the notebook. The scaler is loaded from `scaler.pkl` — the actual scaler fit on training data — and only `.transform()` (not `.fit()`) is called at inference. `feature_names.pkl` ensures exact column alignment. No re-fitting ever happens in the backend.

**Q: What happens if an unknown Make is submitted?**
> The `validate_categorical_field()` function in `utils.py` is called by the Pydantic validator for every string field. If the submitted make is not in `VALID_VALUES["Make"]`, a `ValueError` is raised with a human-readable message listing all accepted values → Pydantic converts it to a 422 HTTP response. It never reaches the model.

**Q: How would you deploy this to production?**
> 1. Containerize with Docker (Dockerfile for backend)
> 2. Deploy backend to cloud (AWS Lambda, GCP Cloud Run, or EC2)
> 3. Serve frontend via CDN (S3 + CloudFront)
> 4. Add HTTPS and restrict CORS to specific origins
> 5. Add API authentication (API key or JWT)
> 6. Set up logging, monitoring (Prometheus/Grafana or cloud-native tools)

---

## 🔑 KEY NUMBERS TO REMEMBER

| Fact | Value |
|---|---|
| Model Accuracy | ~94.6% |
| Dataset size | ~3.6 MB |
| Binary encoded columns | 6 |
| Ordinal encoded columns | 6 |
| One-hot encoded columns | 12 |
| Total features after encoding | ~60+ columns |
| `.pkl` files | 3 (model, scaler, feature_names) |
| XGBoost model size | ~764 KB |
| Backend port | 8000 |
| Frontend port | 8080 |
| API prediction endpoint | POST `/predict` |
| Visible form fields (UI) | 18 |
| Hidden auto-filled fields | 12 |
| API docs (Swagger) | `http://127.0.0.1:8000/docs` |

---

## 💡 QUICK REVISION BULLETS (Last-minute read)

- **Dataset:** `fraud_oracle.csv` → target: `FraudFound_P` (0 = genuine, 1 = fraud)
- **Dropped:** `PolicyNumber`, `RepNumber`
- **Imputed:** `Age==0` → median; `MonthClaimed/DayOfWeekClaimed == '0'` → mode
- **3 encoding methods:** LabelEncode 6 cols, Ordinal 6 cols, OneHot/dummies 12 cols
- **StandardScaler:** fit on train only, saved as `scaler.pkl`, only transform at inference
- **Model:** XGBoostClassifier, ~94.6% accuracy, saved via `joblib.dump()`
- **3 `.pkl` files:** model weights, scaler mean/std, feature column order list
- **`feature_names.pkl`:** ensures exact column order at inference — critical for one-hot alignment
- **Backend:** FastAPI + Pydantic validation + SHAP explanations per prediction
- **SHAP:** TreeExplainer (global), SHAP values aggregated for one-hot columns
- **Frontend:** Vanilla HTML/CSS/JS, `fetch()` API, animated SHAP bar charts
- **Run:** `run_app.bat` → opens 2 CMD windows → port 8000 + port 8080
- **Swagger docs:** auto-generated at `http://127.0.0.1:8000/docs`

---

*You built every single part of this — own it with confidence in your interview! 🚀*
