# Insurance Fraud Detection Backend

This is the FastAPI backend for the Insurance Fraud Detection machine learning project. It provides an API endpoint to predict whether an insurance claim is fraudulent using a pre-trained XGBoost model.

## Prerequisites

- Python 3.9+
- The `models/` directory must contain the following artifacts:
  - `insurance_fraud_xgboost.pkl` (The trained XGBoost model)
  - `scaler.pkl` (StandardScaler)
  - `feature_names.pkl` (List of feature names expected by the model)

## Setup Instructions

1. **Open a terminal** in the root of the project (`insurance` folder).
2. **Activate the virtual environment**:
   - On Windows:
     ```powershell
     .\venv\Scripts\activate
     ```
   - On Mac/Linux:
     ```bash
     source venv/bin/activate
     ```
3. **Install the dependencies** (if not already installed):
   ```powershell
   pip install -r backend/requirements.txt
   ```

## Running the Server

To start the FastAPI server, you need to run `uvicorn` from inside the `backend` directory.

1. **Navigate to the `backend` folder**:
   ```powershell
   cd backend
   ```
2. **Start the Uvicorn server**:
   ```powershell
   uvicorn app:app --reload
   ```

*(Alternatively, if you want to run it from the root directory without activating the environment first, you can use: `cd backend` followed by `..\venv\Scripts\uvicorn app:app --reload`)*

## API Endpoints

Once the server is running, you can access the following URLs in your browser:

- **Swagger UI (Interactive API Docs):** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **Health Check:** [http://127.0.0.1:8000/](http://127.0.0.1:8000/)

### Predict Endpoint (`POST /predict`)

Submit a JSON payload containing the claim details to receive a fraud prediction.

**Example Request:**
```json
{
  "WeekOfMonth": 2,
  "WeekOfMonthClaimed": 1,
  "Age": 35,
  "Deductible": 400,
  "DriverRating": 3,
  "Year": 1994,
  "Sex": "Male",
  "AccidentArea": "Urban",
  "PoliceReportFiled": "No",
  "WitnessPresent": "No",
  "Fault": "Policy Holder",
  "AgentType": "External",
  "VehiclePrice": "30000 to 39000",
  "PastNumberOfClaims": "none",
  "AgeOfPolicyHolder": "31 to 35",
  "NumberOfCars": "1 vehicle",
  "AgeOfVehicle": "3 years",
  "NumberOfSuppliments": "none",
  "Month": "Jan",
  "DayOfWeek": "Monday",
  "Make": "Honda",
  "DayOfWeekClaimed": "Wednesday",
  "MonthClaimed": "Jan",
  "MaritalStatus": "Single",
  "PolicyType": "Sedan - Collision",
  "VehicleCategory": "Sedan",
  "Days_Policy_Accident": "more than 30",
  "Days_Policy_Claim": "more than 30",
  "AddressChange_Claim": "no change",
  "BasePolicy": "Collision"
}
```

**Example Response:**
```json
{
  "prediction": "Fraud",
  "probability": 0.5792,
  "confidence": "58%"
}
```

---

## Backend Project Structure & Architecture

To understand how the backend was built and how the data flows from a raw API request to a machine learning prediction, here is a detailed breakdown of the internal architecture. The codebase is heavily modularized for clean separation of concerns.

### Directory Structure

```text
backend/
├── app.py                 # Main FastAPI application and API routing
├── predict.py             # Prediction pipeline logic
├── preprocessing.py       # Data transformation and scaling
├── mappings.py            # Encoding dictionaries derived from the notebook
├── model_loader.py        # Centralized loading of ML artifacts (pickles)
├── utils.py               # Input validation logic
└── requirements.txt       # Project dependencies
```

### File-by-File Breakdown

#### 1. `mappings.py`
This file contains the core logic for converting human-readable categorical values into numerical formats that the XGBoost model understands. The encoding logic was manually mapped from the original Jupyter Notebook (`Insurance_Fraud.ipynb`) into reusable Python dictionaries.
- **`BINARY_ENCODING_MAP`**: Handles simple binary variables (e.g. Yes/No, Male/Female) mapped alphabetically to 0/1.
- **`ORDINAL_ENCODING_MAP`**: Handles categorical variables that have a strict sequential order (e.g., `AgeOfVehicle`, `VehiclePrice`).
- **`ONE_HOT_COLS`**: Defines which remaining categorical columns need to be expanded into dummy variables via one-hot encoding.

#### 2. `preprocessing.py`
This is the data engine of the backend. It ensures that the JSON payload received from the API perfectly matches the feature matrix the model was trained on, preventing shape-mismatch errors. 
- **Step 1:** It applies the binary and ordinal encodings from `mappings.py`.
- **Step 2:** It uses pandas to one-hot encode the remaining categorical variables (`pd.get_dummies(drop_first=True)`).
- **Step 3:** It aligns the columns exactly with `feature_names.pkl`, appending missing columns (as zeros) and dropping extraneous ones.
- **Step 4:** Finally, it applies the loaded standard scaler (`scaler.pkl`) to normalize the numerical data before passing it to the model.

#### 3. `model_loader.py`
To prevent the application from experiencing high latency by reading large `.pkl` files from the disk on every single API request, this module loads the XGBoost model, the StandardScaler, and the feature names into memory exactly **once** when the server starts. This singleton pattern guarantees fast inference times.

#### 4. `predict.py`
This file acts as the orchestrator for the inference pipeline. It accepts the validated user input dict, passes it to `preprocessing.py` to get the scaled feature matrix, and then feeds that matrix into the XGBoost model to generate both the discrete class prediction (Fraud / Not Fraud) and the continuous probability score via `predict_proba`.

#### 5. `utils.py`
This module handles data integrity. It stores strictly defined lists of all acceptable categorical values (e.g., specific Car Makes, specific Policy Types) mapping precisely to what the model expects. The FastAPI app uses these helpers to instantly reject malformed API requests with clear, human-readable error messages before they execute the expensive machine learning pipeline.

#### 6. `app.py`
This is the entry point of the FastAPI server. It defines the robust API schema using Pydantic, ensuring strictly typed data boundaries.
- Defines the `ClaimInput` schema, documenting every required field, its data type, and its constraints.
- Exposes the main `/predict` POST endpoint.
- Includes comprehensive error handling to return structured HTTP 422 errors for bad inputs, or HTTP 500 for server-side pipeline failures.

---

## Frontend Setup & Execution

The frontend of this project is designed as a standalone, enterprise-grade static web application, focusing solely on the UI/UX.

### Running the Frontend Manually

Since the frontend consists only of static files (`index.html` and `style.css`), no complex build steps or Node.js server are required to view it. 

1. **Navigate to the frontend folder**:
   ```powershell
   cd frontend
   ```
2. **Open the `index.html` file in your browser**:
   - **Method A (Directly):** Double-click the `index.html` file in your File Explorer.
   - **Method B (Using a local server - Recommended):** If you use VS Code, install the **Live Server** extension, right-click `index.html`, and select "Open with Live Server". Alternatively, use Python's built-in HTTP server:
     ```powershell
     python -m http.server 8080
     ```
     Then, open your browser and go to `http://localhost:8080`.

### Frontend Architecture (Interview Preparation)

If asked about the frontend implementation in an interview, here are key points to cover:

- **Technology Stack:** Pure HTML5 and CSS3 (Vanilla). No frontend frameworks (like React or Angular) or CSS frameworks (like Bootstrap or Tailwind) were used to demonstrate core frontend proficiency and understanding of the DOM/CSS Object Model.
- **Design Philosophy:** The design follows a modern, enterprise SaaS aesthetic (inspired by Apple, Stripe, Microsoft Fluent Design). It uses a clean light theme, avoiding overly saturated "gamer" aesthetics to maintain professionalism for a B2B insurance context.
- **CSS Architecture:** 
  - **CSS Variables (Custom Properties):** Used extensively for theming (colors, typography, spacing) ensuring maintainability and a single source of truth.
  - **Flexbox & CSS Grid:** Used for layout structuring, ensuring 100% pixel-perfect responsiveness across Desktop, Tablet, and Mobile devices.
  - **Glassmorphism & Soft UI:** Implemented using CSS `backdrop-filter` and layered `box-shadow` to create depth and hierarchy without heavy borders.
  - **Hardware-Accelerated Animations:** All transitions and animations use `transform` and `opacity` properties which are delegated to the GPU, avoiding layout repaints and ensuring smooth 60fps performance.
- **Form Design Strategy:** The prediction form is split into multiple distinct cards (Policy, Customer, Vehicle, Claim, Additional) to reduce cognitive load and provide a wizard-like, premium user experience.
## Frontend-Backend Integration & Explainable AI (SHAP)

### How They Are Connected
The frontend (`script.js`) and backend (`app.py`) are fully connected to provide a seamless full-stack experience.

1. **CORS Configuration**: The FastAPI backend is explicitly configured with `CORSMiddleware` to allow cross-origin requests from the frontend running on a different port (`http://localhost:8080`).
2. **Data Collection & Validation**: When a user clicks "Generate Prediction", JavaScript extracts all values from the HTML form. It strictly validates numeric constraints (e.g., Age must be 18-100, Deductible must be standard values) and catches missing inputs before they ever hit the server, preventing FastAPI `422 Unprocessable Entity` errors.
3. **Safe Default Values**: For machine learning features that aren't exposed in the clean frontend form (like `WeekOfMonth` or `DriverRating`), the frontend safely injects fallback default values into the JSON payload before transmitting it.
4. **Fetch API**: The frontend sends an asynchronous `POST` request to `http://127.0.0.1:8000/predict` and awaits the JSON response.
5. **Dynamic UI Rendering**: The backend returns the discrete prediction (Fraud/Genuine), continuous probability, and SHAP AI explanation impacts. The frontend parses this and instantly populates dynamic glassmorphic cards, colored progress bars, and clear plain-english logic explaining why the AI made its decision.

---

## How to Run Manually

To run the entire application (both frontend and backend) simultaneously, follow these steps:

### Step 1: Start the Backend Server
1. Open a terminal in the root of the project (`insurance` folder).
2. Activate the virtual environment:
   - **Windows:** `.\venv\Scripts\activate`
   - **Mac/Linux:** `source venv/bin/activate`
3. Navigate to the backend directory:
   ```powershell
   cd backend
   ```
4. Start the FastAPI server:
   ```powershell
   uvicorn app:app --reload
   ```
*(Keep this terminal open. The backend is now running at `http://127.0.0.1:8000`)*

### Step 2: Open the Frontend Application
1. Open a **new, separate terminal** in the root of the project.
2. Navigate to the frontend directory:
   ```powershell
   cd frontend
   ```
3. Start a local HTTP server:
   ```powershell
   python -m http.server 8080
   ```
4. Open your web browser and navigate to [http://localhost:8080](http://localhost:8080).

---

## 🚀 Cloud Deployment Guide (Running Online)

To make your application accessible to anyone on the internet, you should deploy the Frontend and Backend separately using modern cloud platforms. Both can be hosted for free.

### Step 1: Deploy the Backend (FastAPI)
The backend requires a Python environment to run the machine learning model. Good free platforms for this are **Render** (render.com), **Railway** (railway.app), or **Heroku**.

1. Create a `requirements.txt` file (already done).
2. Push your code to a GitHub repository.
3. Connect your GitHub repository to **Render** and create a new "Web Service".
4. Set the Root Directory to `backend` (if supported) or adjust the start command.
5. Set the Start Command to:
   ```bash
   uvicorn app:app --host 0.0.0.1 --port $PORT
   ```
6. Once deployed, Render will give you a live URL like: `https://insurance-fraud-api.onrender.com`

### Step 2: Deploy the Frontend (HTML/JS)
Since the frontend is purely static HTML, CSS, and JS, it can be deployed on blazing-fast static hosts like **Vercel** (vercel.com), **Netlify** (netlify.com), or **GitHub Pages**.

1. Connect your GitHub repository to Vercel or Netlify.
2. Set the Root Directory to the `frontend/` folder.
3. Deploy it. They will give you a live URL like: `https://insurance-fraud-app.vercel.app`

### Step 3: Connect Them Together (Important Code Changes)

Once you have your live URLs, you must update the code so the live frontend talks to the live backend (instead of localhost).

1. **Update Frontend (`script.js`)**:
   Change the API URL at the top of the file to your live backend URL:
   ```javascript
   // Change this:
   // const API_BASE_URL = "http://127.0.0.1:8000";
   
   // To this:
   const API_BASE_URL = "https://insurance-fraud-api.onrender.com";
   ```

2. **Update Backend CORS (`app.py`)**:
   For security, FastAPI will block the frontend unless you tell it that your Vercel/Netlify URL is allowed.
   ```python
   app.add_middleware(
       CORSMiddleware,
       # Change this from ["*"] or localhost to your exact frontend live URL:
       allow_origins=["https://insurance-fraud-app.vercel.app"],
       allow_credentials=True,
       allow_methods=["*"],
       allow_headers=["*"],
   )
   ```

Once you commit these two changes and push them to GitHub, your cloud providers will automatically rebuild your app, and your full-stack AI platform will be live on the internet!