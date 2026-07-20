console.log("SCRIPT LOADED");
const API_BASE_URL = "https://insurance-fraud-api-5pvm.onrender.com";

// Required fields exactly matching FastAPI ClaimInput
const requiredFields = [
    "Age",
    "Deductible",
    "Sex",
    "AccidentArea",
    "PoliceReportFiled",
    "WitnessPresent",
    "Fault",
    "AgentType",
    "VehiclePrice",
    "PastNumberOfClaims",
    "AgeOfVehicle",
    "NumberOfSuppliments",
    "Make",
    "MaritalStatus",
    "PolicyType",
    "VehicleCategory",
    "AddressChange_Claim",
    "BasePolicy"
];

// Map existing HTML IDs to backend field names
const idMap = {
    policyType: "PolicyType",
    basePolicy: "BasePolicy",
    deductible: "Deductible",
    age: "Age",
    sex: "Sex",
    maritalStatus: "MaritalStatus",
    pastClaims: "PastNumberOfClaims",
    make: "Make",
    vehicleCategory: "VehicleCategory",
    vehiclePrice: "VehiclePrice",
    ageOfVehicle: "AgeOfVehicle",
    accidentArea: "AccidentArea",
    fault: "Fault",
    policeReport: "PoliceReportFiled",
    witness: "WitnessPresent",
    agentType: "AgentType",
    addressChange: "AddressChange_Claim",
    supplements: "NumberOfSuppliments"
};

document.addEventListener("DOMContentLoaded", () => {
    const predictBtn = document.getElementById("predictBtn");
    const predictText = document.getElementById("predictText");
    const form = document.querySelector(".prediction-form");
    const formView = document.getElementById("formView");
    const reportView = document.getElementById("reportView");

    console.log("DOM Loaded");
    if (!predictBtn || !form) return;

    // Premium hover effect
    document.querySelectorAll(".form-card").forEach(card => {
        card.addEventListener("mouseenter", () => {
            card.style.transform = "translateY(-2px)";
            card.style.boxShadow = "var(--shadow-lg)";
        });

        card.addEventListener("mouseleave", () => {
            card.style.transform = "translateY(0)";
            card.style.boxShadow = "var(--shadow-md)";
        });
    });

    // Reset field border when changed
    document.querySelectorAll("input, select").forEach(input => {
        input.addEventListener("change", () => {
            input.style.borderColor = "var(--borders)";
        });
    });

    predictBtn.addEventListener("click", async (e) => {
        console.log("Predict button clicked");
        e.preventDefault();

        const payload = {};
        let isValid = true;
        let firstInvalidElement = null;
        
        function showError(element, message) {
            alert(message);
            element.focus();
            element.style.borderColor = "red";
        }

        // Collect values
        requiredFields.forEach(backendKey => {
            const htmlId =
                Object.keys(idMap).find(key => idMap[key] === backendKey) ||
                backendKey;

            const element = document.getElementById(htmlId);

            if (!element) {
                console.error(`Missing HTML element with ID: ${htmlId}`);
                isValid = false;
                return;
            }

            const value = element.value?.trim();
            
            // Age validation
            if (backendKey === "Age") {
                const age = parseInt(value);
                if (isNaN(age) || age < 18 || age > 100) {
                    showError(element, "Age must be between 18 and 100.");
                    isValid = false;
                    if (!firstInvalidElement) firstInvalidElement = element;
                    return;
                }
            }

            // Deductible validation
            if (backendKey === "Deductible") {
                const deductible = parseInt(value);
                if (![300, 400, 500, 700].includes(deductible)) {
                    showError(element, "Deductible must be one of these values: 300, 400, 500, or 700.");
                    isValid = false;
                    if (!firstInvalidElement) firstInvalidElement = element;
                    return;
                }
            }

            if (!value) {
                isValid = false;
                element.style.borderColor = "var(--warning)";
                if (!firstInvalidElement) firstInvalidElement = element;
                return;
            }

            element.style.borderColor = "var(--borders)";

            // Numeric fields
            if (
                [
                    "WeekOfMonth",
                    "WeekOfMonthClaimed",
                    "Age",
                    "Deductible",
                    "DriverRating",
                    "Year"
                ].includes(backendKey)
            ) {
                payload[backendKey] = parseInt(value, 10);
            } else {
                payload[backendKey] = value;
            }
        });

        if (!isValid) {
            if (firstInvalidElement) {
                firstInvalidElement.scrollIntoView({ behavior: "smooth", block: "center" });
                alert("Please correct errors and fill all required fields.");
            }
            return;
        }
        
        // Loading state
        predictBtn.disabled = true;
        predictText.textContent = "Analyzing Claim...";
        
        const currentIcon = document.getElementById("predictIcon");
        if (currentIcon) {
            const loadingIcon = document.createElement("i");
            loadingIcon.id = "predictIcon";
            loadingIcon.setAttribute("data-lucide", "loader-2");
            loadingIcon.classList.add("spin-animation");
            currentIcon.parentNode.replaceChild(loadingIcon, currentIcon);
        }
        
        if (window.lucide) window.lucide.createIcons();

        try {
            // Hardcoded required values missing from the form but needed by backend
            payload.WeekOfMonth = 1;
            payload.WeekOfMonthClaimed = 1;
            payload.DriverRating = 2;
            payload.Year = 1994;
            payload.AgeOfPolicyHolder = "31 to 35";
            payload.NumberOfCars = "1 vehicle";
            payload.Month = "Jan";
            payload.DayOfWeek = "Monday";
            payload.DayOfWeekClaimed = "Monday";
            payload.MonthClaimed = "Jan";
            payload.Days_Policy_Accident = "more than 30";
            payload.Days_Policy_Claim = "more than 30";
            
            console.log("Payload:", payload);
            const response = await fetch(`${API_BASE_URL}/predict`, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify(payload)
            });

            const data = await response.json();
            console.log("Backend Response:", data);

            if (!response.ok) {
                const message = data.detail || "Prediction failed";
                throw new Error(message);
            }

            // --- BUILD DYNAMIC REPORT DASHBOARD --- //
            const isFraud = data.prediction === "Fraud";
            const fraudScore = (data.probability * 100).toFixed(1);
            
            let shapHtml = "";
            let maxShap = 0;
            if (data.shap_explanation && data.shap_explanation.length > 0) {
                maxShap = Math.max(...data.shap_explanation.map(item => Math.abs(item.shap_value)));
                
                data.shap_explanation.forEach(item => {
                    const isPositive = item.shap_value > 0;
                    const percent = Math.min((Math.abs(item.shap_value) / maxShap) * 100, 100);
                    const desc = isPositive 
                        ? "This factor drove the model's suspicion of fraud higher." 
                        : "This factor aligns strongly with safe, genuine claims.";
                    
                    shapHtml += `
                        <div class="shap-item">
                            <div class="shap-header">
                                <span>${item.feature} <span style="color: var(--text-secondary); font-weight: 400; font-size: 0.85rem;">(${item.value})</span></span>
                                <span style="color: ${isPositive ? 'var(--danger)' : 'var(--success)'}">${isPositive ? '+' : ''}${item.shap_value.toFixed(2)}</span>
                            </div>
                            <div class="shap-bar-container">
                                <div class="shap-bar ${isPositive ? 'up' : 'down'}" style="width: 0%;" data-target-width="${percent}%"></div>
                            </div>
                            <div class="shap-desc">${desc}</div>
                        </div>
                    `;
                });
            }

            const explanationLines = (data.explanation || []).map(line => `<div>${line}</div>`).join('');

            const reportHTML = `
                <div class="report-topbar">
                    <button id="backBtn" class="btn-back">
                        <i data-lucide="arrow-left"></i> Back to Edit Details
                    </button>
                </div>
                
                <div class="report-dashboard">
                    <!-- Summary Area -->
                    <div class="report-summary-cards">
                        <div class="summary-card ${isFraud ? 'fraud' : 'genuine'}">
                            <div class="summary-icon ${isFraud ? 'danger' : 'success'}">
                                <i data-lucide="${isFraud ? 'alert-triangle' : 'check-circle-2'}"></i>
                            </div>
                            <div class="summary-value">${data.prediction}</div>
                            <div class="summary-label">Prediction Result</div>
                        </div>
                        <div class="summary-card">
                            <div class="summary-icon primary">
                                <i data-lucide="activity"></i>
                            </div>
                            <div class="summary-value">${fraudScore}%</div>
                            <div class="summary-label">Fraud Probability</div>
                        </div>
                        <div class="summary-card">
                            <div class="summary-icon primary">
                                <i data-lucide="shield"></i>
                            </div>
                            <div class="summary-value">${isFraud ? 'High' : 'Low'}</div>
                            <div class="summary-label">Risk Level</div>
                        </div>
                        <div class="summary-card">
                            <div class="summary-icon primary">
                                <i data-lucide="bar-chart-2"></i>
                            </div>
                            <div class="summary-value">${data.confidence || 'High'}</div>
                            <div class="summary-label">Confidence Score</div>
                        </div>
                    </div>
                    
                    <!-- Recommendations -->
                    <div class="report-section">
                        <div class="report-section-header">
                            <i data-lucide="info"></i>
                            <h3>Recommended Actions</h3>
                        </div>
                        <div class="rec-box">
                            <strong>Recommendation:</strong> ${isFraud ? 'Escalate to Special Investigations Unit (SIU) immediately.' : 'Proceed with standard automated claims processing.'}
                        </div>
                    </div>

                    <!-- AI Insights -->
                    <div class="report-section">
                        <div class="report-section-header">
                            <i data-lucide="brain-circuit"></i>
                            <h3>Model Insights & Key Factors</h3>
                        </div>
                        <div class="rec-box" style="margin-bottom: 24px;">
                            ${explanationLines}
                        </div>
                        
                        <div class="shap-container">
                            ${shapHtml}
                        </div>
                    </div>
                </div>
            `;

            // Swap view
            reportView.innerHTML = reportHTML;
            formView.classList.remove("view-active");
            reportView.classList.add("view-active");
            
            // Render icons in report
            if (window.lucide) {
                window.lucide.createIcons();
            }

            // Animate SHAP bars after rendering
            setTimeout(() => {
                const bars = reportView.querySelectorAll('.shap-bar');
                bars.forEach(bar => {
                    bar.style.width = bar.getAttribute('data-target-width');
                });
            }, 100);

            // Back button listener
            document.getElementById("backBtn").addEventListener("click", () => {
                reportView.classList.remove("view-active");
                formView.classList.add("view-active");
            });

            // Scroll to top of section
            document.querySelector(".prediction-section").scrollIntoView({ behavior: "smooth", block: "start" });

        } catch (error) {
            console.error(error);
            alert(`Failed to analyze claim:\n\n${error.message}\n\nMake sure the backend is running on ${API_BASE_URL}`);
        } finally {
            // Revert predict button state
            predictBtn.disabled = false;
            predictText.textContent = "Generate Prediction";
            
            const currentIcon = document.getElementById("predictIcon");
            if (currentIcon) {
                const refreshIcon = document.createElement("i");
                refreshIcon.id = "predictIcon";
                refreshIcon.setAttribute("data-lucide", "activity");
                currentIcon.parentNode.replaceChild(refreshIcon, currentIcon);
            }
            if (window.lucide) window.lucide.createIcons();
        }
    });
});