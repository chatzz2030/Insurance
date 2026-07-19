console.log("SCRIPT LOADED");
const API_BASE_URL = "http://127.0.0.1:8000";

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
    const predictIcon = document.getElementById("predictIcon");
    const predictText = document.getElementById("predictText");

    const stateWaiting = document.getElementById("stateWaiting");
    const stateFraud = document.getElementById("stateFraud");
    const stateGenuine = document.getElementById("stateGenuine");
    const stateExplanation = document.getElementById("stateExplanation");

    const form = document.querySelector(".prediction-form");

    console.log("DOM Loaded");
    console.log(predictBtn);
    console.log(form);
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
            // Deductible validation
            // Age validation
            if (backendKey === "Age") {
                const age = parseInt(value);

                if (isNaN(age) || age < 18 || age > 100) {
                    showError(
                        element,
                        "Age must be between 18 and 100."
                    );

                    isValid = false;

                    if (!firstInvalidElement) {
                        firstInvalidElement = element;
                    }

                    return;
                }
            }

            // Deductible validation
            if (backendKey === "Deductible") {
                const deductible = parseInt(value);

                if (![300, 400, 500, 700].includes(deductible)) {
                    showError(
                        element,
                        "Deductible must be one of these values: 300, 400, 500, or 700."
                    );

                    isValid = false;

                    if (!firstInvalidElement) {
                        firstInvalidElement = element;
                    }

                    return;
                }
            }

            if (!value) {
                isValid = false;
                element.style.borderColor = "var(--warning)";

                if (!firstInvalidElement) {
                    firstInvalidElement = element;
                }

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

        stateWaiting?.classList.add("hidden");
        stateFraud?.classList.add("hidden");
        stateGenuine?.classList.add("hidden");
        stateExplanation?.classList.add("hidden");

        document.querySelectorAll(".meter-fill").forEach(meter => {
            meter.style.width = "0%";
        });

        try {

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

            // Backend returns:
            // { prediction, probability, confidence }
            const isFraud = data.prediction === "Fraud";
            const targetState = isFraud ? stateFraud : stateGenuine;

            targetState?.classList.remove("hidden");

            const fraudScore = (data.probability * 100).toFixed(1) + "%";

            // Update score labels
            targetState?.querySelectorAll(".meter-label span").forEach(span => {
                if (span.textContent.includes("%")) {
                    span.textContent = fraudScore;
                }
            });

            // Update confidence
            targetState?.querySelectorAll(".metric-value").forEach(el => {
                if (el.textContent.includes("%")) {
                    el.textContent = data.confidence;
                }
            });

            // Render AI Decision Explanation
            stateExplanation?.classList.remove("hidden");

            const explanationText = document.getElementById("explanationText");
            if (explanationText && data.explanation) {
                explanationText.innerHTML = data.explanation.map(line => `<div>${line}</div>`).join('');
            }

            const positiveImpacts = document.getElementById("positiveImpacts");
            const negativeImpacts = document.getElementById("negativeImpacts");
            const posContainer = document.getElementById("positiveFactorsContainer");
            const negContainer = document.getElementById("negativeFactorsContainer");
            
            if (positiveImpacts) positiveImpacts.innerHTML = "";
            if (negativeImpacts) negativeImpacts.innerHTML = "";
            posContainer?.classList.add("hidden");
            negContainer?.classList.add("hidden");

            if (data.shap_explanation && data.shap_explanation.length > 0) {
                const maxShap = Math.max(...data.shap_explanation.map(item => Math.abs(item.shap_value)));
                
                data.shap_explanation.forEach(item => {
                    const isPositive = item.shap_value > 0;
                    if (isPositive) posContainer?.classList.remove("hidden");
                    if (!isPositive) negContainer?.classList.remove("hidden");
                    
                    const percent = Math.min((Math.abs(item.shap_value) / maxShap) * 100, 100);
                    
                    const div = document.createElement("div");
                    div.className = "shap-item";
                    
                    const header = document.createElement("div");
                    header.className = "shap-header";
                    header.innerHTML = `
                        <span>${item.feature} <span style="color: var(--text-secondary); font-weight: 400; font-size: 0.85rem;">(${item.value})</span></span>
                        <span style="color: ${isPositive ? 'var(--danger)' : 'var(--success)'}">${isPositive ? '+' : ''}${item.shap_value.toFixed(2)}</span>
                    `;
                    
                    const barContainer = document.createElement("div");
                    barContainer.className = "shap-bar-container";
                    const bar = document.createElement("div");
                    bar.className = `shap-bar ${isPositive ? 'up' : 'down'}`;
                    bar.style.width = "0%"; // start at 0 for animation
                    
                    // Trigger animation
                    setTimeout(() => {
                        bar.style.width = `${percent}%`;
                    }, 100);
                    
                    barContainer.appendChild(bar);
                    
                    const desc = document.createElement("div");
                    desc.className = "shap-desc";
                    desc.textContent = isPositive 
                        ? "This factor drove the model's suspicion of fraud higher." 
                        : "This factor aligns strongly with safe, genuine claims.";
                        
                    div.appendChild(header);
                    div.appendChild(barContainer);
                    div.appendChild(desc);
                    
                    if (isPositive) {
                        positiveImpacts?.appendChild(div);
                    } else {
                        negativeImpacts?.appendChild(div);
                    }
                });
            }

            // Re-initialize lucide icons for dynamically added elements (like trending-up)
            if (window.lucide) {
                window.lucide.createIcons();
            }

            // Animate meter
            setTimeout(() => {
                const meterFill = targetState?.querySelector(".meter-fill");

                if (meterFill) {
                    meterFill.style.width = fraudScore;
                }
            }, 100);

            // Scroll on mobile
            if (window.innerWidth < 1024) {
                document
                    .querySelector(".results-panel")
                    ?.scrollIntoView({ behavior: "smooth", block: "start" });
            }
        } catch (error) {
            console.error(error);

            alert(
                `Failed to analyze claim:\n\n${error.message}\n\nMake sure the backend is running on ${API_BASE_URL}`
            );

            stateWaiting?.classList.remove("hidden");
        } finally {
            // Revert button state
            predictBtn.disabled = false;
            predictText.textContent = "Generate Again";
            
            const currentIcon = document.getElementById("predictIcon");
            if (currentIcon) {
                const refreshIcon = document.createElement("i");
                refreshIcon.id = "predictIcon";
                refreshIcon.setAttribute("data-lucide", "refresh-cw");
                currentIcon.parentNode.replaceChild(refreshIcon, currentIcon);
            }
            if (window.lucide) window.lucide.createIcons();
        }
    });
});