from preprocessing import preprocess_input

dummy_input = {
    "Age": 35,
    "WeekOfMonth": 2,
    "Deductible": 400,
    "DriverRating": 3,
    "Month": "Jan",
    "AccidentArea": "Urban",
    "Sex": "Male",
    "Fault": "Policy Holder"
}

result = preprocess_input(dummy_input)

print(result.T[result.T[0] != 0])