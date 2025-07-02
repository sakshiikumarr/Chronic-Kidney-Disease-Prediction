import numpy as np
import pickle

# Load the trained Gradient Boosting model
with open("gradient_boosting_model.pkl", "rb") as f:
    model = pickle.load(f)

# Prediction function
def predict_ckd(input_data):
    """
    Accepts a list of 24 float/int values corresponding to features in correct order.
    Returns: 'CKD Detected' or 'No CKD Detected'
    """
    try:
        # Convert to NumPy array and reshape
        data = np.array(input_data).reshape(1, -1)

        # Predict with model
        prediction = model.predict(data)[0]

        if prediction == 1:
            return "✅ Chronic Kidney Disease Detected"
        else:
            return "❌ No Chronic Kidney Disease Detected"
    except Exception as e:
        return f"Error in prediction: {e}"
