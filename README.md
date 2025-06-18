# 🪙 Bitcoin Price Prediction Using LSTM (PyTorch)

This project trains a deep learning model to predict Bitcoin's **closing price** using its **high, low, and open prices** as inputs.

### 🔍 What It Does

- Loads historical Bitcoin price data from `coin_Bitcoin.csv`
- Uses the "High", "Low", and "Open" columns to predict the "Close" price
- Scales the data for better neural network performance
- Trains a **5-layer LSTM model** with fully connected layers to make predictions
- Evaluates the model using **R² score** on the test set
- Prints training loss and model accuracy after training

### 🧠 Model Highlights

- **Input:** 3 features (High, Low, Open)
- **Architecture:** LSTM → Dense → ReLU → Dense → Output
- **Output:** 1 value — the predicted closing price
- **Batch size:** 64
- **Epochs:** 10
- **Loss function:** Mean Squared Error (MSE)
- **Optimizer:** Adam

### 📊 Result

After training, the model makes predictions on the test set and compares them to real prices using inverse scaling. It reports the **R² score**, showing how well the model performs.

---

This is a basic implementation to demonstrate time series forecasting with LSTM in PyTorch using financial data.
