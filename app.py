import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from sklearn.preprocessing import MinMaxScaler

# Load and preprocess data
data = pd.read_csv("https://raw.githubusercontent.com/jbrownlee/Datasets/master/shampoo.csv")
data.columns = ["Month", "Sales"]

# Fix Date Format
def fix_date_format(date_str):
    month, year = date_str.split("-")
    year = "19" + year if int(year) > 30 else "20" + year
    return f"{year}-{month}"

data["Month"] = data["Month"].apply(fix_date_format)
data["Month"] = pd.to_datetime(data["Month"], format='%Y-%m')
data.set_index("Month", inplace=True)

# Scale Sales Data
scaler = MinMaxScaler()
data["Sales"] = scaler.fit_transform(data["Sales"].values.reshape(-1, 1))

seq_length = 12

# LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

# Load Model
model = LSTMModel(input_size=1, hidden_size=50, num_layers=2, output_size=1)
model.load_state_dict(torch.load("demand_forecasting_lstm.pth"))
model.eval()

# Initialize Flask App
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    input_data = np.array(data['sales']).reshape(1, seq_length, 1)
    input_tensor = torch.tensor(input_data, dtype=torch.float32)
    
    with torch.no_grad():
        prediction = model(input_tensor).item()
    
    prediction = scaler.inverse_transform([[prediction]])[0][0]
    return jsonify({'predicted_sales': prediction})

if __name__ == '__main__':
    app.run(debug=True)
