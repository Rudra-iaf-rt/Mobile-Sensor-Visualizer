import json
import numpy as np
from kafka import KafkaConsumer
import psycopg2
import torch
import torch.nn as nn

# ---------------- CNN MODEL ---------------- #
class VehicleCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(6, 32, 3),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, 3),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 30, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )

    def forward(self, x):
        return self.fc(self.conv(x))

model = VehicleCNN()
model.eval()

# ---------------- KAFKA ---------------- #
consumer = KafkaConsumer(
    "sensors",
    bootstrap_servers="kafka:9092",
    value_deserializer=lambda m: json.loads(m.decode("utf-8"))
)

# ---------------- DATABASE ---------------- #
conn = psycopg2.connect(
    host="timescaledb",
    database="sensorDetector",
    user="postgres",
    password="rudra2006"
)
cursor = conn.cursor()

# Create table
cursor.execute("""
CREATE TABLE IF NOT EXISTS sensor_data (
    time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    acc_x FLOAT, acc_y FLOAT, acc_z FLOAT,
    gyro_x FLOAT, gyro_y FLOAT, gyro_z FLOAT,
    prediction INT
);
""")
conn.commit()

# ---------------- BUFFER ---------------- #
buffer = []
WINDOW = 128

# ---------------- STREAM LOOP ---------------- #
for msg in consumer:

    data = msg.value

    try:
        acc = data["acc"]
        gyro = data["gyro"]

        values = acc + gyro
        buffer.append(values)

    except:
        continue

    if len(buffer) >= WINDOW:

        window = np.array(buffer[-WINDOW:])
        x = torch.tensor(window).float().unsqueeze(0).permute(0,2,1)

        with torch.no_grad():
            pred = model(x)
            prediction = int(pred.argmax(1))

        print("Prediction:", prediction)

        # Insert latest row into DB
        cursor.execute("""
            INSERT INTO sensor_data (
                acc_x, acc_y, acc_z,
                gyro_x, gyro_y, gyro_z,
                prediction
            ) VALUES (%s,%s,%s,%s,%s,%s,%s)
        """, (*values, prediction))

        conn.commit()