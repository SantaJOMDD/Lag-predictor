import joblib
import time
import pandas as pd
from ping3 import ping

# load model
model = joblib.load("model/lag_model.pkl")

# feature names (IMPORTANT)
features = ['bandwidth', 'throughput', 'congestion', 'packet_loss', 'latency', 'jitter']

def get_live_data():
    latency = ping("8.8.8.8")

    if latency is None:
        latency = 1000
    else:
        latency = latency * 1000

    bandwidth = 5
    throughput = 3
    congestion = 0.2
    packet_loss = 0 if latency < 100 else 1
    jitter = latency * 0.1

    return [bandwidth, throughput, congestion, packet_loss, latency, jitter]


while True:
    try:
        values = get_live_data()

        # ✅ convert to DataFrame with column names
        data = pd.DataFrame([values], columns=features)

        prediction = model.predict(data)[0]

        latency = values[4]

        if prediction == 1:
            print(f"🔴 Lag Incoming! | Ping: {latency:.2f} ms")
        else:
            print(f"🟢 Stable | Ping: {latency:.2f} ms")

        time.sleep(2)

    except KeyboardInterrupt:
        print("\nStopped")
        break