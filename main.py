from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
import joblib

# load data
df = pd.read_csv("data/network_dataset_labeled.csv")

# 🔥 SHIFT TARGET TO PREDICT THE FUTURE
# Target is now: Will there be an anomaly 3 timesteps from now?
df['future_anomaly'] = df['anomaly'].shift(-3)
df = df.dropna()

features = ['bandwidth', 'throughput', 'congestion', 'packet_loss', 'latency', 'jitter']
X = df[features]
y = df['future_anomaly']

# 🔥 APPLY SMOTE
smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X, y)

# split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2)

# model
model = RandomForestClassifier()

# train
model.fit(X_train, y_train)

# predict
y_pred = model.predict(X_test)

# report
print("Future Lag Prediction Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

joblib.dump(model, "model/lag_model.pkl")
print("Future Lag Predictor Model saved successfully!")