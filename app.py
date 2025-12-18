
### Loading the dataset
import pandas as pd
import numpy as np
df=pd.read_csv(r"D:\urban grievance folder\311_Service_Requests_from_2010_to_Present_20251124.csv",nrows=100000)
df

### Cleaning the data
df = df[[ "Created Date", "Agency Name", "Complaint Type", "Descriptor", "Status", "Borough", "Latitude", "Longitude", "Location"]]
df.info()
df.isnull().sum()

df=df.drop_duplicates()
df["Descriptor"] = df["Descriptor"].fillna("Unknown")
new_df=df.dropna(subset=["Latitude","Longitude","Location"]).copy()
new_df.info()

"""### Comparing data before and after removing duplicates on specific columns"""

import matplotlib.pyplot as plt
import seaborn as sns

# Create the KDE plot for the 'Latitude' column of the original DataFrame
sns.kdeplot(df['Latitude'], color='blue', label='Original Data')

# Create the KDE plot for the 'Latitude' column of the empty values DataFrame
sns.kdeplot(new_df['Latitude'], color='red', label='After removing empty val ')

plt.title('Distribution of Latitude Before and After removing empty values')
plt.xlabel('Latitude')
plt.ylabel('Density')
plt.legend() # Show the labels
plt.show()
new_df

new_df
new_df["Created Date"]=pd.to_datetime(new_df["Created Date"])
new_df["Hour"] = new_df["Created Date"].dt.hour
new_df["DayOfWeek"] = new_df["Created Date"].dt.dayofweek
new_df["Month"] = new_df["Created Date"].dt.month
new_df=new_df.drop(columns=["Created Date","Status","Location"])
new_df

### from the above classfication report . we can observe that the many of the labels are having smaples less then 20 . its very difficult for model for learning the patterns with few samples
###
frequency_complaint_type= new_df["Complaint Type"].value_counts()
print(frequency_complaint_type)
rare_samples=frequency_complaint_type[frequency_complaint_type<30].index
new_df["Complaint Type"]=new_df["Complaint Type"].replace(rare_samples,"Other")
new_df
x=new_df.drop(columns=["Complaint Type"])
y=new_df["Complaint Type"]
print(x)
print(y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20)
print(x_train.shape)
print(x_test.shape)
### converting the descriptor in to numerical vector
from sklearn.feature_extraction.text import TfidfVectorizer
tfdif=TfidfVectorizer(ngram_range=(1,2),max_features=5000,stop_words="english")
x_train_tfdif_desc = tfdif.fit_transform(x_train["Descriptor"])
x_test_tfdif_desc=tfdif.transform(x_test["Descriptor"])
# 4) show shapes and example
print("TF-IDF shape (train):", x_train_tfdif_desc.shape)   # (n_train_rows, n_features)
print("TF-IDF shape (test) : ", x_test_tfdif_desc.shape)

# Optional: show top 10 features (words) learned by TF-IDF
feature_names = tfdif.get_feature_names_out()
print("Top 10 TF-IDF features:", feature_names[:10])

##converting agency and the borough columns in to numericals
x_train_cat=pd.get_dummies(x_train[["Agency Name","Borough"]],drop_first=True)
x_test_cat=pd.get_dummies(x_test[["Agency Name","Borough"]],drop_first=True)
x_train_cat,x_test_cat=x_train_cat.align(x_test_cat,join="left",axis=1,fill_value=0)
num_cols = ['Latitude','Longitude','Hour','DayOfWeek','Month']
x_train_num = x_train[num_cols].fillna(0).values    # shape: (n_train, 5)
x_test_num  = x_test[num_cols].fillna(0).values
# Convert TF-IDF sparse matrices to dense arrays (small slice) and combine horizontally
# If memory is fine you can use .toarray(); if not, tell me and I'll give the sparse combine version.
x_train_tfidf_dense = x_train_tfdif_desc.toarray()
x_test_tfidf_dense  = x_test_tfdif_desc.toarray()
#concatinating the columns that are converted in to numericals
x_train_full = np.hstack([x_train_tfidf_dense, x_train_cat.values, x_train_num])
x_test_full  = np.hstack([x_test_tfidf_dense,  x_test_cat.values,  x_test_num])
print("TF-IDF (train) shape:", x_train_tfidf_dense.shape)
print("One-Hot (train) shape:", x_train_cat.shape)
print("Numeric (train) shape:", x_train_num.shape)
print("Final TRAIN shape:", x_train_full.shape)

print("\nTF-IDF (test) shape:", x_test_tfidf_dense.shape)
print("One-Hot (test) shape:", x_test_cat.shape)
print("Numeric (test) shape:",x_test_num.shape)
print("Final TEST shape:", x_test_full.shape)
# small sanity: confirm same number of columns in train/test
print("\nColumns match? ", x_train_full.shape[1] == x_test_full.shape[1])

##model for classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score , classification_report
model=RandomForestClassifier(n_estimators=200,random_state=42,n_jobs=1)
model.fit(x_train_full,y_train)
y_pred=model.predict(x_test_full)
accuracy=accuracy_score(y_test,y_pred)
classification_report=classification_report(y_test,y_pred)
print("Accuracy:"+str(accuracy))
print("classification report:"+str(classification_report))
probs = model.predict_proba(x_test_full)
confidence_scores = probs.max(axis=1)
# app.py — Final version: uses grievance_pipeline.joblib + emergency override + robust priority
# Put this file in the same folder as grievance_pipeline.joblib and templates/index.html

import traceback
from pathlib import Path
from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import numpy as np
import warnings
import re
import sklearn

print("sklearn version (serving):", sklearn.__version__)

# ---------------- CONFIG ----------------
MODEL_FILE = Path(r"D:\urban grievance folder\grievance_pipeline.joblib")   # use absolute path if you prefer
PORT = 8000
# ----------------------------------------

warnings.filterwarnings("ignore", message="X has feature names")
app = Flask(__name__, template_folder="templates")

# -------------- LOAD PIPELINE --------------
model_global = None
model_load_error = None
try:
    if not MODEL_FILE.exists():
        raise FileNotFoundError(f"Pipeline not found at: {MODEL_FILE.resolve()}")
    print("Loading pipeline from:", MODEL_FILE.resolve())
    model_global = joblib.load(str(MODEL_FILE.resolve()))
    print("Loaded object type:", type(model_global))
    if not hasattr(model_global, "named_steps"):
        raise RuntimeError("Loaded object is not a sklearn Pipeline. Save full pipeline as grievance_pipeline.joblib")
    print("Pipeline steps:", list(model_global.named_steps.keys()))
except Exception as e:
    model_global = None
    model_load_error = str(e)
    traceback.print_exc()

# -------------- PRIORITY / EMERGENCY LOGIC --------------
EMERGENCY_WORDS = {"gas", "leak", "smell", "odor", "explosion", "fire", "smoke", "radiation",
                   "chemical", "toxic", "hazard"}
EMERGENCY_PHRASES = {"gas leak", "gas odor", "smell of gas", "explosion",
                     "water main break", "burst pipe", "chemical spill"}
MEDIUM_WORDS = {"water", "sewage", "garbage", "overflow", "spill", "flood",
                "road", "street", "pothole", "hole"}

CRITICAL_LABELS_IF_DESC_MISSING = {
    "Gas Leak", "Radiation", "Radioactive Leak", "Hazardous Materials",
    "Chemical Spill", "Asbestos", "Fire", "WATER LEAK"
}

PRIORITY_MAP = {
    'Water System': 'High', 'WATER LEAK': 'High', 'Sewer': 'High',
    'Gas Leak': 'High', 'Heat/Hot Water': 'High',
    'Rodent': 'Medium', 'Homeless Assistance': 'Medium', 'Unsanitary Condition': 'Medium',
    'Noise': 'Low', 'Street Condition': 'Low', 'Illegal Parking': 'Low'
}
CONF_THRESHOLD = 0.60

def _tokens(text: str):
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [t for t in text.split() if t]

def detect_emergency(descriptor: str):
    """
    Detect obvious emergencies from descriptor text.
    Returns (is_emergency: bool, suggested_label: str|None, reason: str|None)
    """
    desc = (descriptor or "").lower()
    toks = set(_tokens(desc))

    # phrase match
    for ph in EMERGENCY_PHRASES:
        if ph in desc:
            return True, "Gas Leak" if "gas" in ph else "Hazard", "phrase"

    # token combos
    if ("gas" in toks and ("smell" in toks or "leak" in toks or "odor" in toks)) \
       or ("leak" in toks and "water" not in toks):
        return True, "Gas Leak", "tokens"

    # single strong tokens
    if toks & {"explosion", "fire"}:
        return True, "Fire", "token"

    if "gas" in toks:
        return True, "Gas Leak", "token"

    return False, None, None

def priority_decider(original_label, predicted_label, descriptor, confidence):
    desc = (descriptor or "").strip().lower()
    pred = (predicted_label or "").strip()
    toks = set(_tokens(desc))

    # emergency patterns (if we got here, there was no hard emergency override, but still check)
    for ph in EMERGENCY_PHRASES:
        if ph in desc:
            return "High (Emergency Phrase)", "keyword"
    if toks:
        if ("gas" in toks and ("smell" in toks or "leak" in toks or "odor" in toks)) \
           or ("leak" in toks and "water" not in toks):
            return "High (Emergency Tokens)", "keyword"
        if toks & EMERGENCY_WORDS:
            return "High (Emergency Token)", "keyword"

    # medium issues
    if toks & MEDIUM_WORDS:
        if confidence is not None and confidence > 0.50:
            return "High", "descriptor-medium+confident"
        return "Medium", "descriptor-medium"

    # descriptor missing -> look at original label
    if not desc and original_label:
        if any(cl.lower() == original_label.lower() for cl in CRITICAL_LABELS_IF_DESC_MISSING):
            return "High (Label Emergency - Missing description)", "label-missing"

    # base priority from predicted label
    base = PRIORITY_MAP.get(pred, PRIORITY_MAP.get(pred.upper(), "Low"))

    if confidence is not None:
        if confidence < 0.40:
            return "Low (Review - low confidence)", "low-confidence"
        if confidence < CONF_THRESHOLD:
            return f"{base} (Review)", "low-confidence"

    return base, "predicted"

# -------------- SAFE PARSING HELPERS --------------
def to_float_safe(val, default=0.0):
    if val is None:
        return default
    s = str(val).strip()
    if s == "":
        return default
    try:
        return float(s)
    except Exception:
        return default

def to_int_safe(val, default=0):
    if val is None:
        return default
    s = str(val).strip()
    if s == "":
        return default
    try:
        return int(float(s))
    except Exception:
        return default

def get_form_value(names, default=""):
    """
    Try multiple possible form field names (for robustness: lower/camel/space).
    Example: get_form_value(["descriptor", "Descriptor"])
    """
    for n in names:
        v = request.form.get(n)
        if v is not None:
            return v
    return default

# -------------- PIPELINE PREDICTION --------------
def predict_with_confidence(df_row: pd.DataFrame):
    if model_global is None:
        raise RuntimeError(f"Pipeline not loaded: {model_load_error}")
    try:
        pred = model_global.predict(df_row)[0]
        conf = None
        if hasattr(model_global, "predict_proba"):
            proba = model_global.predict_proba(df_row)
            conf = float(np.max(proba, axis=1)[0])
        return pred, conf
    except Exception as e:
        raise RuntimeError(f"Pipeline prediction error: {e}")

# -------------- ROUTES --------------
@app.route("/", methods=["GET", "POST"])
def index():
    if model_global is None:
        return render_template("index.html", result=None, input_data=None, model_error=model_load_error)

    result = None
    input_data = None

    if request.method == "POST":
        # use correct lowercase names, but allow variants
        original_label = get_form_value(["complaint_type", "Complaint_Type"]).strip()
        descriptor = get_form_value(["descriptor", "Descriptor"]).strip()
        agency = get_form_value(["agency_name", "Agency_Name"]).strip()
        borough = get_form_value(["borough", "Borough"]).strip()

        lat = to_float_safe(get_form_value(["latitude", "Latitude"], None), 0.0)
        lon = to_float_safe(get_form_value(["longitude", "Longitude"], None), 0.0)
        hour = to_int_safe(get_form_value(["hour", "Hour"], None), 0)
        dow = to_int_safe(get_form_value(["day_of_week", "DayOfWeek"], None), 0)
        month = to_int_safe(get_form_value(["month", "Month"], None), 0)

        df = pd.DataFrame([{
            "Descriptor": descriptor,
            "Agency Name": agency,
            "Borough": borough,
            "Latitude": lat,
            "Longitude": lon,
            "Hour": hour,
            "DayOfWeek": dow,
            "Month": month
        }])

        print("\n--- PREDICTION ATTEMPT ---")
        print("Raw descriptor:", repr(descriptor))
        print("TOKENS:", _tokens(descriptor))
        print("INPUT DF:", df.to_dict(orient="records")[0])

        # 1) Emergency override – this runs BEFORE model
        is_emg, suggested_label, emg_reason = detect_emergency(descriptor)
        if is_emg:
            pred_label = suggested_label or "Emergency"
            conf = 0.99
            priority = "High (Emergency Override)"
            reason = f"emergency-{emg_reason}"
            print("Emergency detected -> override. Label:", pred_label, "Reason:", reason)
            result = {"prediction": pred_label, "confidence": conf, "priority": priority, "reason": reason}
            input_data = df.iloc[0].to_dict()
            return render_template("index.html", result=result, input_data=input_data, model_error=None)

        # 2) Normal pipeline prediction + priority_decider
        try:
            pred_label, conf = predict_with_confidence(df)
            print("Model prediction:", pred_label, "confidence:", conf)
            priority, reason = priority_decider(original_label, pred_label, descriptor, conf)
            result = {"prediction": pred_label, "confidence": conf, "priority": priority, "reason": reason}
            input_data = df.iloc[0].to_dict()
        except Exception as e:
            traceback.print_exc()
            result = {"error": str(e)}
            input_data = df.iloc[0].to_dict()

    return render_template("index.html", result=result, input_data=input_data, model_error=None)

@app.route("/api/predict", methods=["POST"])
def api_predict():
    if model_global is None:
        return jsonify({"error": model_load_error}), 500

    payload = request.get_json(force=True)
    descriptor = (payload.get("descriptor") or "").strip()
    agency = (payload.get("agency_name") or "").strip()
    borough = (payload.get("borough") or "").strip()

    df = pd.DataFrame([{
        "Descriptor": descriptor,
        "Agency Name": agency,
        "Borough": borough,
        "Latitude": to_float_safe(payload.get("latitude", None), 0.0),
        "Longitude": to_float_safe(payload.get("longitude", None), 0.0),
        "Hour": to_int_safe(payload.get("hour", None), 0),
        "DayOfWeek": to_int_safe(payload.get("day_of_week", None), 0),
        "Month": to_int_safe(payload.get("month", None), 0)
    }])

    # emergency override for API as well
    is_emg, suggested_label, emg_reason = detect_emergency(descriptor)
    if is_emg:
        pred_label = suggested_label or "Emergency"
        conf = 0.99
        priority = "High (Emergency Override)"
        reason = f"emergency-{emg_reason}"
        return jsonify({"prediction": pred_label, "confidence": conf, "priority": priority, "reason": reason})

    try:
        pred_label, conf = predict_with_confidence(df)
        priority, reason = priority_decider(payload.get("complaint_type", ""), pred_label, descriptor, conf)
        return jsonify({"prediction": pred_label, "confidence": conf, "priority": priority, "reason": reason})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False, port=PORT)


