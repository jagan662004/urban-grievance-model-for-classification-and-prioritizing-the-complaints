# urban-grievance-model-for-classification-and-prioritizing-the-complaints
# Urban Grievance Classification & Priority System

##  Project Overview

Urban local bodies receive a huge number of citizen complaints every day related to noise, sanitation, traffic, utilities, public safety, and infrastructure. Handling these complaints manually is time-consuming and often leads to delays in resolving critical issues.

This project builds an **AI-based system to automatically classify urban grievance complaints and assign priority levels** using Machine Learning and Natural Language Processing (NLP). The system helps authorities quickly identify **high‑risk and emergency complaints** (such as gas leaks or fire hazards) and act on them promptly.

The solution is implemented as a **Flask web application** with a trained ML model running in the backend.

---

##  Objectives

* Automate classification of urban grievance complaints
* Reduce manual effort and response time
* Assign priority levels (High / Medium / Low)
* Detect emergency complaints using keyword intelligence
* Provide a simple web interface for complaint input

---

##  Key Features

* NLP-based complaint classification using TF-IDF and ML models
* Priority prediction based on complaint type, keywords, and confidence
* Emergency override logic for critical situations
* User-friendly web interface (dark theme UI)
* Scalable and extendable architecture

---

## Technologies Used

### Programming & Frameworks

* Python 3.10+
* Flask (Web Framework)

### Machine Learning & NLP

* Scikit-learn
* RandomForestClassifier / Logistic Regression
* TF-IDF Vectorizer
* Pandas, NumPy

### Frontend

* HTML5
* CSS3 (Dark UI theme)

---

##  Dataset

* **NYC 311 Service Requests Dataset**
* https://data.cityofnewyork.us/Social-Services/311-Service-Requests-from-2010-to-Present/erm2-nwe9/about_data --> u can download the dataset from here 
* Contains millions of real-world urban complaints

### Selected Features Used:

* Descriptor (Text description)
* Agency Name
* Borough
* Latitude, Longitude
* Created Date (Hour, DayOfWeek, Month)

### Target Variable:

* Complaint Type

---

## Data Preprocessing Steps

1. Selected relevant columns from raw dataset
2. Removed duplicate records
3. Handled missing values
4. Extracted time-based features (Hour, Day, Month)
5. Grouped rare complaint types into **Other**
6. Encoded text using **TF-IDF**
7. One-hot encoded categorical features
8. Combined text, categorical, and numeric features

---

##  Model Training

* TF-IDF Vectorizer for text representation
* One-Hot Encoding for categorical variables
* Random Forest / Logistic Regression classifier
* Stratified train-test split (80:20)

The trained model is saved as a **joblib pipeline** and loaded directly into the Flask app.

---

## 🚦 Priority Assignment Logic

The system assigns priority using a hybrid approach:

### Priority Levels:

* **High** – Emergency / Safety related
* **Medium** – Nuisance or recurring issues
* **Low** – General or informational complaints

### Decision Factors:

* Emergency keywords (gas, fire, explosion, smoke)
* Complaint classification output
* Model confidence score
* Rule-based overrides for critical cases

---

## Web Application Flow

1. User enters complaint description
2. Optional fields: agency, borough, time context
3. Flask server processes input
4. Model predicts complaint type and confidence
5. Priority logic assigns urgency level
6. Result is displayed on UI

---

##  Example Input

```
Neighbor's dog barking loudly every night after 11 PM
```

### Output:

* Complaint Type: Noise – Barking Dog
* Priority: Medium
* Confidence: Model dependent

---

## How to Run the Project

### Step 1: Install Dependencies

```bash
pip install flask pandas numpy scikit-learn joblib
```

### Step 2: Train the Model

```bash
python pipeline.py
```

### Step 3: Run Flask App

```bash
python app.py
```

### Step 4: Open Browser

```
http://127.0.0.1:8000
```

---

## Results & Observations

* Automation significantly reduces manual workload
* Emergency complaints are detected faster
* Text quality directly impacts prediction accuracy
* Rule-based logic improves reliability for critical cases

---

## Limitations

* Depends on quality of input text
* Large number of complaint categories causes confusion
* Classic ML struggles with semantic understanding

---

## Future Enhancements

* Upgrade to deep NLP models (BERT / MiniLM)
* Add voice-based complaint input
* Real-time dashboard for authorities
* Online deployment (Cloud)
* Feedback loop for continuous learning

---

## Academic Context

* **Course:** INT 395
* **Domain:** Machine Learning & NLP
* **Project Type:** Academic / Mini Project

---

## Conclusion

This project demonstrates how Machine Learning and NLP can be effectively used to improve urban grievance management systems. By automating complaint classification and prioritization, the solution helps authorities respond faster to critical issues and enhances overall civic service efficiency.

---


