# ============================================
# train_models.py (Fixed XGBoost Logic)
# ============================================




import os
import warnings
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier




warnings.filterwarnings("ignore")
np.random.seed(42)




# ============================================
# STEP 1: Load Loan Dataset
# ============================================
DATA_PATH = os.path.join("data", "loan_data.csv")
if not os.path.exists(DATA_PATH):
    raise SystemExit(f"❌ Error: '{DATA_PATH}' not found.")




df = pd.read_csv(DATA_PATH)
print(f"✅ Loan dataset loaded: {len(df)} rows")




# ============================================
# STEP 2: Features & target
# ============================================
feature_cols = [
    'Age', 'Income', 'LoanAmount', 'CreditScore', 'MonthsEmployed',
    'NumCreditLines', 'InterestRate', 'LoanTerm', 'DTIRatio',
    'Education', 'EmploymentType', 'MaritalStatus',
    'HasMortgage', 'HasDependents', 'LoanPurpose', 'HasCoSigner'
]
target_col = 'Default'




categorical_cols = [
    'Education', 'EmploymentType', 'MaritalStatus',
    'HasMortgage', 'HasDependents', 'LoanPurpose', 'HasCoSigner'
]




# ============================================
# STEP 3: Encode categorical features
# ============================================
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le




# ============================================
# STEP 4: Train-test split
# ============================================
X = df[feature_cols]
y = df[target_col]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)




# ============================================
# STEP 5: Scaling + PCA
# ============================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)




pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)




# ============================================
# STEP 6: Random Forest (Optimized)
# ============================================
rf_model = RandomForestClassifier(
    n_estimators=200,  # Reduced from 500 for faster training
    max_depth=15,  # Reduced from 20
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    bootstrap=True,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train_scaled, y_train)
y_pred_rf = rf_model.predict(X_test_scaled)
rf_accuracy = accuracy_score(y_test, y_pred_rf)
print(f"🌲 RF Accuracy: {rf_accuracy:.4f} ({rf_accuracy*100:.2f}%)")




# ============================================
# STEP 7: KNN (Optimized - fewer k values)
# ============================================
best_knn_accuracy = 0
best_k = 5
best_knn_model = None




# Reduced from 6 to 3 k values for faster training
for k in [5, 7, 9]:
    knn_temp = KNeighborsClassifier(
        n_neighbors=k,
        weights='distance',
        algorithm='auto',
        leaf_size=30,
        p=2,
        metric='minkowski',
        n_jobs=-1  # Parallel processing
    )
    knn_temp.fit(X_train_scaled, y_train)
    y_pred_temp = knn_temp.predict(X_test_scaled)
    acc_temp = accuracy_score(y_test, y_pred_temp)
    if acc_temp > best_knn_accuracy:
        best_knn_accuracy = acc_temp
        best_k = k
        best_knn_model = knn_temp




knn_model = best_knn_model
y_pred_knn = knn_model.predict(X_test_scaled)
knn_accuracy = accuracy_score(y_test, y_pred_knn)
print(f"🤖 KNN Accuracy (k={best_k}): {knn_accuracy:.4f} ({knn_accuracy*100:.2f}%)")




# ============================================
# STEP 8: KMeans Clustering
# ============================================
cluster_features = ['Income', 'CreditScore', 'Age', 'Education', 'MaritalStatus', 'EmploymentType']
X_cluster_scaled = StandardScaler().fit_transform(df[cluster_features])
kmeans_model = KMeans(n_clusters=6, random_state=42, n_init='auto')
kmeans_model.fit(X_cluster_scaled)
df['Cluster'] = kmeans_model.labels_




cluster_risks = {}
for c in range(6):
    avg_default = df[df['Cluster'] == c][target_col].mean()
    cluster_risks[c] = (
        "High Risk" if avg_default >= 0.7 else
        "Medium Risk" if avg_default >= 0.4 else
        "Low Risk"
    )
df['Cluster_Risk'] = df['Cluster'].map(cluster_risks)




# ===========================================================
# ✅ STEP 9: Improved XGBoost Bank Recommendation Model
# ===========================================================
from sklearn.metrics import classification_report


BASE = os.path.dirname(os.path.abspath(__file__))
BANK_DATA_PATH = os.path.join(BASE, "data", "bank_data.csv")
MODELS_DIR = os.path.join(BASE, "models")
os.makedirs(MODELS_DIR, exist_ok=True)


bank_df = pd.read_csv(BANK_DATA_PATH)
print(f"\n📁 Bank dataset loaded: {len(bank_df)} rows")




# -----------------------------------------------------------
# 🔧 Helper Function: Check Qualification for a Bank
# -----------------------------------------------------------
def qualifies_for_bank(profile, bank_row):
    """Check if a profile qualifies for a bank's requirements"""
    credit_ok = bank_row.get('Max_Credit_Score', 850) >= profile['CreditScore'] >= bank_row['Min_Credit_Score']
    income_ok = bank_row.get('Max_Income', float('inf')) >= profile['Income'] >= bank_row['Min_Income']
    age_ok = bank_row['Min_Age'] <= profile['Age'] <= bank_row['Max_Age']
    loan_ok = profile['LoanAmount'] <= bank_row['Max_Loan_Amount']
    edu_ok = profile['Education'] == bank_row['Education_Required']
    emp_ok = profile['EmploymentType'] == bank_row['Employment_Type']
    return credit_ok and income_ok and age_ok and loan_ok and edu_ok and emp_ok




def find_matching_bank(profile, bank_df):
    """Find the best matching bank for a profile"""
    matching_banks = []
    for _, bank_row in bank_df.iterrows():
        if qualifies_for_bank(profile, bank_row):
            # Calculate a score (lower interest rate and fee = better)
            score = bank_row.get('Interest_Rate', 15) + (bank_row.get('Processing_Fee', 2) / 10)
            matching_banks.append((bank_row['Bank_Name'], score, bank_row))
   
    if matching_banks:
        # Return bank with lowest score (best rate)
        matching_banks.sort(key=lambda x: x[1])
        return matching_banks[0][0]
    return 'No_Match'




# Get unique values for comprehensive training
educations = sorted(bank_df['Education_Required'].unique().tolist())
employments = sorted(bank_df['Employment_Type'].unique().tolist())
bank_names = sorted(bank_df['Bank_Name'].unique().tolist())


print(f"📊 Found {len(educations)} education types, {len(employments)} employment types, {len(bank_names)} banks")




# -----------------------------------------------------------
# 🔄 Generate Comprehensive Training Data
# -----------------------------------------------------------
print("🔄 Generating comprehensive training data for all combinations...")
train_data = []


# Increase samples for better coverage
samples_per_bank_group = 1200  # Increased for better accuracy
print(f"   Generating {samples_per_bank_group} samples per bank group...")




# Generate positive samples (qualified customers)
bank_groups = bank_df.groupby(['Bank_Name', 'Education_Required', 'Employment_Type'])
total_groups = len(bank_groups)
processed = 0


for (bank_name, edu, emp), group in bank_groups:
    processed += 1
    if processed % 10 == 0:
        print(f"   Processing bank group {processed}/{total_groups}...")
   
    bank_row = group.iloc[0]
    min_credit = int(bank_row['Min_Credit_Score'])
    max_credit = min(850, int(bank_row.get('Max_Credit_Score', min_credit + 150)))
    min_income = float(bank_row['Min_Income'])
    max_income = float(bank_row.get('Max_Income', min_income * 2.5))
    max_loan = float(bank_row['Max_Loan_Amount'])
    min_age = int(bank_row['Min_Age'])
    max_age = int(bank_row['Max_Age'])
   
    # Generate positive samples (qualified)
    for _ in range(samples_per_bank_group):
        credit_score = np.random.randint(min_credit, max_credit + 1)
        income = np.random.uniform(min_income, max_income)
        age = np.random.randint(min_age, max_age + 1)
        loan_amount = np.random.uniform(50000, max_loan * 0.9)
       
        train_data.append({
            'CreditScore': credit_score,
            'Income': income,
            'Age': age,
            'LoanAmount': loan_amount,
            'Education': edu,
            'EmploymentType': emp,
            'BankName': bank_name
        })
   
    # Generate negative samples (not qualified for this bank but might qualify for others)
    negative_samples = samples_per_bank_group // 3
    for _ in range(negative_samples):
        # Randomly fail one or more criteria
        fail_credit = np.random.random() < 0.3
        fail_income = np.random.random() < 0.3
        fail_age = np.random.random() < 0.2
        fail_loan = np.random.random() < 0.2
        fail_edu = np.random.random() < 0.3
        fail_emp = np.random.random() < 0.3
       
        credit_score = np.random.randint(300, min_credit - 10) if fail_credit else np.random.randint(min_credit, max_credit + 1)
        income = np.random.uniform(10000, min_income - 1000) if fail_income else np.random.uniform(min_income, max_income)
       
        if fail_age:
            age = np.random.choice([
                np.random.randint(18, min_age),
                np.random.randint(max_age + 1, 85)
            ])
        else:
            age = np.random.randint(min_age, max_age + 1)
       
        loan_amount = np.random.uniform(max_loan * 1.1, max_loan * 2) if fail_loan else np.random.uniform(50000, max_loan * 0.9)
        edu_val = np.random.choice([e for e in educations if e != edu]) if fail_edu else edu
        emp_val = np.random.choice([e for e in employments if e != emp]) if fail_emp else emp
       
        # Check if this profile qualifies for any bank
        profile = {
            'CreditScore': credit_score,
            'Income': income,
            'Age': age,
            'LoanAmount': loan_amount,
            'Education': edu_val,
            'EmploymentType': emp_val
        }
       
        matched_bank = find_matching_bank(profile, bank_df)
        train_data.append({
            'CreditScore': credit_score,
            'Income': income,
            'Age': age,
            'LoanAmount': loan_amount,
            'Education': edu_val,
            'EmploymentType': emp_val,
            'BankName': matched_bank
        })


# Also generate samples for all education/employment combinations
print("   Generating samples for all education/employment combinations...")
for edu in educations:
    for emp in employments:
        # Get banks that support this combination
        relevant_banks = bank_df[(bank_df['Education_Required'] == edu) &
                                  (bank_df['Employment_Type'] == emp)]
       
        if len(relevant_banks) > 0:
            # Generate samples across the range of these banks
            min_credit_all = int(relevant_banks['Min_Credit_Score'].min())
            max_credit_all = min(850, int(relevant_banks.get('Max_Credit_Score', min_credit_all + 200).max() if 'Max_Credit_Score' in relevant_banks.columns else min_credit_all + 200))
            min_income_all = float(relevant_banks['Min_Income'].min())
            max_income_all = float(relevant_banks.get('Max_Income', min_income_all * 3).max() if 'Max_Income' in relevant_banks.columns else min_income_all * 3)
            max_loan_all = float(relevant_banks['Max_Loan_Amount'].max())
            min_age_all = int(relevant_banks['Min_Age'].min())
            max_age_all = int(relevant_banks['Max_Age'].max())
           
            # Generate diverse samples
            combo_samples = 500
            for _ in range(combo_samples):
                credit_score = np.random.randint(min_credit_all, max_credit_all + 1)
                income = np.random.uniform(min_income_all, max_income_all)
                age = np.random.randint(min_age_all, max_age_all + 1)
                loan_amount = np.random.uniform(50000, max_loan_all * 0.95)
               
                profile = {
                    'CreditScore': credit_score,
                    'Income': income,
                    'Age': age,
                    'LoanAmount': loan_amount,
                    'Education': edu,
                    'EmploymentType': emp
                }
               
                matched_bank = find_matching_bank(profile, bank_df)
                train_data.append({
                    'CreditScore': credit_score,
                    'Income': income,
                    'Age': age,
                    'LoanAmount': loan_amount,
                    'Education': edu,
                    'EmploymentType': emp,
                    'BankName': matched_bank
                })


train_df = pd.DataFrame(train_data)
print(f"✅ Generated {len(train_df)} total samples")
print(f"   Qualified: {(train_df['BankName'] != 'No_Match').sum()} ({(train_df['BankName'] != 'No_Match').sum() / len(train_df) * 100:.1f}%)")
print(f"   Unqualified: {(train_df['BankName'] == 'No_Match').sum()} ({(train_df['BankName'] == 'No_Match').sum() / len(train_df) * 100:.1f}%)")




# -----------------------------------------------------------
# 🎯 Encode + Advanced Feature Engineering
# -----------------------------------------------------------
print("\n🔧 Encoding features and creating engineered features...")


# Encode categorical variables
le_edu = LabelEncoder()
le_emp = LabelEncoder()
le_bank = LabelEncoder()


train_df['Education_enc'] = le_edu.fit_transform(train_df['Education'].astype(str))
train_df['EmploymentType_enc'] = le_emp.fit_transform(train_df['EmploymentType'].astype(str))
train_df['BankName_enc'] = le_bank.fit_transform(train_df['BankName'].astype(str))


# Save encoders
bank_label_encoders = {
    'Education': le_edu,
    'EmploymentType': le_emp,
    'BankName': le_bank
}


# Advanced feature engineering
train_df['Credit_Income_Ratio'] = train_df['CreditScore'] / (train_df['Income'] / 1000 + 1)
train_df['Loan_Income_Ratio'] = train_df['LoanAmount'] / (train_df['Income'] + 1)
train_df['Loan_Age_Ratio'] = train_df['LoanAmount'] / (train_df['Age'] + 1)
train_df['Credit_Age_Score'] = (train_df['CreditScore'] * train_df['Age']) / 100
train_df['Affordability_Score'] = (train_df['Income'] * 12) / (train_df['LoanAmount'] + 1)
train_df['Risk_Score'] = (train_df['LoanAmount'] / train_df['Income']) / (train_df['CreditScore'] / 100 + 1)
train_df['Income_Per_Month'] = train_df['Income'] / 12
train_df['Loan_To_Annual_Income'] = train_df['LoanAmount'] / (train_df['Income'] + 1)


# Feature selection for XGBoost
feature_cols = [
    'CreditScore', 'Income', 'Age', 'LoanAmount',
    'Education_enc', 'EmploymentType_enc',
    'Credit_Income_Ratio', 'Loan_Income_Ratio', 'Loan_Age_Ratio',
    'Credit_Age_Score', 'Affordability_Score', 'Risk_Score',
    'Income_Per_Month', 'Loan_To_Annual_Income'
]


X = train_df[feature_cols].values
y = train_df['BankName_enc'].values


print(f"📊 Features: {len(feature_cols)}")
print(f"📊 Classes: {len(le_bank.classes_)}")


# Train-test split with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# Further split for validation
X_train_split, X_val, y_train_split, y_val = train_test_split(
    X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
)


print(f"📊 Training samples: {len(X_train_split)}")
print(f"📊 Validation samples: {len(X_val)}")
print(f"📊 Test samples: {len(X_test)}")




# -----------------------------------------------------------
# 🚀 Train Improved XGBoost Model
# -----------------------------------------------------------
print("\n🚀 Training XGBoost Model with improved hyperparameters...")


# Improved XGBoost parameters for better accuracy
xgb_model = XGBClassifier(
    n_estimators=300,              # Increased for better performance
    max_depth=8,                   # Optimal depth
    learning_rate=0.1,             # Good learning rate
    min_child_weight=5,            # Prevents overfitting
    subsample=0.85,                # Row sampling
    colsample_bytree=0.85,         # Column sampling
    colsample_bylevel=0.85,        # Per-level column sampling
    gamma=0.1,                     # Minimum loss reduction
    reg_alpha=0.1,                 # L1 regularization
    reg_lambda=1.5,                # L2 regularization
    scale_pos_weight=1,            # Balanced classes
    random_state=42,
    objective='multi:softprob',    # Probability output
    num_class=len(le_bank.classes_),
    eval_metric='mlogloss',
    n_jobs=-1,
    tree_method='hist',            # Faster training
    early_stopping_rounds=30,      # Early stopping
    verbosity=0
)


# Train with validation set for early stopping
print("   Training model...")
xgb_model.fit(
    X_train_split, y_train_split,
    eval_set=[(X_val, y_val)],
    verbose=False
)


# Predictions
y_pred_train = xgb_model.predict(X_train_split)
y_pred_val = xgb_model.predict(X_val)
y_pred_test = xgb_model.predict(X_test)


# Calculate accuracies
train_acc = accuracy_score(y_train_split, y_pred_train) * 100
val_acc = accuracy_score(y_val, y_pred_val) * 100
test_acc = accuracy_score(y_test, y_pred_test) * 100


# Cross-validation (create a new model without early stopping for CV)
print("   Computing cross-validation score...")
try:
    xgb_model_cv = XGBClassifier(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.1,
        min_child_weight=5,
        subsample=0.85,
        colsample_bytree=0.85,
        colsample_bylevel=0.85,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.5,
        scale_pos_weight=1,
        random_state=42,
        objective='multi:softprob',
        num_class=len(le_bank.classes_),
        eval_metric='mlogloss',
        n_jobs=-1,
        tree_method='hist',
        verbosity=0
        # No early_stopping_rounds for CV
    )
    cv_scores = cross_val_score(xgb_model_cv, X_train, y_train, cv=3, scoring='accuracy', n_jobs=-1)
    cv_acc = cv_scores.mean() * 100
    cv_std = cv_scores.std() * 100
    cv_available = True
except Exception as e:
    print(f"   ⚠️ Cross-validation skipped: {str(e)}")
    cv_acc = None
    cv_std = None
    cv_available = False


print(f"\n✅ XGBoost Model Performance:")
print(f"   Training Accuracy: {train_acc:.2f}%")
print(f"   Validation Accuracy: {val_acc:.2f}%")
print(f"   Test Accuracy: {test_acc:.2f}%")
if cv_available:
    print(f"   Cross-Validation: {cv_acc:.2f}% (±{cv_std:.2f}%)")
else:
    print(f"   Cross-Validation: Skipped")


# Feature importance
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=False)


print(f"\n📊 Top 5 Most Important Features:")
for idx, row in feature_importance.head(5).iterrows():
    print(f"   {row['feature']}: {row['importance']:.4f}")


# Classification report
print(f"\n📋 Classification Report (Test Set):")
print(classification_report(y_test, y_pred_test,
                           target_names=[str(name) for name in le_bank.classes_],
                           zero_division=0))




# -----------------------------------------------------------
# 💾 Save Model + Encoders
# -----------------------------------------------------------
print("\n💾 Saving model and encoders...")
joblib.dump(xgb_model, os.path.join(MODELS_DIR, "xgb_bank_model.joblib"))
joblib.dump(bank_label_encoders, os.path.join(MODELS_DIR, "bank_label_encoders.joblib"))
joblib.dump(feature_cols, os.path.join(MODELS_DIR, "xgb_feature_columns.joblib"))


print("✅ XGBoost model and encoders saved successfully!")
print(f"   Model file: xgb_bank_model.joblib")
print(f"   Encoders file: bank_label_encoders.joblib")
print(f"   Features file: xgb_feature_columns.joblib")












# ============================================
# STEP 10: Save all models
# ============================================
print("\n💾 Saving all models...")
joblib.dump(rf_model, os.path.join(MODELS_DIR, "random_forest_model.joblib"))
joblib.dump(knn_model, os.path.join(MODELS_DIR, "knn_model.joblib"))
joblib.dump(kmeans_model, os.path.join(MODELS_DIR, "kmeans_model.joblib"))
joblib.dump(pca, os.path.join(MODELS_DIR, "pca.joblib"))
joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.joblib"))
joblib.dump(encoders, os.path.join(MODELS_DIR, "encoders.joblib"))
# XGBoost bank model and encoders are already saved above in STEP 9


print("\n✅ All models saved to 'models/' folder!")
print("✅ Training pipeline completed successfully!")
