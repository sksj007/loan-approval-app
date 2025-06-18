import pandas as pd
from sklearn.model_selection import train_test_split

def load_data():
  url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
  columns = ['status', 'duration', 'credit_history', 'purpose', 'amount',
             'savings', 'employment', 'installment_rates', 'personal_status',
             'other_debtors', 'residence_since', 'property', 'age',
             'other_installments', 'housing', 'existing_credits', 'job',
             'dependents', 'telephone', 'foregin_worker', 'target']
  df = pd.read_csv(url, delimiter=' ', header=None, names=columns)
  df['target'] = df['target'].replace({2:0})
  return df

def preprocess_data(df):
  df = pd.get_dummies(df, drop_first=True)
  X = df.drop('target', axis=1)
  y = df['target']
  return train_test_split(X, y, test_size=0.2, random_state=42)

df = load_data()
X_train, X_test, y_train, y_test = preprocess_data(df)
