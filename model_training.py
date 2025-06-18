from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

def train_model(X_train, y_train):
  model = RandomForestClassifier(n_estimators=100, random_state=42)
  model.fit(X_train, y_train)
  return model

def evaluate_model(model, X_test, y_test):
  y_pred = model.predict(X_test)
  y_pred_proba = model.predict_proba(X_test)[:, 1]
  print(classification_report(y_test, y_pred))
  print(f"AUC-ROC: {roc_auc_score(y_test, y_pred_proba):.2f}")

model = train_model(X_train, y_train)
evaluate_model(model, X_test, y_test)
