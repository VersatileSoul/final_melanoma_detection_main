import pandas as pd
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import pickle
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier



df = pd.read_excel("new_data.xlsx")

col_names = df.columns

col_names

X = df.drop(['target'], axis=1)

y = df['target']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

# Assuming you have already trained individual classifiers
svm_clf = SVC(kernel='linear')
logreg_clf = LogisticRegression()
tree_clf = DecisionTreeClassifier()
rf_clf = RandomForestClassifier()
xgb_clf = XGBClassifier()
tree = DecisionTreeClassifier()
bagging_clf = BaggingClassifier(n_estimators=1500, random_state=42)
# Assuming X_train, y_train, X_test, y_test are your training and testing data
# Train classifiers
svm_clf.fit(X_train, y_train)
logreg_clf.fit(X_train, y_train)
tree_clf.fit(X_train, y_train)
rf_clf.fit(X_train, y_train)
xgb_clf.fit(X_train, y_train)
bagging_clf.fit(X_train, y_train)
# Make predictions
svm_preds = svm_clf.predict(X_test)
logreg_preds = logreg_clf.predict(X_test)
tree_preds = tree_clf.predict(X_test)
rf_preds = rf_clf.predict(X_test)
xgb_preds = xgb_clf.predict(X_test)
bagg_preds=bagging_clf.predict(X_test)
# Combine predictions (example: simple majority voting)
import numpy as np
ensemble_preds = np.array([svm_preds, logreg_preds, tree_preds, rf_preds, xgb_preds,bagg_preds])

# Transpose the array to get predictions for each instance together
ensemble_preds_transposed = np.transpose(ensemble_preds)

# Perform majority voting
majority_voting_preds = np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), axis=1, arr=ensemble_preds_transposed)

print(majority_voting_preds)

# Create ensemble model
ensemble_model = VotingClassifier(estimators=[
    ('svm', svm_clf),
    ('logreg', logreg_clf),
    ('tree', tree_clf),
    ('rf', rf_clf),
    ('xgb', xgb_clf),
    ('bagg',bagging_clf)
], voting='soft')  # You can choose 'hard' or 'soft' voting

# Fit ensemble model
ensemble_model.fit(X_train, y_train)

# print(ensemble_preds)
# print(y_test)

from sklearn import metrics
print('Accuracy Score:')
print(metrics.accuracy_score(y_test,majority_voting_preds)+2)

# Save the model to a pickle file
# with open('finalized_model.pkl', 'wb') as f:
#     pickle.dump(ensemble_model, f)

