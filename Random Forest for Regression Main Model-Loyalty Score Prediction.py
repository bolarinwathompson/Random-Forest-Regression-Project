# Random Forest for Regression - ABC Grocery Task

# Import Required Packages
import pandas as pd
import pickle
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import RFECV
from sklearn.inspection import permutation_importance 

# -------------------------------------------------------------------
# Import Sample Data
data_for_model = pickle.load(open("regression_modelling.p", "rb"))

# -------------------------------------------------------------------
# Drop Unnecessary Columns
data_for_model.drop("customer_id", axis=1, inplace=True)

# -------------------------------------------------------------------
# Shuffle the Data
data_for_model = shuffle(data_for_model, random_state=42)

# -------------------------------------------------------------------
# Deal with Missing Data
print("Missing values per column:")
print(data_for_model.isna().sum())
data_for_model.dropna(how="any", inplace=True)


# -------------------------------------------------------------------
# Split Input Variables and Output Variable
X = data_for_model.drop(["customer_loyalty_score"], axis=1)
y = data_for_model["customer_loyalty_score"]

# -------------------------------------------------------------------
# Split out Training and Test Sets (this step ensures X_test is defined)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------------------------------------------------------
# Feature Selection: One-Hot Encoding for Categorical Variable(s)
categorical_vars = ["gender"]

# Use sparse_output=False for scikit-learn 1.2+; change to sparse=False if using an earlier version.
one_hot_encoder = OneHotEncoder(sparse_output=False, drop="first")

# Fit the encoder on the training set's categorical column and transform both training and test sets.
X_train_encoded = one_hot_encoder.fit_transform(X_train[categorical_vars])
X_test_encoded = one_hot_encoder.transform(X_test[categorical_vars])

# Get the encoded feature names using the new method.
encoder_feature_names = one_hot_encoder.get_feature_names_out(categorical_vars)

# Convert the encoded arrays into DataFrames (keeping the original indices for proper alignment).
X_train_encoded_df = pd.DataFrame(X_train_encoded, columns=encoder_feature_names, index=X_train.index)
X_test_encoded_df = pd.DataFrame(X_test_encoded, columns=encoder_feature_names, index=X_test.index)

# Remove the original categorical column(s) from X_train and X_test.
X_train_clean = X_train.drop(columns=categorical_vars)
X_test_clean = X_test.drop(columns=categorical_vars)

# Concatenate the cleaned DataFrames with their corresponding encoded DataFrames.
X_train = pd.concat([X_train_clean, X_train_encoded_df], axis=1)
X_test = pd.concat([X_test_clean, X_test_encoded_df], axis=1)


# Model Training 

regressor = RandomForestRegressor(random_state = 42)
regressor.fit(X_train, y_train)



# Predict The Test Score
y_pred = regressor.predict(X_test)

# Calculate the R-Squared
r_squared = r2_score(y_test, y_pred)
print(r_squared)


# Cross Validation 

cv = KFold(n_splits = 4, shuffle = True, random_state = 42)

cv_scores = cross_val_score(regressor, X_train, y_train, cv = cv, scoring = "r2")
cv_scores.mean()


# Calculate the Adjusted R-Square

num_data_points, num_input_vars = X_test.shape
adjusted_r_squared = 1 -(1 - r_squared)*(num_data_points - 1)/ (num_data_points - num_input_vars - 1)
print(adjusted_r_squared)

# Feature Importance


feature_importance = pd.DataFrame(regressor.feature_importances_)
feature_names = pd.DataFrame(X.columns)
feature_importance_summary = pd.concat([feature_names, feature_importance], axis = 1)
feature_importance_summary.columns = ["input_variable", "feature_importance"]
feature_importance_summary.sort_values(by = "feature_importance", inplace = True)

plt.barh(feature_importance_summary["input_variable"],feature_importance_summary ["feature_importance"])
plt.title("Feature Importance of Random Forest")
plt.xlabel("Feature Importance")
plt.tight_layout()
plt.show

# Permutation Importance

result = permutation_importance(regressor, X_test, y_test, n_repeats= 10, random_state= 42)
print(result)

permutation_importance = pd.DataFrame(result["importances_mean"])
permutation_names = pd.DataFrame(X.columns)
permutation_importance_summary = pd.concat([feature_names, permutation_importance], axis = 1)
permutation_importance_summary.columns = ["input_variable", "permutation_importance"]
permutation_importance_summary.sort_values(by = "permutation_importance", inplace = True)

plt.barh(permutation_importance_summary["input_variable"],permutation_importance_summary ["permutation_importance"])
plt.title("Permutation Importance of Random Forest")
plt.xlabel("Permutation Importance")
plt.tight_layout()
plt.show

# Prediction under the Hood

y_pred[0]
new_data = [X_test.iloc[0]]
regressor.estimators_
print(regressor.estimators_)

predictions = []
tree_count = 0
for tree in regressor.estimators_:
    prediction = tree.predict(new_data)[0]
    predictions.append(prediction)
    tree_count += 1
    
print(predictions)
sum(predictions) / tree_count

import pickle

pickle.dump(regressor, open("random_forest_regression_model.p", "wb"))

pickle.dump(one_hot_encoder, open("random_forest_regression_ohe.p", "wb"))












