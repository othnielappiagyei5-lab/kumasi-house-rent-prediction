# ============================================================
# MACHINE LEARNING-BASED HOUSE RENT PREDICTION
# IN KUMASI, GHANA
#




# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

import statsmodels.api as sm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor


# ============================================================
# 2. LOAD THE DATASET
# ============================================================

df = pd.read_csv("house_rentals.csv")

print("\n================ ORIGINAL DATASET ================\n")
print(df.head())

print("\nDataset shape:")
print(df.shape)


# ============================================================
# 3. FILTER THE DATASET TO KUMASI
# ============================================================

df = df[
    df["location"].str.contains("Kumasi", case=False, na=False)
].copy()

print("\n================ KUMASI DATASET ================\n")
print("Number of Kumasi rental listings:", len(df))

print("\nFirst five Kumasi listings:")
print(df.head())

print("\nKumasi dataset shape:")
print(df.shape)


# ============================================================
# 4. SELECT VARIABLES FOR THE PROJECT
# ============================================================

# These are the variables being used to predict rent.

selected_columns = [
    "price",
    "bedrooms",
    "bathrooms",
    "floor_area",
    "category",
    "condition",
    "parking_space",
    "is_furnished"
]

df = df[selected_columns].copy()


# ============================================================
# 5. DATA INFORMATION
# ============================================================

print("\n================ DATASET INFORMATION ================\n")

print("\nDataset shape:")
print(df.shape)

print("\nSummary statistics:")
print(df.describe())

print("\nDataset information:")
print(df.info())

print("\nColumn names:")
print(df.columns)

print("\nMissing values:")
print(df.isna().sum())

print("\nDuplicate rows:")
print(df.loc[df.duplicated()])


# ============================================================
# 6. REMOVE INVALID/MISSING VALUES
# ============================================================


df = df.dropna().copy()

# Remove records where the target rent is zero or negative.
df = df[df["price"] > 0].copy()

print("\nDataset after cleaning:")
print(df.shape)


# ============================================================
# 7. HOUSE RENT DISTRIBUTION
# ============================================================

plt.figure(figsize=(20, 8))

plt.subplot(1, 2, 1)

plt.title("House Rent Distribution Plot")

sns.histplot(
    df["price"],
    kde=True
)

plt.xlabel("Monthly Rent (GH₵)")
plt.ylabel("Frequency")


plt.subplot(1, 2, 2)

sns.boxplot(
    y=df["price"]
)

plt.title("House Rent Spread")

plt.ylabel("Monthly Rent (GH₵)")

plt.tight_layout()
plt.show()


# ============================================================
# 8. RENT DESCRIPTIVE STATISTICS
# ============================================================

print("\n================ RENT STATISTICS ================\n")

print(
    df["price"].describe(
        percentiles=[
            0.25,
            0.50,
            0.75,
            0.85,
            0.90,
            1
        ]
    )
)


# ============================================================
# 9. IDENTIFY CATEGORICAL VARIABLES
# ============================================================

categorical_list = [
    x for x in df.columns
    if df[x].dtype == "object"
]

print("\n================ CATEGORICAL VARIABLES ================\n")

for x in categorical_list:
    print(x)


# ============================================================
# 10. CATEGORICAL VARIABLE DISTRIBUTIONS
# ============================================================

# ------------------------------------------------------------
# PROPERTY CATEGORY
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

df["category"].value_counts().plot(kind="bar")

plt.title("Property Category Distribution")
plt.xlabel("Property Category")
plt.ylabel("Frequency")

plt.xticks(rotation=45)

plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# CONDITION
# ------------------------------------------------------------

plt.figure(figsize=(8, 6))

df["condition"].value_counts().plot(kind="bar")

plt.title("Property Condition Distribution")
plt.xlabel("Condition")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# PARKING
# ------------------------------------------------------------

plt.figure(figsize=(8, 6))

df["parking_space"].astype(str).value_counts().plot(kind="bar")

plt.title("Parking Space Distribution")
plt.xlabel("Parking Space")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# FURNISHING
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

df["is_furnished"].value_counts().plot(kind="bar")

plt.title("Furnishing Status Distribution")
plt.xlabel("Furnishing Status")
plt.ylabel("Frequency")

plt.xticks(rotation=45)

plt.tight_layout()
plt.show()


# ============================================================
# 11. CATEGORICAL VARIABLES VS RENT
# ============================================================

# ------------------------------------------------------------
# CATEGORY VS RENT
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

sns.boxplot(
    x=df["category"],
    y=df["price"]
)

plt.title("Property Category vs Rent")

plt.xlabel("Property Category")
plt.ylabel("Monthly Rent (GH₵)")

plt.xticks(rotation=45)

plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# CONDITION VS RENT
# ------------------------------------------------------------

plt.figure(figsize=(8, 6))

sns.boxplot(
    x=df["condition"],
    y=df["price"]
)

plt.title("Property Condition vs Rent")

plt.xlabel("Condition")
plt.ylabel("Monthly Rent (GH₵)")

plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# PARKING VS RENT
# ------------------------------------------------------------

plt.figure(figsize=(8, 6))

sns.boxplot(
    x=df["parking_space"].astype(str),
    y=df["price"]
)

plt.title("Parking Space vs Rent")

plt.xlabel("Parking Space")
plt.ylabel("Monthly Rent (GH₵)")

plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# FURNISHING VS RENT
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

sns.boxplot(
    x=df["is_furnished"],
    y=df["price"]
)

plt.title("Furnishing Status vs Rent")

plt.xlabel("Furnishing Status")
plt.ylabel("Monthly Rent (GH₵)")

plt.xticks(rotation=45)

plt.tight_layout()
plt.show()


# ============================================================
# 12. NUMERICAL VARIABLES
# ============================================================

numerical_list = [
    x for x in df.columns
    if df[x].dtype in ("int64", "float64")
]

print("\n================ NUMERICAL VARIABLES ================\n")

print(numerical_list)


# ============================================================
# 13. NUMERICAL VARIABLES VS RENT
# ============================================================

def scatter(x, fig_number):

    plt.subplot(3, 1, fig_number)

    plt.scatter(
        df[x],
        df["price"],
        alpha=0.6
    )

    plt.title(x + " vs Rent")

    plt.ylabel("Monthly Rent (GH₵)")

    plt.xlabel(x)


plt.figure(figsize=(10, 15))

scatter("floor_area", 1)
scatter("bedrooms", 2)
scatter("bathrooms", 3)

plt.tight_layout()
plt.show()


# ============================================================
# 14. PAIRPLOT
# ============================================================

# We use the main numerical variables rather than every
# column in the original dataset.

pairplot_columns = [
    "price",
    "floor_area",
    "bedrooms",
    "bathrooms"
]

sns.pairplot(
    df[pairplot_columns]
)

plt.show()


# ============================================================
# 15. CORRELATION HEATMAP
# ============================================================

correlation_columns = [
    "price",
    "floor_area",
    "bedrooms",
    "bathrooms"
]

cor_matrix = df[correlation_columns].corr()

plt.figure(figsize=(10, 8))

sns.heatmap(
    cor_matrix,
    annot=True,
    cmap="coolwarm",
    linewidths=0.5
)

plt.title("Correlation Heatmap")

plt.tight_layout()
plt.show()


# ============================================================
# 16. CREATE DUMMY VARIABLES
# ============================================================

print("\n================ CREATING DUMMY VARIABLES ================\n")


def dummies(x, df):

    temp = pd.get_dummies(
        df[x],
        drop_first=True
    ).astype(int)

    df = pd.concat(
        [df, temp],
        axis=1
    )

    df.drop(
        [x],
        axis=1,
        inplace=True
    )

    return df


# Convert categorical variables to numerical dummy variables.

df = dummies("category", df)

df = dummies("condition", df)

df = dummies("parking_space", df)

df = dummies("is_furnished", df)


print("\nData after creating dummy variables:")
print(df.head())

print("\nNew dataset shape:")
print(df.shape)

print("\nNew columns:")
print(df.columns.tolist())


# ============================================================
# 17. TRAIN / TEST SPLIT
# ============================================================

# Matching the lecturer's 75% training and 25% testing split.

np.random.seed(0)

df_train, df_test = train_test_split(
    df,
    train_size=0.75,
    test_size=0.25,
    random_state=100
)

print("\n================ TRAIN / TEST SPLIT ================\n")

print("Training data:", df_train.shape)

print("Testing data:", df_test.shape)


# ============================================================
# 18. SCALE NUMERICAL VARIABLES
# ============================================================

scaler = MinMaxScaler()

# Only scale the numerical predictor variables.
# Do not scale the target price here.

scale_columns = [
    "floor_area",
    "bedrooms",
    "bathrooms"
]

df_train[scale_columns] = scaler.fit_transform(
    df_train[scale_columns]
)

print("\nScaled training data:")
print(df_train.head())


# ============================================================
# 19. DIVIDE DATA INTO X AND y
# ============================================================

y_train = df_train.pop("price")

X_train = df_train

print("\nX_train shape:")
print(X_train.shape)

print("\ny_train shape:")
print(y_train.shape)


# ============================================================
# 20. RECURSIVE FEATURE ELIMINATION (RFE)
# ============================================================

print("\n================ RFE FEATURE SELECTION ================\n")

rfe = RFE(
    estimator=LinearRegression(),
    n_features_to_select=10
)

rfe = rfe.fit(
    X_train,
    y_train
)

print("\nFeature ranking:")

print(
    list(
        zip(
            X_train.columns,
            rfe.support_,
            rfe.ranking_
        )
    )
)


print("\nSelected features:")

print(
    X_train.columns[rfe.support_].tolist()
)


X_train_rfe = X_train[
    X_train.columns[rfe.support_]
].copy()

print("\nRFE training data:")

print(X_train_rfe.head())


# ============================================================
# 21. BUILD LINEAR REGRESSION MODEL
# ============================================================

def build_model(X, y):

    X = sm.add_constant(X)

    lm = sm.OLS(
        y,
        X
    ).fit()

    print(lm.summary())

    return X


# ============================================================
# 22. FIRST MODEL
# ============================================================

print("\n================ MODEL 1 ================\n")

X_train_new = build_model(
    X_train_rfe,
    y_train
)


# ============================================================
# 23. CHECK P-VALUES AND REMOVE NON-SIGNIFICANT FEATURES
# ============================================================

print("\n================ P-VALUE CHECK ================\n")

print(
    X_train_new.columns
)


# Automatically identify the least statistically significant
# predictor, excluding the constant.

model_without_constant = X_train_new.drop(
    "const",
    axis=1
)

first_model = sm.OLS(
    y_train,
    X_train_new
).fit()

p_values = first_model.pvalues.drop(
    "const"
)

highest_p_feature = p_values.idxmax()

highest_p_value = p_values.max()

print(
    "\nFeature with highest p-value:",
    highest_p_feature
)

print(
    "P-value:",
    highest_p_value
)


# Following the lecturer's approach, remove a feature when
# its p-value is greater than 0.05.

if highest_p_value > 0.05:

    print(
        "\nRemoving feature because p-value > 0.05:",
        highest_p_feature
    )

    X_train_new = X_train_new.drop(
        highest_p_feature,
        axis=1
    )

else:

    print(
        "\nAll remaining predictors have p-values <= 0.05."
    )


# ============================================================
# 24. SECOND MODEL
# ============================================================

print("\n================ MODEL 2 ================\n")

X_train_new = build_model(
    X_train_new.drop("const", axis=1),
    y_train
)


# ============================================================
# 25. CALCULATE VIF
# ============================================================

def checkVIF(X):

    vif = pd.DataFrame()

    vif["Features"] = X.columns

    vif["VIF"] = [
        variance_inflation_factor(
            X.values,
            i
        )
        for i in range(X.shape[1])
    ]

    vif["VIF"] = round(
        vif["VIF"],
        2
    )

    vif = vif.sort_values(
        by="VIF",
        ascending=False
    )

    return vif


print("\n================ VIF CHECK ================\n")

vif_table = checkVIF(
    X_train_new.drop("const", axis=1)
)

print(vif_table)


# ============================================================
# 26. REMOVE HIGH-VIF VARIABLE
# ============================================================

# The lecturer removed a variable with very high VIF.
# Here we apply the same principle automatically.

vif_predictors = X_train_new.drop(
    "const",
    axis=1
)

vif_table = checkVIF(
    vif_predictors
)

highest_vif_feature = vif_table.iloc[0]["Features"]

highest_vif_value = vif_table.iloc[0]["VIF"]

print(
    "\nHighest VIF feature:",
    highest_vif_feature
)

print(
    "VIF:",
    highest_vif_value
)


if highest_vif_value > 10:

    print(
        "\nRemoving high-VIF feature:",
        highest_vif_feature
    )

    X_train_new = X_train_new.drop(
        highest_vif_feature,
        axis=1
    )

else:

    print(
        "\nNo variable has VIF greater than 10."
    )


# ============================================================
# 27. FINAL TRAINING MODEL
# ============================================================

print("\n================ FINAL MODEL ================\n")

X_train_new = build_model(
    X_train_new.drop("const", axis=1),
    y_train
)


# ============================================================
# 28. FINAL VIF CHECK
# ============================================================

print("\n================ FINAL VIF ================\n")

final_vif = checkVIF(
    X_train_new.drop("const", axis=1)
)

print(final_vif)


# ============================================================
# 29. FIT FINAL OLS MODEL
# ============================================================

lm = sm.OLS(
    y_train,
    X_train_new
).fit()


# ============================================================
# 30. TRAINING PREDICTIONS
# ============================================================

y_train_price = lm.predict(
    X_train_new
)


# ============================================================
# 31. RESIDUAL / ERROR ANALYSIS
# ============================================================

print("\n================ RESIDUAL ANALYSIS ================\n")

errors = y_train - y_train_price

fig = plt.figure()

sns.histplot(
    errors,
    bins=20,
    kde=True
)

fig.suptitle(
    "Error Terms",
    fontsize=20
)

plt.xlabel(
    "Errors",
    fontsize=18
)

plt.ylabel(
    "Frequency",
    fontsize=16
)

plt.tight_layout()
plt.show()


# ============================================================
# 32. PREPARE THE TEST SET
# ============================================================

# Use the SAME scaler fitted on the training data.
# This avoids information leakage from the test data.

df_test[scale_columns] = scaler.transform(
    df_test[scale_columns]
)


# ============================================================
# 33. DIVIDE TEST DATA INTO X AND y
# ============================================================

y_test = df_test.pop("price")

X_test = df_test


# ============================================================
# 34. SELECT THE SAME FEATURES USED BY THE MODEL
# ============================================================

selected_features = X_train_new.columns.tolist()

selected_features.remove("const")


X_test_new = X_test[
    selected_features
].copy()


# ============================================================
# 35. ADD CONSTANT
# ============================================================

X_test_new = sm.add_constant(
    X_test_new,
    has_constant="add"
)


# Make sure the columns have exactly the same order
# as the training model.

X_test_new = X_test_new[
    X_train_new.columns
]


# ============================================================
# 36. MAKE TEST PREDICTIONS
# ============================================================

print("\n================ TEST PREDICTIONS ================\n")

y_pred = lm.predict(
    X_test_new
)

print(
    pd.DataFrame({
        "Actual Rent": y_test.values[:10],
        "Predicted Rent": y_pred.values[:10]
    })
)


# ============================================================
# 37. MODEL R-SQUARED
# ============================================================

r_squared = r2_score(
    y_test,
    y_pred
)

print("\n================ MODEL PERFORMANCE ================\n")

print(
    "R-squared:",
    round(r_squared, 4)
)

print(
    "R-squared percentage:",
    round(r_squared * 100, 2),
    "%"
)


# ============================================================
# 38. ACTUAL VS PREDICTED RENT
# ============================================================

fig = plt.figure()

plt.scatter(
    y_test,
    y_pred,
    alpha=0.7
)

fig.suptitle(
    "Actual Rent vs Predicted Rent",
    fontsize=20
)

plt.xlabel(
    "Actual Rent (GH₵)",
    fontsize=18
)

plt.ylabel(
    "Predicted Rent (GH₵)",
    fontsize=16
)

plt.tight_layout()
plt.show()


# ============================================================
# 39. FINAL MODEL SUMMARY
# ============================================================

print("\n================ FINAL MODEL SUMMARY ================\n")

print(lm.summary())

# ============================================================
# 40. NEW HOUSE RENT PREDICTION
# ============================================================

print("\n================ NEW HOUSE PREDICTION ================\n")


# ------------------------------------------------------------
# HOUSE INFORMATION
# ------------------------------------------------------------

new_house_original = pd.DataFrame({
    "floor_area": [100],
    "bedrooms": [2],
    "bathrooms": [2],
    "category": ["Flats"],
    "condition": ["New"],
    "parking_space": [True],
    "is_furnished": ["Semi-Furnished"]
})


# ------------------------------------------------------------
# CREATE DUMMY VARIABLES
# ------------------------------------------------------------

new_house_model = pd.get_dummies(
    new_house_original,
    columns=[
        "category",
        "condition",
        "parking_space",
        "is_furnished"
    ],
    drop_first=True
).astype(int)


# ------------------------------------------------------------
# MAKE SURE ALL NUMERICAL VARIABLES EXIST
# ------------------------------------------------------------

for column in scale_columns:

    if column not in new_house_model.columns:
        new_house_model[column] = 0


# ------------------------------------------------------------
# SCALE NUMERICAL VARIABLES
# ------------------------------------------------------------

# Scale BEFORE selecting the final model features.

new_house_model[scale_columns] = scaler.transform(
    new_house_model[scale_columns]
)


# ------------------------------------------------------------
# MAKE SURE ALL MODEL FEATURES EXIST
# ------------------------------------------------------------

for column in selected_features:

    if column not in new_house_model.columns:
        new_house_model[column] = 0


# ------------------------------------------------------------
# KEEP ONLY FEATURES USED BY THE FINAL MODEL
# ------------------------------------------------------------

new_house_model = new_house_model[
    selected_features
]


# ------------------------------------------------------------
# ADD CONSTANT
# ------------------------------------------------------------

new_house_model = sm.add_constant(
    new_house_model,
    has_constant="add"
)


# ------------------------------------------------------------
# MATCH THE MODEL'S COLUMN ORDER
# ------------------------------------------------------------

new_house_model = new_house_model[
    X_train_new.columns
]


# ------------------------------------------------------------
# MAKE THE PREDICTION
# ------------------------------------------------------------

predicted_rent = lm.predict(
    new_house_model
)


# ------------------------------------------------------------
# DISPLAY THE PROPERTY
# ------------------------------------------------------------

print("Example property:")

print("Floor area: 100 m²")
print("Bedrooms: 2")
print("Bathrooms: 2")
print("Category: Flats")
print("Condition: New")
print("Parking: yes")
print("Furnishing: Semi-Furnished")


# ------------------------------------------------------------
# DISPLAY PREDICTION
# ------------------------------------------------------------

print("\nEstimated monthly rent:")

print(
    f"GH₵{predicted_rent.iloc[0]:,.2f}"
)


# ============================================================
# 41. PROJECT CONCLUSION
# ============================================================

print("\n====================================================")
print("PROJECT COMPLETED")
print("====================================================")

print(
    f"Kumasi rental listings analysed: {len(df)}"
)

print(
    f"Final model R-squared: {r_squared:.4f}"
)

print(
    f"Example predicted monthly rent: "
    f"GH₵{predicted_rent.iloc[0]:,.2f}"
)

print("====================================================")