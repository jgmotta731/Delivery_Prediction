# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, FunctionTransformer, LabelEncoder, PowerTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, ConfusionMatrixDisplay, auc
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import warnings

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

# Logistics Supply Chain Data Set
delivery = pd.read_csv('Logistics.csv')

# Diesel Prices Supplemental Data
fuel = pd.read_excel('Diesel_Prices.xlsx', sheet_name='Data 1', skiprows=2)
fuel.columns = ['date', 'diesel_price']

# Convert dates to datetime type
delivery['shipping_date'] = pd.to_datetime(delivery['shipping_date'], utc=True)
delivery['order_date'] = pd.to_datetime(delivery['order_date'], utc=True)
fuel['date'] = pd.to_datetime(fuel['date'])

# Create year-week columns for joining purposes
delivery['year_week'] = delivery['shipping_date'].dt.strftime('%Y-%U')
fuel['date'] = fuel['date'].dt.strftime('%Y-%U')

# Left join with the diesel prices df
merged_df = delivery.merge(fuel, how='left', left_on='year_week', right_on='date')
merged_df = merged_df.drop(columns=['year_week', 'date'])
merged_df = merged_df.sort_values('order_date') 

merged_df['label_display'] = merged_df['label'].map({-1: 'Early', 0: 'On Time', 1: 'Delayed'})
id_cols = ['category_id', 'customer_id', 'customer_zipcode',
           'department_id', 'order_customer_id', 'order_id', 
           'order_item_cardprod_id', 'order_item_id', 'product_card_id',
           'product_category_id', 'label', 'label_display']
print('Table 2: Summary Statistics of Numerical Columns')

# Plot Figures 1-4 in 2x2 grid
fig, axes = plt.subplots(2, 2, figsize=(17, 9))
fig.suptitle("Figures 1-4: Summary Visualizations", fontsize=16, y=0.95)

# Figure 1: Delivery Outcome
sns.countplot(data=merged_df, x='label_display', hue='label_display', ax=axes[0, 0])
axes[0, 0].set_title("Figure 1: Bar Plot of Delivery Outcome")
axes[0, 0].set_xlabel("Delivery Outcome")
axes[0, 0].set_ylabel("Frequency")
if axes[0, 0].get_legend():
    axes[0, 0].get_legend().remove()

# Figure 2: Shipping Mode By Delivery Outcome
sns.countplot(data=merged_df, x='shipping_mode', hue='label_display', ax=axes[1, 0])
axes[1, 0].set_title("Figure 2: Bar Plot of Shipping Mode By Delivery Outcome")
axes[1, 0].set_xlabel("Shipping Mode")
axes[1, 0].set_ylabel("Count")
axes[1, 0].tick_params(axis='x')
axes[1, 0].legend(title='Delivery Status')

# Figure 3: Skewness of Numerical Features
numerics = merged_df.drop(columns=id_cols).select_dtypes(include='number')
skew_df = numerics.apply(lambda x: x.skew()).sort_values(ascending=False)
sns.barplot(x=skew_df.values, y=skew_df.index, ax=axes[0, 1])
axes[0, 1].set_title("Figure 3: Bar Plot of Skewness of Numerical Features")
axes[0, 1].set_xlabel("Skewness")
axes[0, 1].set_ylabel("Feature")

# Figure 4: Destination Market By Delivery Status
sns.countplot(data=merged_df, x='market', hue='label_display', ax=axes[1, 1])
axes[1, 1].set_title("Figure 4: Bar Plot of Destination Market By Delivery Status")
axes[1, 1].set_xlabel("Market")
axes[1, 1].set_ylabel("Count")
axes[1, 1].legend(title='Delivery Status')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# Figure 5: Store Location By Delivery Status
fig, ax = plt.subplots(figsize=(16, 5))
fig.suptitle("Figure 5: Bar Plot of Store Location By Delivery Status", fontsize=16)
sns.countplot(data=merged_df, x='customer_state', hue='label_display', ax=ax)
ax.set_xlabel("Store Location")
ax.set_ylabel("Count")
ax.legend(title='Delivery Status')
plt.tight_layout()
plt.show()

# Figure 6: Correlation Heatmap Plot
fig, ax = plt.subplots(figsize=(17, 6))
fig.suptitle("Figure 6: Correlation Heatmap Plot", fontsize=16)
sns.heatmap(merged_df.drop(columns=id_cols).select_dtypes(include='number').corr(),
            annot=True, cmap='coolwarm', ax=ax)
plt.tight_layout()
plt.show()

# Count the occurrences of each class in 'label_display'
label_counts = merged_df['label_display'].value_counts()

# Identify majority and minority class sizes
majority_class = label_counts.max()
minority_class = label_counts.min()
imbalance_ratio = round(minority_class / majority_class, 3)

# Create DataFrame with imbalance ratio info
class_distribution = pd.DataFrame({
    'Count': label_counts
})
class_distribution['Imbalance vs Majority'] = round(label_counts / majority_class, 3)

print(class_distribution)
print(f"\nOverall Imbalance Ratio (Minority / Majority): {imbalance_ratio}")

# Check no. of rows where shipping date occurs before order date
print("Invalid rows:", (merged_df['shipping_date'] < merged_df['order_date']).sum())

# Extract date info from `order_date`
merged_df['order_dayofweek'] = merged_df['order_date'].dt.dayofweek
merged_df['order_month'] = merged_df['order_date'].dt.month
merged_df['order_year'] = merged_df['order_date'].dt.year
merged_df['is_weekend'] = merged_df['order_dayofweek'].isin([5, 6]).astype(int)

# Create shipping delay, flag invalid values, and make invalid values NaN
merged_df['days_between_order_and_ship'] = (merged_df['shipping_date'] - merged_df['order_date']).dt.days
merged_df['invalid_date_flow'] = (merged_df['shipping_date'] < merged_df['order_date']).astype(int)
merged_df.loc[merged_df['days_between_order_and_ship'] < 0, 'days_between_order_and_ship'] = np.nan

# Relabeling
column_relabel = {
    'profit_per_order': 'pre_discount_profit_per_order',
    'order_profit_per_order': 'post_discount_profit_per_order',
    'sales_per_customer': 'total_sales_per_customer',
    'order_item_product_price': 'item_price_before_discount',
    'order_item_total_amount': 'item_total_after_discount',
    'product_price': 'product_retail_price',
    'sales': 'total_sales',
    'order_item_discount': 'item_discount_amount',
    'order_item_discount_rate': 'item_discount_rate',
    'order_item_profit_ratio': 'item_profit_margin',
    'order_status': 'order_current_status',
    'order_state': 'order_delivery_state',
    'order_country': 'order_delivery_country',
    'order_city': 'order_delivery_city',
    'customer_country': 'customer_purchase_country',
    'customer_city': 'customer_purchase_city',
    'customer_state': 'store_location_state',
}
merged_df = merged_df.rename(columns=column_relabel)
print('Column Names After Relabeling:\n' + str(merged_df.columns))

# Columns with NaNs
na_counts = merged_df.isnull().sum()
print(f'Sum of NAs in Columns with at least 1 NA:\n{na_counts[na_counts > 0]}') # Show sum of NAs for columns with at least 1 NA

# Histogram 1: Days Between Order and Ship
fig, axes = plt.subplots(1, 2, figsize=(8, 3))
axes[0].hist(merged_df['days_between_order_and_ship'], bins=20, edgecolor='black')
axes[0].set_title("Days Between Order and Shipping Date")
axes[0].set_xlabel("Days Between Order and Ship")
axes[0].set_ylabel("Frequency")

# Histogram 2: Diesel Price
axes[1].hist(merged_df['diesel_price'], bins=10, edgecolor='black')
axes[1].set_title("Diesel Price Distribution")
axes[1].set_xlabel("Diesel Price")
axes[1].set_ylabel("Frequency")
fig.suptitle("Figure 7 & 8: Histograms of Order Processing and Diesel Prices", fontsize=14)
fig.tight_layout()
plt.show()

# Preprocessing
merged_df_copy = merged_df.copy() # Make a copy for merged_df to let XGBoost handle NAs
merged_df['store_location_state'] = merged_df['store_location_state'].replace('91732', 'CA')
merged_df['diesel_price'] = merged_df['diesel_price'].fillna(merged_df['diesel_price'].mean())
merged_df['days_between_order_and_ship'] = merged_df['days_between_order_and_ship'].fillna(merged_df['days_between_order_and_ship'].median())

# Custom log to ensure negatives don't cause errors
def safe_log1p(X):
    return np.log1p(np.nan_to_num(X)) + 1e-6

# Specify columns to transform
transform_cols = [
    'item_discount_amount', 'item_price_before_discount', 'product_retail_price',
    'total_sales_per_customer', 'total_sales', 'item_total_after_discount',
    'item_profit_margin', 'pre_discount_profit_per_order', 'post_discount_profit_per_order',
    'days_between_order_and_ship'
]

# Columns to transform via log/Yeo-Johnson based on presence of negatives
safe_log_cols = [col for col in transform_cols if (merged_df[col] >= 0).all()]
yeo_johnson_cols = [col for col in transform_cols if (merged_df[col] < 0).any()]

print('Columns to Log:', safe_log_cols)
print("Columns for Yeo-Johnson Transform:", yeo_johnson_cols)

# Column transformer containing preprocessing
ct = ColumnTransformer(
    [("log", FunctionTransformer(safe_log1p, feature_names_out='one-to-one'),
      safe_log_cols),
     ("yeojohnson", PowerTransformer(method='yeo-johnson'), yeo_johnson_cols),
     ("ordinal", OrdinalEncoder(categories=[['Standard Class', 'Second Class', 'First Class', 'Same Day']],
                                handle_unknown='use_encoded_value', unknown_value=-1), 
      ['shipping_mode']),     
     ("onehot", OneHotEncoder(drop='first'),
      ['payment_type', 'customer_purchase_country', 'customer_segment', 'department_name',
       'market', 'order_region', 'order_current_status', 'order_dayofweek', 'order_month'])], 
    remainder='passthrough', verbose_feature_names_out=False
)

# Label Encode delivery outcome
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(merged_df['label'])

# Select only relevant and not redundant non-ID columns
X = merged_df.iloc[:, [0, 1, 2, 6, 8, 12, 13, 14, 15, 22, 
                       23, 25, 26, 27, 28, 29, 30, 31, 33, 
                       37, 39, 41, 43, 44, 45, 46, 47, 48]]

# Stratified 70/30 Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, train_size=.7, random_state=1, stratify=y)

# Fit transformer on training data
ct.fit(X_train)

# Get total number of output features
num_transformed_features = len(ct.get_feature_names_out())
print(f"Number of predictors after ColumnTransformer: {num_transformed_features}")

# Linearity of Log-Odds assumption check
X_ct = ct.fit_transform(X_train)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_ct)
model = LogisticRegression(max_iter=1000)
model.fit(X_scaled, y_train)
probs = model.predict_proba(X_scaled)[:, 1]
logit = np.log(probs / (1 - probs))

feature_names = ct.get_feature_names_out()
selected_features = ['item_profit_margin', 'total_sales_per_customer', 'product_retail_price']
df_plot = pd.DataFrame({
    feat: X_scaled[:, [j for j, name in enumerate(feature_names) if feat in name][0]]
    for feat in selected_features
})
df_plot['logit'] = logit
df_plot['label'] = y_train

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("Figure 9: Linearity Check of Logit vs Selected Predictors (Scaled)", fontsize=14)
for ax, feat in zip(axes, selected_features):
    sns.scatterplot(data=df_plot, x=feat, y='logit', hue='label', ax=ax, alpha=0.6)
    ax.set_title(f"{feat.replace('_', ' ').title()}")
    ax.set_xlabel(f"{feat.replace('_', ' ').title()}")
    ax.set_ylabel("Logit")
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# Custom Multinomial Model Tuning and Evaluation Function
def evaluate_model(display_name, y_test, y_pred):
    print(f"\n{display_name}")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred, digits=3))

def run_grid_search_pipeline(display_name, pipeline, param_grid, X_train, y_train, X_test, y_test):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    grid = GridSearchCV(pipeline, param_grid=param_grid, cv=cv, n_jobs=-1, verbose=1, error_score='raise', scoring='recall_macro')
    grid.fit(X_train, y_train)
    print(f"\n{display_name} Best Params: {grid.best_params_}")
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)
    evaluate_model(display_name, y_test, y_pred)
    return best_model

# Softmax Regression
lr_pipeline = Pipeline([("preprocess", ct),
                        ("impute", SimpleImputer(strategy="mean")),
                        ("scale", StandardScaler()),
                        ("lr", LogisticRegression(random_state=1, solver='lbfgs', max_iter=1000, n_jobs=-1))])
lr_param_grid = {'lr__C': np.logspace(-4, 4, 50)}
best_lr = run_grid_search_pipeline("Softmax Regression", lr_pipeline, lr_param_grid, X_train, y_train, X_test, y_test)

# Random Forest
rf_pipeline = Pipeline([("preprocess", ct),
                        ("impute", SimpleImputer(strategy="mean")),
                        ("scale", StandardScaler()),
                        ("rf", RandomForestClassifier(random_state=1, n_jobs=-1))])
rf_param_grid = {'rf__n_estimators': [25, 50, 75, 100], 'rf__max_depth': [15, 20, 25], 'rf__min_samples_split': [2, 3, 4]}
best_rf = run_grid_search_pipeline("Random Forest", rf_pipeline, rf_param_grid, X_train, y_train, X_test, y_test)

# XGBoost (with NaNs left alone)
xgb_pipeline = Pipeline([("preprocess", ct), 
                         ("scale", StandardScaler()), 
                         ("xgb", XGBClassifier(objective='multi:softprob', random_state=1, n_jobs=-1, eval_metric='mlogloss'))])
xgb_param_grid = {'xgb__n_estimators': [30, 55, 90, ], 'xgb__max_depth': [4, 5, 6], 'xgb__learning_rate': [0.005, 0.01, 0.05], 'xgb__subsample': [0.7, 0.8, 0.9]}
best_xgb = run_grid_search_pipeline("XGBoost", xgb_pipeline, xgb_param_grid, X_train, y_train, X_test, y_test)

# Custom Binary Model Tuning and Evaluation Functions
def evaluate_model_binary(display_name, y_test, y_pred):
    print(f"\n{display_name}")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred, digits=3))
    
def run_grid_search_pipeline_binary(display_name, pipeline, param_grid, X_train, y_train, X_test, y_test, scoring='roc_auc'):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    grid = GridSearchCV(pipeline, param_grid=param_grid, cv=cv, scoring=scoring, n_jobs=-1, verbose=1, error_score='raise')
    grid.fit(X_train, y_train)
    print(f"\n{display_name} Best Params: {grid.best_params_}")
    y_pred = grid.best_estimator_.predict(X_test)
    evaluate_model_binary(display_name, y_test, y_pred)
    return grid.best_estimator_

# XGBoost on Binary Outcome: Delayed (1) vs Not Delayed (0)
y = (merged_df_copy['label'] == 1).astype(int).values
X = merged_df_copy.iloc[:, [0, 1, 2, 6, 8, 12, 13, 14, 15, 22, 
                       23, 25, 26, 27, 28, 29, 30, 31, 33, 
                       37, 39, 41, 43, 44, 45, 46, 47, 48]]
X_train_binary, X_test_binary, y_train_binary, y_test_binary = train_test_split(X, y, test_size=.3, train_size=.7, random_state=1, stratify=y)
xgb_pipeline = Pipeline([("preprocess", ct), ("scale", StandardScaler()), ("xgb", XGBClassifier(objective='binary:logistic', random_state=1, n_jobs=-1, eval_metric='logloss'))])
xgb_param_grid = {'xgb__n_estimators': [30, 55, 90], 'xgb__max_depth': [4, 6], 'xgb__learning_rate': [0.01, 0.05], 'xgb__subsample': [0.8, 0.9], 'xgb__colsample_bytree': [0.8, 0.9, 1]}
best_xgb_binary = run_grid_search_pipeline_binary("XGBoost (Binary Classification)", xgb_pipeline, xgb_param_grid, X_train_binary, y_train_binary, X_test_binary, y_test_binary)

# Confusion Matrices Comparison Display
models = [("Softmax Regression", best_lr), ("Random Forest", best_rf), ("XGBoost (Multiclass)", best_xgb), ("XGBoost (Binary)", best_xgb_binary)]
fig, axes = plt.subplots(1, len(models), figsize=(4 * len(models), 4))
for ax, (name, model) in zip(axes, models):
    if "Binary" in name:
        y_true = y_test_binary; y_pred = model.predict(X_test_binary)
    else:
        y_true = y_test; y_pred = model.predict(X_test)
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, cmap="Blues", ax=ax); ax.set_title(name); ax.grid(False)
plt.tight_layout(); plt.subplots_adjust(top=0.8); plt.suptitle('Figure 10: Model Confusion Matrices', fontsize=14); plt.show()

# ROC Curve of Best Model (XGBoost Binary)
y_proba = best_xgb_binary.predict_proba(X_test_binary)[:, 1]
fpr, tpr, _ = roc_curve(y_test_binary, y_proba)
auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6)); plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
plt.plot([0, 1], [0, 1], 'k--', label='Random Baseline'); plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
plt.title('Figure 11: ROC Curve of XGBoost Binary Model', fontsize=14)
plt.legend(loc='lower right'); plt.grid(True); plt.show()

# Get Feature Names and Importances
feature_names = ct.get_feature_names_out()
xgb_model = best_xgb_binary.named_steps['xgb']
importances = xgb_model.feature_importances_

# Sort by top 30 importances
indices = np.argsort(importances)[::-1]
sorted_features = np.array(feature_names)[indices]
sorted_importances = importances[indices]
top_n = 30
top_features = sorted_features[:top_n]
top_importances = sorted_importances[:top_n]
top_features_reversed = top_features[:top_n][::-1]
top_importances_reversed = top_importances[:top_n][::-1]

# Feature Importance Plot
plt.figure(figsize=(10, 7))
sns.set(style="whitegrid")
bars = plt.barh(range(top_n), top_importances_reversed, align='center', color=sns.color_palette("Blues", top_n))
plt.yticks(range(top_n), top_features_reversed, fontsize=10)
plt.xlabel("Importance Score", fontsize=12)
plt.title("Figure 12: Top 30 Feature Importances from XGBoost (Binary Classification)", fontsize=14)
for i, bar in enumerate(bars):
    plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
             f"{top_importances_reversed[i]:.3f}", va='center', fontsize=9)
plt.tight_layout()
plt.show()