import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, auc, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cluster import KMeans
from imblearn.over_sampling import SMOTE


df = pd.read_csv(r'C:\Users\valen\OneDrive\Desktop\HR_comma_sep.csv')


missing = df.isnull().sum()
print("Missing values:\n", missing)
if missing.sum() == 0:
    print("There are no missing values in the dataset.")
else:
    print("There are missing values. Please handle them before proceeding.")


print("Duplicates:", df.duplicated().sum())
df = df.drop_duplicates().reset_index(drop=True) 


plt.figure(figsize=(10,8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.savefig('correlation_heatmap.png')
plt.show()
plt.close()


plt.figure()
sns.histplot(df['satisfaction_level'], kde=True)
plt.title('Employee Satisfaction Distribution')
plt.tight_layout()
plt.savefig('satisfaction_dist.png')
plt.show()
plt.close()


plt.figure()
sns.histplot(df['last_evaluation'], kde=True)
plt.title('Employee Evaluation Distribution')
plt.tight_layout()
plt.savefig('evaluation_dist.png')
plt.show()
plt.close()


plt.figure()
sns.histplot(df['average_montly_hours'], kde=True)
plt.title('Average Monthly Hours Distribution')
plt.tight_layout()
plt.savefig('monthly_hours_dist.png')
plt.show()
plt.close()


plt.figure(figsize=(8,5))
sns.countplot(x='number_project', hue='left', data=df)
plt.title('Project Count by Employee Turnover')
plt.xlabel('Number of Projects')
plt.ylabel('Number of Employees')
plt.legend(title='Left', labels=['Stayed', 'Left'])
plt.tight_layout()
plt.savefig('project_count_left.png')
plt.show()
plt.close()
print("\nBar Plot Explanation:")
print("The bar plot shows the distribution of project counts for employees who stayed and those who left. "
      "Employees with very low or very high project counts are more likely to leave, while those with moderate project counts tend to stay.")


left_df = df[df['left'] == 1][['satisfaction_level', 'last_evaluation']].reset_index(drop=True)
scaler = StandardScaler()
left_scaled = scaler.fit_transform(left_df)
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(left_scaled)
left_df['cluster'] = clusters


plt.figure(figsize=(8,6))
sns.scatterplot(data=left_df, x='satisfaction_level', y='last_evaluation', hue='cluster', palette='Set1')
plt.title('KMeans Clusters of Employees Who Left')
plt.tight_layout()
plt.savefig('kmeans_clusters_left.png')
plt.show()
plt.close()
print("\nKMeans Clustering Explanation:")
print("The scatter plot shows three clusters of employees who left, based on satisfaction and evaluation. "
      "Clusters may represent: (1) low satisfaction/low evaluation, (2) high evaluation/low satisfaction, (3) high satisfaction/high evaluation. "
      "This suggests different types of turnover risk.")


cat_cols = ['sales', 'salary']
num_cols = [col for col in df.columns if col not in cat_cols + ['left']]


scaler = StandardScaler()
df_num = pd.DataFrame(scaler.fit_transform(df[num_cols]), columns=num_cols).reset_index(drop=True)
df_cat = pd.get_dummies(df[cat_cols], drop_first=True).reset_index(drop=True)


X = pd.concat([df_num, df_cat], axis=1)
y = df['left'].reset_index(drop=True)


assert len(X) == len(y), f"X and y have different lengths: {len(X)} vs {len(y)}"


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=123)


smote = SMOTE(random_state=123)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)


models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=123),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=123),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=123)
}


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
results = {}


for name, model in models.items():
    print(f"\n--- {name} ---")
    y_pred = cross_val_predict(model, X_train_sm, y_train_sm, cv=skf)
    print(classification_report(y_train_sm, y_pred))
    model.fit(X_train_sm, y_train_sm)
    y_proba = model.predict_proba(X_test)[:,1]
    auc_score = roc_auc_score(y_test, y_proba)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    y_test_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred)
    recall = recall_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred)
    results[name] = {
        'model': model,
        'auc': auc_score,
        'fpr': fpr,
        'tpr': tpr,
        'confusion_matrix': cm,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    print(f"AUC: {auc_score:.3f}")
    print("Confusion Matrix:\n", cm)
    print(f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1-score: {f1:.3f}")


plt.figure(figsize=(8,6))
for name, res in results.items():
    plt.plot(res['fpr'], res['tpr'], label=f"{name} (AUC={res['auc']:.2f})")
plt.plot([0,1],[0,1],'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend()
plt.tight_layout()
plt.savefig('roc_curves.png')
plt.show()
plt.close()


best_model_name = max(results, key=lambda x: results[x]['auc'])
best_model = results[best_model_name]['model']
print(f"\nBest Model: {best_model_name}")


y_test_proba = best_model.predict_proba(X_test)[:,1]
zones = pd.cut(y_test_proba, bins=[-0.01,0.2,0.6,0.9,1.0], labels=['Safe Zone','Low-Risk Zone','Medium-Risk Zone','High-Risk Zone'])
zone_counts = pd.Series(zones).value_counts().sort_index()
print("\nEmployee Risk Zones in Test Set:")
print(zone_counts)


test_results = X_test.copy()
test_results['left'] = y_test.values
test_results['turnover_probability'] = y_test_proba
test_results['risk_zone'] = zones
test_results.to_csv('employee_turnover_risk_zones.csv', index=False)


print("\n--- Retention Strategy Suggestions ---")
print("""
Safe Zone (Green, <20%): Maintain engagement, recognize good performance.
Low-Risk Zone (Yellow, 20-60%): Monitor, offer growth opportunities, check for early signs of disengagement.
Medium-Risk Zone (Orange, 60-90%): Conduct stay interviews, address workload or satisfaction issues, consider incentives.
High-Risk Zone (Red, >90%): Immediate intervention, personalized retention plans, review compensation and work environment.
""")