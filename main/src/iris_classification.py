import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
)

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, roc_auc_score
)

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
)

from sklearn.naive_bayes import GaussianNB

warnings.filterwarnings('ignore')


BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR  = os.path.join(BASE, 'data')
PLOTS_DIR = os.path.join(BASE, 'plots')
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

print("=" * 60)
print("IRIS FLOWER CLASSIFICATION — CodSoft Task 3")
print("=" * 60)

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target
df['species_name'] = df['species'].map({
    0: 'Iris-setosa', 1: 'Iris-versicolor', 2: 'Iris-virginica'

})


df.to_csv(os.path.join(DATA_DIR, 'iris.csv'), index=False)
print(f"\n✔ Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"  Classes : {df['species_name'].unique().tolist()}")
print(f"  Balance : \n{df['species_name'].value_counts().to_string()}")

FEATURES = iris.feature_names
COLORS   = ['#4C72B0', '#55A868', '#C44E52']
SPECIES  = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

print("\n[EDA] Generating pairplot …")
pair_df = df[FEATURES + ['species_name']].copy()
pair_df.columns = ['Sepal Len', 'Sepal Wid', 'Petal Len', 'Petal Wid', 'Species']
palette = dict(zip(SPECIES, COLORS))
g = sns.pairplot(pair_df, hue='Species', palette=palette,
                 diag_kind='kde', plot_kws={'alpha': 0.7})
g.fig.suptitle('Iris Feature Pairplot', y=1.02, fontsize=14, fontweight='bold')
g.fig.savefig(os.path.join(PLOTS_DIR, '01_pairplot.png'), dpi=120, bbox_inches='tight')
plt.close()
print("  ✔ 01_pairplot.png")

print("[EDA] Generating feature boxplots …")
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('Feature Distributions by Species', fontsize=14, fontweight='bold')
for ax, feat in zip(axes.flatten(), FEATURES):
    data_by_species = [df.loc[df['species'] == i, feat].values for i in range(3)]
    bp = ax.boxplot(data_by_species, patch_artist=True, notch=True,
                    medianprops=dict(color='black', linewidth=2))
    for patch, color in zip(bp['boxes'], COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_xticklabels(['Setosa', 'Versicolor', 'Virginica'], fontsize=9)
    ax.set_title(feat.replace('(cm)', '').strip(), fontsize=11)
    ax.set_ylabel('cm')
    ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '02_boxplots.png'), dpi=120, bbox_inches='tight')
plt.close()
print("  ✔ 02_boxplots.png")

print("[EDA] Generating correlation heatmap …")
fig, ax = plt.subplots(figsize=(7, 5))
corr = df[FEATURES].corr()
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, linewidths=0.5, ax=ax,
            xticklabels=[f.replace(' (cm)', '') for f in FEATURES],
            yticklabels=[f.replace(' (cm)', '') for f in FEATURES])
ax.set_title('Feature Correlation Matrix', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '03_correlation.png'), dpi=120, bbox_inches='tight')
plt.close()
print("  ✔ 03_correlation.png")
print("\n[Feature Engineering]")

def engineer_features(frame):
    f = frame.copy()
    feat_cols = iris.feature_names
    sl, sw, pl, pw = (f[c] for c in feat_cols)

    # Ratios
    f['petal_ratio']    = pl / (pw + 1e-6)
    f['sepal_ratio']    = sl / (sw + 1e-6)
    f['petal_area']     = pl * pw
    f['sepal_area']     = sl * sw

    # Differences
    f['len_diff']       = sl - pl
    f['wid_diff']       = sw - pw

    f['petal_sepal_area_ratio'] = f['petal_area'] / (f['sepal_area'] + 1e-6)

    return f

df_eng = engineer_features(df)
ENG_FEATURES = [c for c in df_eng.columns
                if c not in ('species', 'species_name')]
print(f"  Features after engineering: {len(ENG_FEATURES)}")
print(f"  New features: petal_ratio, sepal_ratio, petal_area, sepal_area,")
print(f"                len_diff, wid_diff, petal_sepal_area_ratio")


# Train / Test Split

X = df_eng[ENG_FEATURES].values
y = df_eng['species'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n  Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")


# Model Training & Evaluation
print("\n[Model Training — 8 algorithms]")

models = {
    'Logistic Regression':     LogisticRegression(max_iter=1000, random_state=42),
    'K-Nearest Neighbors':     KNeighborsClassifier(),
    'Support Vector Machine':  SVC(probability=True, random_state=42),
    'Decision Tree':           DecisionTreeClassifier(random_state=42),
    'Random Forest':           RandomForestClassifier(n_estimators=100, random_state=42),
    'Extra Trees':             ExtraTreesClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting':       GradientBoostingClassifier(n_estimators=100, random_state=42),
    'Naive Bayes':             GaussianNB(),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

results = {}
for name, model in models.items():
    cv_scores = cross_val_score(model, X_train_sc, y_train, cv=cv, scoring='accuracy')
    model.fit(X_train_sc, y_train)
    y_pred  = model.predict(X_test_sc)
    y_proba = model.predict_proba(X_test_sc)
    acc     = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro')
    results[name] = {
        'model': model,
        'cv_mean': cv_scores.mean(),
        'cv_std':  cv_scores.std(),
        'test_acc': acc,
        'roc_auc':  roc_auc,
        'y_pred':   y_pred,
        'y_proba':  y_proba,
    }
    print(f"  {name:<26} CV={cv_scores.mean():.4f}±{cv_scores.std():.4f}  "
          f"Test Acc={acc:.4f}  ROC-AUC={roc_auc:.4f}")

best_name = max(results, key=lambda n: results[n]['roc_auc'])
best = results[best_name]
print(f"\n★ Best Model: {best_name}  (ROC-AUC={best['roc_auc']:.4f})")
print(f"\n[Hyperparameter Tuning — {best_name}]")

param_grids = {
    'Support Vector Machine': {
        'C': [0.1, 1, 10, 100], 'gamma': ['scale', 'auto', 0.01, 0.1],
        'kernel': ['rbf', 'poly']
    },
    'Random Forest': {
        'n_estimators': [100, 200], 'max_depth': [None, 5, 10],
        'min_samples_split': [2, 5]
    },
    'Gradient Boosting': {
        'n_estimators': [100, 200], 'learning_rate': [0.05, 0.1, 0.2],
        'max_depth': [3, 5]
    },
    'Extra Trees': {
        'n_estimators': [100, 200], 'max_depth': [None, 5, 10]
    },
    'K-Nearest Neighbors': {
        'n_neighbors': [3, 5, 7, 9], 'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    },
    'Logistic Regression': {
        'C': [0.01, 0.1, 1, 10], 'solver': ['lbfgs', 'saga']
    },
}


tuned_model  = best['model']
tuned_y_pred = best['y_pred']

if best_name in param_grids:
    grid = GridSearchCV(
        models[best_name], param_grids[best_name],
        cv=cv, scoring='accuracy', n_jobs=-1
    )
    grid.fit(X_train_sc, y_train)
    tuned_model  = grid.best_estimator_
    tuned_y_pred = tuned_model.predict(X_test_sc)
    tuned_acc    = accuracy_score(y_test, tuned_y_pred)
    print(f"  Best params : {grid.best_params_}")
    print(f"  Tuned Acc   : {tuned_acc:.4f}")
else:
    print(f"  No grid defined for {best_name}; using default fit.")


# 7. Visualisations — Results

print("\n[Plots — Results]")

print("[Plot] Model comparison …")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Model Comparison', fontsize=14, fontweight='bold')

names = list(results.keys())
cv_means = [results[n]['cv_mean'] for n in names]
cv_stds  = [results[n]['cv_std']  for n in names]
test_acc = [results[n]['test_acc'] for n in names]
roc_aucs = [results[n]['roc_auc'] for n in names]

bar_colors = ['#C44E52' if n == best_name else '#4C72B0' for n in names]

ax = axes[0]
bars = ax.barh(names, cv_means, xerr=cv_stds, color=bar_colors,
               alpha=0.8, capsize=4, edgecolor='white')
ax.set_xlabel('CV Accuracy (5-fold)', fontsize=11)
ax.set_title('Cross-Validation Accuracy', fontsize=12)
ax.set_xlim(0.8, 1.02)
ax.axvline(0.95, color='grey', linestyle='--', alpha=0.5, label='0.95 ref')
ax.legend(fontsize=9)
for i, v in enumerate(cv_means):
    ax.text(v + 0.002, i, f'{v:.3f}', va='center', fontsize=8)

ax2 = axes[1]
ax2.barh(names, roc_aucs, color=bar_colors, alpha=0.8, edgecolor='white')
ax2.set_xlabel('ROC-AUC (macro-OVR)', fontsize=11)
ax2.set_title('Test ROC-AUC', fontsize=12)
ax2.set_xlim(0.8, 1.02)
for i, v in enumerate(roc_aucs):
    ax2.text(v + 0.002, i, f'{v:.3f}', va='center', fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '04_model_comparison.png'), dpi=120, bbox_inches='tight')
plt.close()
print("  ✔ 04_model_comparison.png")


print("[Plot] Confusion matrix …")
cm = confusion_matrix(y_test, tuned_y_pred)
fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Setosa', 'Versicolor', 'Virginica'],
            yticklabels=['Setosa', 'Versicolor', 'Virginica'],
            linewidths=0.5)
ax.set_xlabel('Predicted', fontsize=12)
ax.set_ylabel('Actual', fontsize=12)
ax.set_title(f'Confusion Matrix — {best_name}', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '05_confusion_matrix.png'), dpi=120, bbox_inches='tight')
plt.close()
print("  ✔ 05_confusion_matrix.png")

print("[Plot] Feature importance …")
rf = results['Random Forest']['model']
importances = rf.feature_importances_
feat_imp_df = pd.DataFrame({
    'Feature': ENG_FEATURES,
    'Importance': importances
}).sort_values('Importance', ascending=True)

fig, ax = plt.subplots(figsize=(9, 7))
colors_fi = ['#C44E52' if imp >= feat_imp_df['Importance'].quantile(0.75)
             else '#4C72B0' for imp in feat_imp_df['Importance']]
ax.barh(feat_imp_df['Feature'], feat_imp_df['Importance'],
        color=colors_fi, alpha=0.8, edgecolor='white')
ax.set_xlabel('Feature Importance (Gini)', fontsize=11)
ax.set_title('Feature Importances — Random Forest', fontsize=13, fontweight='bold')
ax.axvline(feat_imp_df['Importance'].mean(), color='grey',
           linestyle='--', alpha=0.6, label='Mean')
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '06_feature_importance.png'), dpi=120, bbox_inches='tight')
plt.close()
print("  ✔ 06_feature_importance.png")
print("[Plot] Decision boundary (2D) …")
feat_idx = [2, 3]                                              # petal length, petal width
feat_names_2d = [FEATURES[i].replace(' (cm)', '') for i in feat_idx]

X2 = df_eng[FEATURES].values[:, feat_idx]
scaler2 = StandardScaler()
X2_sc = scaler2.fit_transform(X2)

svm2 = SVC(probability=True, C=10, gamma='scale', random_state=42)
svm2.fit(X2_sc, y)

h = 0.02
x_min, x_max = X2_sc[:, 0].min() - 0.5, X2_sc[:, 0].max() + 0.5
y_min, y_max = X2_sc[:, 1].min() - 0.5, X2_sc[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Z = svm2.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

fig, ax = plt.subplots(figsize=(9, 6))
ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.get_cmap('Set1', 3))
for cls, color, label in zip([0, 1, 2], COLORS, SPECIES):
    mask = y == cls
    ax.scatter(X2_sc[mask, 0], X2_sc[mask, 1],
               c=color, label=label, edgecolors='k', linewidths=0.5, s=60)
ax.set_xlabel(f'{feat_names_2d[0]} (scaled)', fontsize=11)
ax.set_ylabel(f'{feat_names_2d[1]} (scaled)', fontsize=11)
ax.set_title('SVM Decision Boundary (Petal Features)', fontsize=13, fontweight='bold')
ax.legend(loc='upper left', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '07_decision_boundary.png'), dpi=120, bbox_inches='tight')
plt.close()
print("  ✔ 07_decision_boundary.png")


print("[Plot] Learning curve …")
from sklearn.model_selection import learning_curve

train_sizes, train_scores, val_scores = learning_curve(
    tuned_model, X_train_sc, y_train,
    cv=cv, scoring='accuracy',
    train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1
)

fig, ax = plt.subplots(figsize=(8, 5))
ax.fill_between(train_sizes,
                train_scores.mean(1) - train_scores.std(1),
                train_scores.mean(1) + train_scores.std(1), alpha=0.15, color='#4C72B0')
ax.fill_between(train_sizes,
                val_scores.mean(1) - val_scores.std(1),
                val_scores.mean(1) + val_scores.std(1), alpha=0.15, color='#C44E52')
ax.plot(train_sizes, train_scores.mean(1), 'o-', color='#4C72B0', label='Train Score')
ax.plot(train_sizes, val_scores.mean(1),   's-', color='#C44E52', label='CV Score')
ax.set_xlabel('Training Set Size', fontsize=11)
ax.set_ylabel('Accuracy', fontsize=11)
ax.set_title(f'Learning Curve — {best_name}', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '08_learning_curve.png'), dpi=120, bbox_inches='tight')
plt.close()
print("  ✔ 08_learning_curve.png")

print("\n" + "=" * 60)
print(f"CLASSIFICATION REPORT — {best_name}")
print("=" * 60)
print(classification_report(
    y_test, tuned_y_pred,
    target_names=['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
))

print("=" * 60)
print("SAMPLE PREDICTIONS (10 test samples)")
print("=" * 60)

species_map = {0: 'Iris-setosa', 1: 'Iris-versicolor', 2: 'Iris-virginica'}
sample_df = pd.DataFrame(X_test[:10], columns=ENG_FEATURES)
sample_df['Actual']    = [species_map[y] for y in y_test[:10]]
sample_df['Predicted'] = [species_map[y] for y in tuned_y_pred[:10]]
sample_df['Correct']   = sample_df['Actual'] == sample_df['Predicted']

display_cols = list(iris.feature_names) + ['Actual', 'Predicted', 'Correct']
print(sample_df[display_cols].to_string(index=False))
print(f"\nAccuracy on these 10 samples: "
      f"{sample_df['Correct'].sum()}/10")

summary_lines = [
    "IRIS FLOWER CLASSIFICATION — RESULTS SUMMARY",
    "=" * 50,
    f"Best Model   : {best_name}",
    f"CV Accuracy  : {best['cv_mean']:.4f} ± {best['cv_std']:.4f}",
    f"Test Accuracy: {accuracy_score(y_test, tuned_y_pred):.4f}",
    f"ROC-AUC      : {best['roc_auc']:.4f}",
    "",
    "All Models:",
]
for n in names:
    summary_lines.append(
        f"  {n:<26} CV={results[n]['cv_mean']:.4f}  "
        f"Acc={results[n]['test_acc']:.4f}  AUC={results[n]['roc_auc']:.4f}"
    )


with open(os.path.join(BASE, 'results_summary.txt'), 'w') as f:
    f.write('\n'.join(summary_lines))


print("\n✔ results_summary.txt saved")
print("\n✔ All done! Check the /plots directory for visualisations.")
