# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('ufc_df.csv')
print(df.head(10))
# %%
print(df.isnull().sum())
#fight가 object이지만 필요 없는 변수라 제거할 예정
print(df.dtypes)
# %%
import platform
import matplotlib.pyplot as plt

if platform.system() == 'Windows':
    plt.rc('font', family='Malgun Gothic')
elif platform.system() == 'Dawin':
    plt.rc('font', family='AppleGothic')
else:
    plt.rc('font', family='NanumGothic')
    
plt.rcParams['axes.unicode_minus'] = False

#value_counts 시각화
df['ko여부'].value_counts().plot(kind='bar')
plt.title('ko여부 분포')
plt.xlabel('ko여부')
plt.ylabel('count')
plt.xticks(rotation=0)
plt.show()

df['ko여부'].value_counts()
# %%
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

#데이터 100개씩 
ko_100 = df[df['ko여부'] == 1].sample(100, random_state=42) 

nko_100 = df[df['ko여부'] == 0].sample(100, random_state=42)

df_200 = pd.concat([ko_100, nko_100]).sample(frac=1, random_state=42)

print(df_200['ko여부'].value_counts())
print(df_200.shape)
# %%
from sklearn.model_selection import train_test_split

X = df_200.drop(columns=['ko여부', 'fight']) #경기 label 필요없을 것 같아 제거
y = df_200['ko여부']

X_train_valid, X_test, y_train_valid, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=1,
    stratify=y
)

X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_valid, y_train_valid,
    test_size=0.25,
    random_state=1,
    stratify=y_train_valid
)

#train:valid:test = 60:20:20
print('Train: ', X_train.shape)
print('Valid: ', X_valid.shape)
print('Test: ', X_test.shape)
# %%
#RandomForest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=4,
    min_samples_leaf=5,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

y_train_pred = rf.predict(X_train)
y_valid_pred = rf.predict(X_valid)

print(classification_report(y_valid, y_valid_pred, digits=3))

print("train")
print("accuracy: ", accuracy_score(y_train, y_train_pred))
print("f1-score: ", f1_score(y_train, y_train_pred))

print("valid")
print("accuracy: ", accuracy_score(y_valid, y_valid_pred))
print("f1-score: ", f1_score(y_valid, y_valid_pred))

# %%
#변수 중요도 파악
importances = rf.feature_importances_
feature_names = X_train.columns

imp_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
})

imp_df = imp_df.sort_values('importance', ascending=False)

print(imp_df.head(10))

#중요도 시각화
features_all = imp_df.sort_values(by='importance', ascending=True)


plt.figure(figsize=(8, 8))
plt.barh(features_all['feature'], features_all['importance'])
plt.gca().invert_yaxis()

plt.xlabel("Importance")
plt.title("변수 중요도", fontsize=16)
plt.legend(["importance"], fontsize=14)

plt.yticks(fontsize=9)
plt.xticks(fontsize=12)

plt.subplots_adjust(left=0.32)
plt.tight_layout()
plt.show()
# %%
#영향력 큰 변수만 남기기
selected_df = imp_df.query("importance >= 0.025").sort_values("importance")

df_plot = selected_df.copy()

plt.figure(figsize=(8, 5))
plt.barh(df_plot['feature'], df_plot['importance'])
plt.gca().invert_yaxis()

plt.xlabel("Importance")
plt.title("변수 중요도 (0.025 이상)")
plt.legend(["importance"])
plt.tight_layout()
plt.show()
# %%
#가설검증에서 나온 변수, 모델에서 나온 변수 교집합, 합집합
hypo_vars = ['winner_sig_str_succ', 'winner_head_succ', 'winner_distance_succ', 'winner_body_att']

rf_vars = ['final_round', 'winner_clinch_succ', 'winner_clinch_att', 'loser_sig_str_pct', 'loser_td_attempt_att', 'winner_distance_succ']

common_vars = set(hypo_vars) & set(rf_vars)
union_vars = set(hypo_vars) | set(rf_vars)

print("교집합: ", list(common_vars))
print("개수: ", len(common_vars))

print("합집합: ", list(union_vars))
print("개수: ", len(union_vars))
# %%
union_vars = list(union_vars)
union_df = imp_df[imp_df['feature'].isin(union_vars)].sort_values(by='importance', ascending=True)

union_plot = union_df.copy()

plt.figure(figsize=(8,5))
plt.barh(union_plot['feature'], union_plot['importance'])
plt.gca().invert_yaxis()

plt.xlabel("Importance")
plt.title("가설검증 변수 & 모델에서 나온 중요도 변수")
plt.legend(["importance"])
plt.tight_layout()
plt.show()

# %%
#변수 간 상호작용 확인
plt.figure(figsize=(12, 10))

sns.heatmap(
    df[union_vars].corr(),
    annot=True, 
    fmt=".2f",
    cmap='coolwarm',
    linewidths=0.4,
    cbar=True
    )
plt.title("합집합 변수들의 상관관계", fontsize=16)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(fontsize=10)

plt.tight_layout()
plt.show()
# %%
remove_vars = ['loser_sig_str_pct', 'loser_td_attempt_att']

union_df = imp_df[
    imp_df['feature'].isin(union_vars) &
    ~imp_df['feature'].isin(remove_vars)
]
union_cols = union_df['feature'].tolist()

print("union_cols: ",union_cols)
# %%
print(df[union_cols + ['ko여부']].isnull().sum())
print(df[union_cols + ['ko여부']].dtypes)

# %%
#union_cols 이상치 점검
plt.figure(figsize=(12,6))
sns.boxplot(data=df[union_cols])
plt.xticks(rotation=0)
plt.title("Boxplot - union_cols 이상치 점검")
plt.show()

Q1 = df[union_cols].quantile(0.25)
Q3 = df[union_cols].quantile(0.75)
IQR = Q3 - Q1

outlier_mask = (df[union_cols] < (Q1 - 1.5 * IQR)) | (df[union_cols] > (Q3 + 1.5 * IQR))
outlier_counts = outlier_mask.sum().sort_values(ascending=False)

print("변수별 이상치 개수")
print(outlier_counts)
# %%
#다중공선성 확인
corr_matrix = df[union_cols].corr().abs()

upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

less_high_corr_pairs = [(row, col, upper.loc[row, col])
              for row in upper.index
              for col in upper.columns
              if upper.loc[row, col] > 0.8]

high_corr_pairs = [(row, col, upper.loc[row, col])
              for row in upper.index
              for col in upper.columns
              if upper.loc[row, col] > 0.9]

print("상관 0.8 이상 변수 쌍")
for p in less_high_corr_pairs:
    print(p)
    
print("상관 0.9 이상 변수 쌍")
for i in high_corr_pairs:
    print(i)
# %%
final_cols = union_cols.copy()
final_cols.remove('winner_clinch_att')
print(final_cols)

plt.figure(figsize=(10,8))
sns.heatmap(
    df[final_cols].corr(),
    annot=True,
    cmap='coolwarm',
    vmin=-1,
    vmax=1)


plt.title("final_cols(최종 변수) 상관관계 히트맵", fontsize=14)
plt.xticks(rotation=45, fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.show()
# %%
#최종 변수로 모델(randomforest)
X = df_200[final_cols]
y = df_200['ko여부']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=1,
    stratify=y
)
#적은 데이터에서는 CV 사용이 더 낫기 때문에 valid X

print("Train: ", X_train.shape)
print("Test: ", X_test.shape)
# %%
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import make_scorer
rf_final = RandomForestClassifier(
    n_estimators=200,
    max_depth=4,
    min_samples_split=5,
    min_samples_leaf=5,
    random_state=1,
    n_jobs=-1
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

f1_scorer = make_scorer(f1_score)

cv_scores = cross_val_score(
    rf_final,
    X_train_valid,
    y_train_valid,
    cv=cv,
    scoring=f1_scorer
)

print("각 Fold F1: ", cv_scores)
print("평균 F1: ", round(cv_scores.mean(), 4))
print("표준편차: ", round(cv_scores.std(), 4))
# %%
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import log_loss

train_proba = rf_final.predict_proba(X_train)
valid_proba = cross_val_predict(rf_final, X_train, y_train, cv=cv, method="predict_proba")

train_loss = log_loss(y_train, train_proba)
valid_loss = log_loss(y_train, valid_proba)

print("Train loss: ", train_loss)
print("CV loss: ", valid_loss)

# %%
from sklearn.model_selection import learning_curve

train_sizes, train_scores, valid_scores = learning_curve(
    rf_final,
    X_train,
    y_train,
    cv=cv,
    scoring="neg_log_loss",         
    train_sizes=np.linspace(0.2, 1.0, 7),
    shuffle=True,
    random_state=1
)

train_scores_mean = train_scores.mean(axis=1)
valid_scores_mean = valid_scores.mean(axis=1)

train_loss = -train_scores_mean
valid_loss = -valid_scores_mean

plt.figure(figsize=(8, 5))
plt.plot(train_sizes, train_loss, "o-", label="Train Loss")
plt.plot(train_sizes, valid_loss, "o-", label="Validation Loss (CV)")
plt.xlabel("Training Samples")
plt.ylabel("Log Loss")
plt.title("Learning Curve (Log Loss)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
#test로 평가
from sklearn.metrics import (
    accuracy_score, precision_score, 
    recall_score, classification_report,
    confusion_matrix
)

y_test_pred = rf_final.predict(X_test)

print("Test 성능")
print("F1-score: ", round(f1_score(y_test, y_test_pred), 4))
print("Accuracy: ", round(accuracy_score(y_test, y_test_pred), 4))
print("Precision: ", round(precision_score(y_test, y_test_pred), 4))
print("Recall: ", round(recall_score(y_test, y_test_pred), 4))

print("\nClassification Report")
print(classification_report(y_test, y_test_pred))

cm = confusion_matrix(y_test, y_test_pred)
cm_df = pd.DataFrame(
    cm,
    index=['Actual_0 (Non-ko)', 'Actual_1 (KO)'],
    columns=['Predict_0 (Non-ko)', 'Predict_1 (KO)']
)
print("Confusion Matrix")
print(cm_df)
# %%
#누적 이득 차트
from sklearn.metrics import roc_curve, roc_auc_score

def plot_cumulative_gain(y_true, y_prob, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
        
    data = sorted(zip(y_true, y_prob), key=lambda x: x[1], reverse=True)
    total = len(y_true)
    total_positives = sum(y_true)
    
    x = np.linspace(0, 1, total)
    
    y = np.cumsum([y for y, _ in data]) / total_positives
    
    ax.plot(x, y, label='Model', color='blue')
    ax.plot([0,1], [0,1], '--', color='gray', label='Baseline')
    ax.set_title('누적 이득 차트(Cumulative Gain)')
    ax.set_xlabel('전체 데이터 비율)')
    ax.set_ylabel('정답 클래스 누적 비율)')
    ax.legend()
    ax.grid(True)
    return ax

#리프트 차트
def plot_lift_chart(y_true, y_prob, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
        
    data = sorted(zip(y_true, y_prob), key=lambda x: x[1], reverse=True)
    total = len(y_true)
    total_positives = sum(y_true)
    
    deciles = np.linspace(0.1, 1.0, 10)
    lift_values = []
    
    for d in deciles:
        cutoff = int(total * d)
        top = data[:cutoff]
        
        positives_in_top = sum([y for y, _ in top])
        lift = (positives_in_top / cutoff) / (total_positives / total)
        lift_values.append(lift)
        
    ax.bar(range(1, 11), lift_values, color='skyblue', edgecolor='black')
    ax.set_title('리프트 차트(Lift)')
    ax.set_xlabel('데시일 (Decile)')
    ax.set_ylabel('리프트 (Lift)')
    ax.set_xticks(range(1, 11))
    ax.set_xticklabels(f'{i*10}%' for i in range(1, 11))
    ax.grid(True)
        
y_test_proba = rf_final.predict_proba(X_test)[:,1]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

plot_cumulative_gain(y_test, y_test_proba, axes[0])
plot_lift_chart(y_test, y_test_proba, axes[1])

plt.tight_layout()
plt.show()        
# %%
#ROC
from sklearn.metrics import roc_curve, roc_auc_score

y_prob = rf_final.predict_proba(X_test)[0:,1]

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
auc_score = roc_auc_score(y_test, y_prob)

plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score: .2f})', color='orange')
plt.plot([0,1], [0,1], 'k--', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate(Recall)')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()