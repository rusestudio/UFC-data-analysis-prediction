# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('ufc_df.csv')
print(df.head(10))

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
plt.show()

df['ko여부'].value_counts()
# %%
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

ko_100 = df[df['ko여부'] == 1].sample(100, random_state=42)

nko_100 = df[df['ko여부'] == 0].sample(100, random_state=42)

df_200 = pd.concat([ko_100, nko_100]).sample(frac=1, random_state=42)

print(df_200['ko여부'].value_counts())
print(df_200.shape)
# %%
from sklearn.model_selection import train_test_split

X = df_200.drop(columns=['ko여부', 'fight']) #경기 label 필요없을 것 같아 제거
y = df_200['ko여부']

X_train_valid, X_test, y_train_valid, y_train = train_test_split(
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
importances = rf.feature_importances_
feature_names = X_train.columns

imp_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
})

imp_df = imp_df.sort_values('importance', ascending=False)

print(imp_df.head(10))

#중요도 시각화
imp_df.plot(kind='barh')
plt.title('변수 중요도(Random Forest)')
plt.show()
# %%
selected_df = imp_df[imp_df['importance'] >= 0.025]

selected_df.plot(kind='barh')
plt.title('변수 중요도(0.025 이상)')
plt.show()
# %%
