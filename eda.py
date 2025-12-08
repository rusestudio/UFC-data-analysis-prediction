import pandas as pd
import numpy as np
import numpy as np
from scipy.stats import ttest_ind
import seaborn as sns
import matplotlib.pyplot as plt

ko_data = pd.read_csv('ko_final.csv')
nko_data = pd.read_csv('nko_final.csv')


ko_mean = ko_data['winner_sig_str_succ'].mean()
nko_mean = nko_data['winner_sig_str_succ'].mean()

print(ko_mean, nko_mean)

ko = ko_data['winner_sig_str_succ'].dropna()
nko = nko_data['winner_sig_str_succ'].dropna()

t_stat, p_val = ttest_ind(ko, nko, equal_var=False)
print("t-stat:", t_stat)
print("p-value:", p_val)

cohen_d = (ko.mean() - nko.mean()) / np.sqrt((ko.var() + nko.var()) / 2)
print('cohen_d:', cohen_d)

sns.kdeplot(ko, label="KO", shade=True)
sns.kdeplot(nko, label="NKO", shade=True)
plt.legend()
plt.title("SigStr% Distribution (Winner)")
plt.show()


ko_mean = ko_data['winner_head_succ'].mean()
nko_mean = nko_data['winner_head_succ'].mean()
print(ko_mean, nko_mean)


ko = ko_data['winner_head_succ'].dropna()
nko = nko_data['winner_head_succ'].dropna()

t_stat, p_val = ttest_ind(ko, nko, equal_var=False)
print("t-stat:", t_stat)
print("p-value:", p_val)


cohen_d = (ko.mean() - nko.mean()) / np.sqrt((ko.var() + nko.var()) / 2)
print("cohen_d:", cohen_d)

sns.kdeplot(ko, shade=True, label="KO")
sns.kdeplot(nko, shade=True, label="NKO")
plt.title("Head Strikes succ (Winner)")
plt.legend()
plt.show()

ko = ko_data['winner_distance_succ'].dropna()
nko = nko_data['winner_distance_succ'].dropna()

ko_mean = ko.mean()
nko_mean = nko.mean()


t_stat, p_val = ttest_ind(ko, nko, equal_var=False)


cohen_d = (ko_mean - nko_mean) / np.sqrt((ko.var() + nko.var()) / 2)

print("KO mean distance:", ko_mean)
print("NKO mean distance:", nko_mean)
print("\nt-stat:", t_stat)
print("p-value:", p_val)
print("Cohen's d:", cohen_d)

plt.figure(figsize=(8,5))
sns.kdeplot(ko, shade=True, label='KO')
sns.kdeplot(nko, shade=True, label='NKO')
plt.title("Winner Distance Strikes Landed (KO vs NKO)")
plt.legend()
plt.show()
