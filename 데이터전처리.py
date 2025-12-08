import pandas as pd
import numpy as np

ko_data2 = pd.read_csv('ufc_totalround_2.csv')
ko_data3 = pd.read_csv('ufc_totalround_3.csv')

nko_data2 = pd.read_csv('ufc_notko_totalround_2.csv')
nko_data3 = pd.read_csv('ufc_notko_totalround_3.csv')


ko_data2['final_round'] = '2'
ko_data3['final_round'] = '3'

nko_data2['final_round'] = '2'
nko_data3['final_round'] = '3'


ko_data2['ko여부'] = '1'
ko_data3['ko여부'] = '1'

nko_data2['ko여부'] = '0'
nko_data3['ko여부'] = '0'

ko_data2 = ko_data2[ko_data2['round'] != 2]
ko_data3 = ko_data3[ko_data3['round'] != 3]

nko_data2 = nko_data2[nko_data2['current_round'] != 2]
nko_data3 = nko_data3[nko_data3['current_round'] != 3]


ko_data3 = ko_data3.drop(columns=['method', 'total_round'])
nko_data2 = nko_data2.drop(columns=['total_round', 'method'])
nko_data3 = nko_data3.drop(columns=['total_round', 'method'])


nko_data2.rename(columns={'current_round': 'round', 'fight_id': 'fight'}, inplace=True)
nko_data3.rename(columns={'current_round': 'round', 'fight_id': 'fight'}, inplace=True)

# ---------------------
import pandas as pd
import ast

def safe_literal_eval(x):
    if pd.isna(x):
        return None
    try:
        return ast.literal_eval(x)
    except (ValueError, TypeError, SyntaxError):
        return None

def preprocess_ufc_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None:
        return None
    
    df = df.copy()

    type1_cols = [
        'sig_str', 'total_str', 'td_attempt', 'head', 'body', 'leg',
        'distance', 'clinch', 'ground'
    ]
    type2_cols = [
        'kd', 'sig_str_pct', 'td_pct', 'sub_att', 'rev', 'ctrl_sec'
    ]

    all_target_cols = type1_cols + type2_cols
    for col in all_target_cols:
        if col in df.columns:
            df[col] = df[col].apply(safe_literal_eval)

    for col in type1_cols:
        if col in df.columns:
            def safe_extract_type1(data, winner_loser, succ_att):
                if isinstance(data, (list, tuple)) and len(data) == 2:
                    try:
                        target = data[winner_loser]
                        if isinstance(target, (list, tuple)) and len(target) == 2:
                            return target[succ_att]
                    except IndexError:
                        pass
                return None
            
            df[f'winner_{col}_succ'] = df[col].apply(lambda x: safe_extract_type1(x, 0, 0))
            df[f'winner_{col}_att'] = df[col].apply(lambda x: safe_extract_type1(x, 0, 1))
            df[f'loser_{col}_succ'] = df[col].apply(lambda x: safe_extract_type1(x, 1, 0))
            df[f'loser_{col}_att'] = df[col].apply(lambda x: safe_extract_type1(x, 1, 1))

    df = df.drop(columns=[col for col in type1_cols if col in df.columns], errors='ignore')

    for col in type2_cols:
        if col in df.columns:
            def safe_extract_type2(data, index):
                if isinstance(data, (list, tuple)) and len(data) == 2:
                    try:
                        return data[index]
                    except IndexError:
                        pass
                return None
            
            df[f'winner_{col}'] = df[col].apply(lambda x: safe_extract_type2(x, 0))
            df[f'loser_{col}'] = df[col].apply(lambda x: safe_extract_type2(x, 1))

    df = df.drop(columns=[col for col in type2_cols if col in df.columns], errors='ignore')
    
    id_cols = ['fight', 'round']
    new_data_cols = [col for col in df.columns if col not in id_cols] 
    
    for col in new_data_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    for col in new_data_cols:
        if df[col].dtype == float and all(df[col].dropna().apply(lambda x: x.is_integer())):
            df[col] = df[col].astype('Int64')

    return df 

# -------------------------------------------------------------


if 'ko_data2' in locals() or 'ko_data2' in globals():
    ko_data2 = preprocess_ufc_df(ko_data2)
    
else:
    print("경고: ko_data2 변수가 정의되지 않았습니다. 건너뜁니다.")

if 'ko_data3' in locals() or 'ko_data3' in globals():
    ko_data3 = preprocess_ufc_df(ko_data3)

else:
    print("경고: ko_data3 변수가 정의되지 않았습니다. 건너뜁니다.")

if 'nko_data2' in locals() or 'nko_data2' in globals():
    nko_data2 = preprocess_ufc_df(nko_data2)
    
else:
    print("경고: nko_data2 변수가 정의되지 않았습니다. 건너뜁니다.")


if 'nko_data3' in locals() or 'nko_data3' in globals():
    nko_data3 = preprocess_ufc_df(nko_data3)
    
else:
    print("경고: nko_data3 변수가 정의되지 않았습니다. 건너뜁니다.")

# --------
id_cols_to_keep = ['final_round', 'ko여부']
group_col = 'fight'
ignore_col = 'round'

all_non_stat_cols = [group_col, ignore_col] + id_cols_to_keep
stat_cols = [col for col in ko_data3.columns if col not in all_non_stat_cols]

aggregation_funcs = {}

for col in stat_cols:
    aggregation_funcs[col] = 'mean'

for col in id_cols_to_keep:
    if col in ko_data3.columns:
        aggregation_funcs[col] = 'max'

if 'ko_data3' in locals() or 'ko_data3' in globals():
    ko_data3 = ko_data3.groupby(group_col).agg(aggregation_funcs).reset_index()
    

if 'nko_data3' in locals() or 'nko_data3' in globals():
    nko_data3 = nko_data3.groupby(group_col).agg(aggregation_funcs).reset_index()



ko_data2['fight'] = ko_data2['fight'].astype('str')
ko_data2['final_round'] = ko_data2['final_round'].astype('str')
ko_data2['ko여부'] = ko_data2['ko여부'].astype('str')

ko_data3['fight'] = ko_data3['fight'].astype('str')
ko_data3['final_round'] = ko_data3['final_round'].astype('str')
ko_data3['ko여부'] = ko_data3['ko여부'].astype('str')

nko_data2['fight'] = nko_data2['fight'].astype('str')
nko_data2['final_round'] = nko_data2['final_round'].astype('str')
nko_data2['ko여부'] = nko_data2['ko여부'].astype('str')

nko_data3['fight'] = nko_data3['fight'].astype('str')
nko_data3['final_round'] = nko_data3['final_round'].astype('str')
nko_data3['ko여부'] = nko_data3['ko여부'].astype('str')


ko_data = pd.concat([ko_data2, ko_data3], ignore_index=True)
nko_data = pd.concat([nko_data2, nko_data3], ignore_index=True)

# 최종 전처리된 데이터를 CSV로 저장
ko_data.to_csv('ko_final.csv', index=False)
nko_data.to_csv('nko_final.csv', index=False)