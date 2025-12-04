import pandas as pd
import ast

def load_ufc_data(file_path: str) -> pd.DataFrame:
    data = pd.read_csv(file_path)
    return data

'''
UFC 경기 데이터에서 KO 승리 선수의 이전 라운드 타격 데이터 추출 + 패배 선수의 이전 라운드 타격 데이터 추출
'''
def extracted_ko_data(ko_data: pd.DataFrame, non_ko_data: pd.DataFrame, ko_round: int) -> pd.DataFrame:
    '''
    ko_winner_body_avg
    ko_winner_leg_avg
    ko_loser_body_avg
    ko_loser_leg_avg
    '''
    # 필요 컬럼만 추출 + string 형태의 리스트를 실제 리스트로 변환
    extracted_ko_prev_data = ko_data[ko_data['round'] < ko_round][['fight', 'body', 'leg']].copy()
    extracted_non_ko_prev_data = non_ko_data[non_ko_data['round'] < ko_round][['fight', 'body', 'leg']].copy()

    for column in ['body', 'leg']:
        extracted_ko_prev_data[column] = extracted_ko_prev_data[column].apply(ast.literal_eval)
        extracted_non_ko_prev_data[column] = extracted_non_ko_prev_data[column].apply(ast.literal_eval)

    # attempts 합산 함수
    def sum_attempts(rounds_list, idx_pair):
        return sum(r[idx_pair][1] for r in rounds_list)

    # KO: fight별 누적 (여러 라운드 → 하나로 합침)
    ko_agg = extracted_ko_prev_data.groupby('fight').agg({'body': list, 'leg': list}).reset_index()
    ko_agg['ko_winner_body_total'] = ko_agg['body'].apply(lambda x: sum_attempts(x, 0))
    ko_agg['ko_winner_leg_total'] = ko_agg['leg'].apply(lambda x: sum_attempts(x, 0))
    ko_agg['ko_loser_body_total'] = ko_agg['body'].apply(lambda x: sum_attempts(x, 1))
    ko_agg['ko_loser_leg_total'] = ko_agg['leg'].apply(lambda x: sum_attempts(x, 1))

    # Non-KO: fight별 누적
    non_agg = extracted_non_ko_prev_data.groupby('fight').agg({'body': list, 'leg': list}).reset_index()
    non_agg['non_ko_winner_body_total'] = non_agg['body'].apply(lambda x: sum_attempts(x, 0))
    non_agg['non_ko_winner_leg_total'] = non_agg['leg'].apply(lambda x: sum_attempts(x, 0))
    non_agg['non_ko_loser_body_total'] = non_agg['body'].apply(lambda x: sum_attempts(x, 1))
    non_agg['non_ko_loser_leg_total'] = non_agg['leg'].apply(lambda x: sum_attempts(x, 1))
    non_agg = non_agg.rename(columns={'fight': 'fight'})

    # 병합 (KO와 Non-KO는 다른 경기들이므로 outer)
    ko_final = ko_agg[['fight','ko_winner_body_total','ko_winner_leg_total','ko_loser_body_total','ko_loser_leg_total']]
    non_final = non_agg[['fight','non_ko_winner_body_total','non_ko_winner_leg_total','non_ko_loser_body_total','non_ko_loser_leg_total']]
    
    merged = pd.merge(ko_final, non_final, on='fight', how='outer')
    return merged


'''
UFC 경기 데이터에서 KO 승리 선수와 Non_KO 승리 선수의 이전 라운드 타격 데이터 추출 및 저장
'''
def process_and_save_data():
    ko_file_path = ["./ufc_totalround_2.csv",
                    "./ufc_totalround_3.csv"]
    non_ko_file_path = ["./ufc_notko_totalround_2.csv",
                        "./ufc_notko_totalround_3.csv"]

    ko_file_name = "ufc_extracted_ko_round_{}_data.csv"

    for i in range(len(ko_file_path)):
        non_ko_data = load_ufc_data(non_ko_file_path[i])
        ko_data = load_ufc_data(ko_file_path[i])
        ko_round = int(ko_file_path[i].split("_")[-1].split(".")[0])
        extracted_data = extracted_ko_data(ko_data, non_ko_data, ko_round)
        extracted_data.to_csv(ko_file_name.format(ko_round), index=False)


if __name__ == "__main__":
    process_and_save_data()