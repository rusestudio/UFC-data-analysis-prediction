import matplotlib.pyplot as plt
import pandas as pd

def open_csv(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    return df

def visualize_ko_data(ko_round_2_data: pd.DataFrame, ko_round_3_data: pd.DataFrame):
    # NaN 버리고 평균
    def safe_mean(df, col):
        return df[col].dropna().mean()

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    # KO Round 2
    axs[0, 0].plot(['Body', 'Leg'],
         [safe_mean(ko_round_2_data,'ko_winner_body_total'), 
          safe_mean(ko_round_2_data,'ko_winner_leg_total')],
         label='Winner', marker='o')
    axs[0, 0].plot(['Body', 'Leg'],
         [safe_mean(ko_round_2_data,'ko_loser_body_total'), 
          safe_mean(ko_round_2_data,'ko_loser_leg_total')],
         label='Loser', marker='o')
    axs[0, 0].set_title('KO Round 2 (Prev R1 Attempts)')
    axs[0, 0].set_ylabel('Avg Attempts')
    axs[0, 0].legend()

    # Non-KO Round 2
    axs[0, 1].plot(['Body', 'Leg'],
         [safe_mean(ko_round_2_data,'non_ko_winner_body_total'), 
          safe_mean(ko_round_2_data,'non_ko_winner_leg_total')],
         label='Winner', marker='o')
    axs[0, 1].plot(['Body', 'Leg'],
         [safe_mean(ko_round_2_data,'non_ko_loser_body_total'), 
          safe_mean(ko_round_2_data,'non_ko_loser_leg_total')],
         label='Loser', marker='o')
    axs[0, 1].set_title('Non-KO Round 2 (Prev R1 Attempts)')
    axs[0, 1].legend()

    # KO Round 3
    axs[1, 0].plot(['Body', 'Leg'],
         [safe_mean(ko_round_3_data,'ko_winner_body_total'), 
          safe_mean(ko_round_3_data,'ko_winner_leg_total')],
         label='Winner', marker='o')
    axs[1, 0].plot(['Body', 'Leg'],
         [safe_mean(ko_round_3_data,'ko_loser_body_total'), 
          safe_mean(ko_round_3_data,'ko_loser_leg_total')],
         label='Loser', marker='o')
    axs[1, 0].set_title('KO Round 3 (Prev R1+R2 Attempts)')
    axs[1, 0].set_xlabel('Target Area')
    axs[1, 0].set_ylabel('Avg Attempts')
    axs[1, 0].legend()

    # Non-KO Round 3
    axs[1, 1].plot(['Body', 'Leg'],
         [safe_mean(ko_round_3_data,'non_ko_winner_body_total'), 
          safe_mean(ko_round_3_data,'non_ko_winner_leg_total')],
         label='Winner', marker='o')
    axs[1, 1].plot(['Body', 'Leg'],
         [safe_mean(ko_round_3_data,'non_ko_loser_body_total'), 
          safe_mean(ko_round_3_data,'non_ko_loser_leg_total')],
         label='Loser', marker='o')
    axs[1, 1].set_title('Non-KO Round 3 (Prev R1+R2 Attempts)')
    axs[1, 1].set_xlabel('Target Area')
    axs[1, 1].legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    ko_round_2_file_path = "./ufc_extracted_ko_round_2_data.csv"
    ko_round_3_file_path = "./ufc_extracted_ko_round_3_data.csv"

    ko_round_2_data = open_csv(ko_round_2_file_path)
    ko_round_3_data = open_csv(ko_round_3_file_path)

    visualize_ko_data(ko_round_2_data, ko_round_3_data)