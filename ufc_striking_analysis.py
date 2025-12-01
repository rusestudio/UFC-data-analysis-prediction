import pandas as pd
import numpy as np
from scipy import stats


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    
    return df


'''
KO 라운드 직전까지의 합계를 '라운드당 평균'으로 나누는 함수.

ko_round = 2 -> 이전 라운드 수 = 1 (1R만 사용)
ko_round = 3 -> 이전 라운드 수 = 2 (1R+2R 합 / 2)
'''
def normalize_per_round(df: pd.DataFrame, ko_round: int) -> pd.DataFrame:
    prev_rounds = ko_round - 1

    # KO 경기 컬럼
    for col in [
        "ko_winner_body_total",
        "ko_winner_leg_total",
        "ko_loser_body_total",
        "ko_loser_leg_total",
    ]:
        if col in df.columns:
            df[f"{col}_per_round"] = df[col] / prev_rounds

    # Non-KO 경기 컬럼
    for col in [
        "non_ko_winner_body_total",
        "non_ko_winner_leg_total",
        "non_ko_loser_body_total",
        "non_ko_loser_leg_total",
    ]:
        if col in df.columns:
            df[f"{col}_per_round"] = df[col] / prev_rounds

    return df


'''
2라운드 KO / 3라운드 KO 전처리 결과를 합쳐서
- ko_all  : KO 경기들만
- non_all : Non-KO 경기들만
반환
'''
def combined_ko_nonko(r2_path: str, r3_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    r2 = load_csv(r2_path)
    r3 = load_csv(r3_path)

    # KO 직전 라운드 수로 나눠서 per-round 평균 만들기
    r2 = normalize_per_round(r2, 2)
    r3 = normalize_per_round(r3, 3)

    # KO 경기만 추출 후 결합
    r2_ko = r2.dropna(subset=["ko_winner_body_total"])
    r3_ko = r3.dropna(subset=["ko_winner_body_total"])
    ko_all = pd.concat([r2_ko, r3_ko], ignore_index=True)

    # Non-KO 경기만 추출 후 결합
    r2_non = r2.dropna(subset=["non_ko_winner_body_total"])
    r3_non = r3.dropna(subset=["non_ko_winner_body_total"])
    non_all = pd.concat([r2_non, r3_non], ignore_index=True)

    return ko_all, non_all


'''
두 그룹 평균 차이에 대한 효과 크기(Hedge's g)를 계산
'''
def hedges_g(a: pd.Series, b: pd.Series) -> float:
    a = a.dropna().to_numpy()
    b = b.dropna().to_numpy()
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan

    va, vb = a.var(ddof=1), b.var(ddof=1)
    sp = np.sqrt((va + vb) / 2.0)
    if sp == 0:
        return 0.0

    g = (a.mean() - b.mean()) / sp
    # 작은 표본 보정
    J = 1.0 - (3.0 / (4.0 * (na + nb) - 9.0))
    return g * J


def analyze() -> None:
    # 2R KO / 3R KO 전처리 결과를 합친 데이터 불러오기
    ko_all, non_all = combined_ko_nonko(
        "./ufc_extracted_ko_round_2_data.csv",
        "./ufc_extracted_ko_round_3_data.csv",
    )

    print(f"KO 경기 수: {len(ko_all)}, Non-KO 경기 수: {len(non_all)}")

    '''
    가설 1: KO 경기 승자의 KO 이전 라운드당 바디 타격 수는
           Non-KO 경기 승자보다 많다.
    '''
    ko_body = ko_all["ko_winner_body_total_per_round"].dropna()
    non_body = non_all["non_ko_winner_body_total_per_round"].dropna()

    t1, p1 = stats.ttest_ind(ko_body, non_body, equal_var=False)
    g1 = hedges_g(ko_body, non_body)


    '''
    가설 2: KO 경기 승자의 KO 이전 라운드당 레그 타격 수는
           Non-KO 경기 승자보다 많다.
    '''
    ko_leg = ko_all["ko_winner_leg_total_per_round"].dropna()
    non_leg = non_all["non_ko_winner_leg_total_per_round"].dropna()

    t2, p2 = stats.ttest_ind(ko_leg, non_leg, equal_var=False)
    g2 = hedges_g(ko_leg, non_leg)


    '''
    가설 3: KO 경기 승자의 KO 이전 바디 비율은
           Non-KO 경기 승자 비율보다 높다.
    '''
    ko_body = ko_all["ko_winner_body_total_per_round"].dropna()
    # body_ratio = body_per_round / (body_per_round + leg_per_round)

    # KO 쪽 ratio
    ko_body_all = ko_all["ko_winner_body_total_per_round"]
    ko_leg_all = ko_all["ko_winner_leg_total_per_round"]
    ko_mask = ko_body_all.notna() & ko_leg_all.notna()
    ko_ratio = ko_body_all[ko_mask] / (
        ko_body_all[ko_mask] + ko_leg_all[ko_mask] + 1e-6
    )

    # Non-KO 쪽 ratio
    non_body_all = non_all["non_ko_winner_body_total_per_round"]
    non_leg_all = non_all["non_ko_winner_leg_total_per_round"]
    non_mask = non_body_all.notna() & non_leg_all.notna()
    non_ratio = non_body_all[non_mask] / (
        non_body_all[non_mask] + non_leg_all[non_mask] + 1e-6
    )

    t3, p3 = stats.ttest_ind(ko_ratio, non_ratio, equal_var=False)
    g3 = hedges_g(ko_ratio, non_ratio)


    '''
    가설 4: KO 경기에서 승자는 패자보다
           라운드당 바디 타격 수가 많다.
           (같은 경기 안의 두 사람 비교 -> paired t-test)
    '''
    win_body_all = ko_all["ko_winner_body_total_per_round"]
    lose_body_all = ko_all["ko_loser_body_total_per_round"]
    pair_mask = win_body_all.notna() & lose_body_all.notna()

    win_body = win_body_all[pair_mask]
    lose_body = lose_body_all[pair_mask]

    # 같은 경기 내에서 winner vs loser 비교 ⇒ 대응표본 t-test
    t4, p4 = stats.ttest_rel(win_body, lose_body)

    # 대응표본이긴 하지만, 대략적인 크기 감 보기 위해 g 계산 (참고용)
    g4 = hedges_g(win_body, lose_body)

    # 결과 출력
    print("\n[가설 1] KO vs Non-KO (승자의 라운드당 바디 타격 수)")
    print(f"  KO     mean = {ko_body.mean():.3f}")
    print(f"  Non-KO mean = {non_body.mean():.3f}")
    print(f"  Welch t = {t1:.3f}, p = {p1:.4f}, Hedge's g = {g1:.3f}")

    print("\n[가설 2] KO vs Non-KO (승자의 라운드당 레그 타격 수)")
    print(f"  KO     mean = {ko_leg.mean():.3f}")
    print(f"  Non-KO mean = {non_leg.mean():.3f}")
    print(f"  Welch t = {t2:.3f}, p = {p2:.4f}, Hedge's g = {g2:.3f}")

    print("\n[가설 3] KO vs Non-KO (승자 바디 비율)")
    print(f"  KO     mean = {ko_ratio.mean():.3f}")
    print(f"  Non-KO mean = {non_ratio.mean():.3f}")
    print(f"  Welch t = {t3:.3f}, p = {p3:.4f}, Hedge's g = {g3:.3f}")

    print("\n[가설 4] KO 경기에서 승자 vs 패자 (라운드당 바디 타격 수)")
    print(f"  Winner mean = {win_body.mean():.3f}")
    print(f"  Loser  mean = {lose_body.mean():.3f}")
    print(f"  Paired t = {t4:.3f}, p = {p4:.4f}, (참고용 Hedge's g = {g4:.3f})")

    print("\n해석")
    print("- p < 0.05 이면 통계적으로 의미 있는 차이가 있다고 볼 수 있음")
    print("- Hedge's g: 0.2(작은 차이), 0.5(중간), 0.8 이상(큰 차이) 정도로 해석")


if __name__ == "__main__":
    analyze()
