import csv
import re
import time
from collections import defaultdict

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm


def split_of_list(lst):
    clean = []
    for val in lst:
        match = re.findall(r"(\d+)\s*of\s*(\d+)", val)  # remove of
        if match:
            clean.append([int(match[0][0]), int(match[0][1])])
        else:
            clean.append([0, 0])
    return clean


def clean_pct_list(lst):
    return [
        float(re.sub(r"[^0-9.]", "", v)) if re.search(r"\d", v) else 0.0 for v in lst
    ]  # % to float


def clean_int_list(lst):
    return [int(re.sub(r"\D", "", v)) if re.search(r"\d", v) else 0 for v in lst]


# to int


def ctrl_to_seconds(lst):
    result = []
    for v in lst:
        match = re.findall(r"(\d+):(\d+)", v)
        if match:
            m, s = map(int, match[0])
            result.append(m * 60 + s)  # min to sec
        else:
            result.append(0)
    return result


# req url
BASE_URL = "http://ufcstats.com"
MATCH_URL = f"{BASE_URL}/statistics/events/completed?page="


def get_match_links(pages=18, stop_date="January 03, 2015"):
    match_links = []
    for page in range(1, pages + 1):
        url = f"{MATCH_URL}{page}"
        print(f"scrape event {page}: {url}")
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")

        rows = soup.find_all("tr", class_="b-statistics__table-row")
        for row in rows:
            link_tag = row.find("a")
            date_tag = row.find("span", class_="b-statistics__date")

            if link_tag and date_tag:
                date_text = date_tag.text.strip()
                if date_text == stop_date:
                    print(f" {stop_date} -stop.")
                    return match_links
                match_links.append(link_tag["href"])
    return match_links


def filter_fight_links(match_links):
    stats_link = []

    for match in tqdm(match_links, desc="Collecting fight links"):
        response = requests.get(match)
        soup = BeautifulSoup(response.text, "html.parser")
        fight_rows = soup.find_all(
            "tr",
            class_="b-fight-details__table-row b-fight-details__table-row__hover js-fight-details-click",
        )

        for row in fight_rows:
            tds = row.find_all("td")
            if len(tds) >= 9:
                weight_class = tds[6].find("p").get_text(strip=True)
                method = tds[7].find_all("p")[0].get_text(strip=True)
                if weight_class == "Lightweight" and method in ["KO/TKO", "KO", "TKO"]:
                    data_link = row.get("data-link")
                    stats_link.append(data_link)

    print(f"{len(stats_link)} filtered fight")
    return stats_link


def extract_fight_data(stats_links):
    fight_data = defaultdict(list)

    for all_stats in tqdm(stats_links, desc="Extracting stats"):
        response = requests.get(all_stats)
        soup = BeautifulSoup(response.text, "html.parser")

        fight_details = soup.find("div", class_="b-fight-details__fight")
        if not fight_details:
            continue

        # div data
        total_round = fight_details.find("i", class_="b-fight-details__text-item")
        total_round_c = (
            total_round.get_text(strip=True).replace("Round:", "").strip()
            if total_round
            else "-"
        )
        method_tag = fight_details.find("i", attrs={"style": "font-style: normal"})
        method = method_tag.get_text(strip=True) if method_tag else "-"

        # tables stats
        tables = soup.find_all("table", class_="b-fight-details__table js-fight-table")
        if len(tables) < 2:
            continue

        t1_rows = [  # total tables
            r
            for r in tables[0].find_all("tr", class_="b-fight-details__table-row")
            if len(r.find_all("td")) >= 10
        ]
        t2_rows = [  # signf strike
            r
            for r in tables[1].find_all("tr", class_="b-fight-details__table-row")
            if len(r.find_all("td")) >= 9
        ]

        for idx, section in enumerate(t1_rows, start=1):
            td_stats = section.find_all("td")

            def extract_col(col_idx):
                return [p.get_text(strip=True) for p in td_stats[col_idx].find_all("p")]

            kd = extract_col(1)
            sig_str = extract_col(2)
            sig_str_pct = extract_col(3)
            total_str = extract_col(4)
            td_attempt = extract_col(5)
            td_pct = extract_col(6)
            sub_att = extract_col(7)
            rev = extract_col(8)
            ctrl_sec = extract_col(9)

            # match index for second table
            if idx - 1 < len(t2_rows):
                td_stats_2 = t2_rows[idx - 1].find_all("td")

                def get_p(idx):
                    return [
                        p.get_text(strip=True) for p in td_stats_2[idx].find_all("p")
                    ]

                head, body, leg = get_p(3), get_p(4), get_p(5)
                distance, clinch, ground = get_p(6), get_p(7), get_p(8)
            else:
                head = body = leg = distance = clinch = ground = ["-", "-"]

            # re apply
            round_data = {
                "link": all_stats,
                "round": idx,
                "kd": clean_int_list(kd),
                "sig_str": split_of_list(sig_str),
                "sig_str_pct": clean_pct_list(sig_str_pct),
                "total_str": split_of_list(total_str),
                "td_attempt": split_of_list(td_attempt),
                "td_pct": clean_pct_list(td_pct),
                "sub_att": clean_int_list(sub_att),
                "rev": clean_int_list(rev),
                "ctrl_sec": ctrl_to_seconds(ctrl_sec),
                "head": split_of_list(head),
                "body": split_of_list(body),
                "leg": split_of_list(leg),
                "distance": split_of_list(distance),
                "clinch": split_of_list(clinch),
                "ground": split_of_list(ground),
                "total_round": total_round_c,
                "method": method,
            }

            fight_data[all_stats].append(round_data)
        time.sleep(0.5)
    return fight_data


def save_to_csv(fight_data):
    fights_by_round_count = defaultdict(list)
    columns = [
        "fight",
        "round",
        "kd",
        "sig_str",
        "sig_str_pct",
        "total_str",
        "td_attempt",
        "td_pct",
        "sub_att",
        "rev",
        "ctrl_sec",
        "head",
        "body",
        "leg",
        "distance",
        "clinch",
        "ground",
        "total_round",
        "method",
    ]

    fight_counter = 1
    for link, rounds in fight_data.items():
        total_round_c = rounds[0]["total_round"]
        fight_name = f"fight{fight_counter}"
        for r in rounds:
            row = {"fight": fight_name, "round": r["round"]}
            for k, v in r.items():
                if k not in ["link", "round", "total_round"]:
                    row[k] = v
            fights_by_round_count[total_round_c].append(row)
        fight_counter += 1

    for total_round_c, rows in fights_by_round_count.items():
        if total_round_c in ["4", "5"]:  # only 123 round
            continue
        filename = f"ufc_totalround_{total_round_c}.csv"
        with open(filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        print(f" {filename} ({len(rows)} rows)")


def main():
    print("start crawl")
    match_links = get_match_links()
    stats_links = filter_fight_links(match_links)
    fight_data = extract_fight_data(stats_links)
    save_to_csv(fight_data)
    print("finish crawl")


if __name__ == "__main__":
    main()
