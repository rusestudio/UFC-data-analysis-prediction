import csv
import re
import time
from collections import defaultdict

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

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