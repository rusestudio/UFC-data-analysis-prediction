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
    """Collect all event match links until a specific date"""
    match_links = []
    for page in range(1, pages + 1):
        url = f"{MATCH_URL}{page}"
        print(f"📄 Scraping event list page {page}: {url}")
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")

        rows = soup.find_all("tr", class_="b-statistics__table-row")
        for row in rows:
            link_tag = row.find("a")
            date_tag = row.find("span", class_="b-statistics__date")

            if link_tag and date_tag:
                date_text = date_tag.text.strip()
                if date_text == stop_date:
                    print(f"🛑 Reached {stop_date} — stopping.")
                    return match_links
                match_links.append(link_tag["href"])
    return match_links