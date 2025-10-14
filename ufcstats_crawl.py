import requests
from bs4 import BeautifulSoup

#to get 10 page
BASE_URL = "http://ufcstats.com"
MATCH_URL = f"{BASE_URL}/statistics/events/completed?page="

match_links = [] #all match link in 10 page

for page in range(1, 11):
    url = f"{MATCH_URL}{page}" #get all url till page 10 and jan
    print(f"Scraping event list page {page}: {url}") #print page link
    
    response = requests.get(url) #req again to every page
    soup = BeautifulSoup(response.text, "html.parser") #get html tag in all page
    
    rows = soup.find_all("tr", class_="b-statistics__table-row") # in html find links of match in each page
    
    for row in rows: # in every row
        link_tag = row.find("a") #find match links tag
        date_tag = row.find("span", class_="b-statistics__date") #find date
        
        if link_tag and date_tag:
            date_text = date_tag.text.strip()
            if date_text == "January 18, 2020":
                match_links.append(link_tag["href"])
                print("Reached Jan 18, 2020 — stopping.") #stop at 2020
                break
            match_links.append(link_tag["href"]) #save match link #all match link in 10 page. there are 245 match
    else:
        continue
    break  # stop outer loop too once we reach Jan 18, 2020

print(f"Collected {len(match_links)} match links.")
print(match_links[:5]) #print first 5 match links

#req again with new link from matchlinks
for link in match_links:  # fixed: iterate actual links
    newresponse = requests.get(link)  # fixed: request the link, not list
    soup = BeautifulSoup(newresponse.text, "html.parser")

    #in each match link table
    #find all right row
    fight_rows = soup.find_all('tr', class_="b-fight-details__table-row b-fight-details__table-row__hover js-fight-details-click")

    for row in fight_rows:
        tds = row.find_all("td") #find all td in each fight row
        if len(tds) >= 9: #total row is 9 - 0~9
            #get data in each row
            #get weight in 6 row. find the p tag with info
            weight_class = tds[6].find("p").get_text(strip=True)
            #get method in 7 row. have 2 p tag get the first one
            method = tds[7].find_all("p")[0].get_text(strip=True)
            #get total round in 8 row
            total_round = tds[8].find("p").get_text(strip=True)

            #get only matching category
            if weight_class == "Lightweight" and (method in ["KO/TKO", "KO", "TKO"]):
                print("match found")
                print("Weight:", weight_class)
                print("Method:", method)
                print("Round:", total_round)


