"""
Scraping second-hand car listings from AutoScout24 (Germany)

This script:
- Uses Selenium to handle dynamic content
- Scrapes overview listing pages (not detail pages)
- Collects real-world market data for used cars
- Stores the result as a CSV file for ML training

"""

# IMPORT REQUIRED LIBRARIES

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options

import pandas as pd
import time
import random
import re

# CONFIGURATION SECTION
# Base URL for AutoScout24 car listings
BASE_URL = "https://www.autoscout24.de/lst"

# Number of pages to scrape
# Each page contains ~20 listings
PAGES = 500

# Output CSV file
OUTPUT_FILE = "raw_car_data_2.csv"

# SELENIUM DRIVER SETUP
# Chrome options to reduce bot detection
options = Options()
options.add_argument("--disable-blink-features=AutomationControlled")
options.add_argument("--start-maximized")

# Automatically download & use correct ChromeDriver
driver = webdriver.Chrome(
    service=Service(ChromeDriverManager().install()),
    options=options
)

# Data container 
# Each element in this list will be one car listing
cars = []

# scraping loop

for page in range(1, PAGES + 1):
    # Construct URL for the current page
    url = f"{BASE_URL}?page={page}"
    print(f"Scraping page {page}")

    # Load page
    driver.get(url)

    # Random sleep to mimic human browsing behavior
    time.sleep(random.uniform(4, 6))

    # Each car listing is contained in an <article> tag
    ads = driver.find_elements(By.XPATH, "//article")

    for ad in ads:

        # TITLE (Brand + Model)
        try:
            title = ad.find_element(By.XPATH, ".//h2").text.replace("\n", " ")
        except:
            # Skip ad if title is missing
            continue

        # Extract brand and model from title
        brand = title.split()[0]
        model = " ".join(title.split()[1:3])

        # PRICE
        price = None
        try:
            price_text = ad.find_element(
                By.XPATH, ".//span[contains(@class,'Price')]"
            ).text

            # Remove currency symbols and text
            price = re.sub(r"[^\d]", "", price_text)
        except:
            # Some listings do not show a price
            pass

        # DETAILS (year, mileage, fuel, power)
        year = mileage = fuel = power = None

        try:
            # Collect all visible span text inside the ad
            details = ad.find_elements(By.XPATH, ".//span")
            details_text = " ".join([d.text for d in details])
        except:
            details_text = ""

        # Year
        year_match = re.search(r"\b(19|20)\d{2}\b", details_text)
        if year_match:
            year = year_match.group()

        # Mileage
        km_match = re.search(r"([\d\.]+)\s*km", details_text)
        if km_match:
            mileage = km_match.group(1).replace(".", "")

        # Power of engine (PS) 
        ps_match = re.search(r"([\d]+)\s*PS", details_text)
        if ps_match:
            power = ps_match.group(1)

        # Fuel type
        for f in ["Benzin", "Diesel", "Hybrid", "Elektro", "Gas"]:
            if f in details_text:
                fuel = f
                break

        # Store result 
        cars.append({
            "brand": brand,
            "model": model,
            "year": year,
            "mileage": mileage,
            "fuel": fuel,
            "power_ps": power,
            "price": price
        })

    print(f"Total collected so far: {len(cars)}")



# CLEANUP & SAVE DATA
# Close browser
driver.quit()

# Convert to DataFrame and save as CSV
df = pd.DataFrame(cars)
df.to_csv(OUTPUT_FILE, index=False)

print("Scraping finished!")
print("Saved:", OUTPUT_FILE)
