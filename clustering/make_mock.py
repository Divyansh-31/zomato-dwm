"""
clustering/make_mock.py — Mock Data Generator for Clustering Development
========================================================================
Creates a mock version of data/processed/zomato_clean.csv with 2,000 realistic
rows matching the exact data contract. Used by Person B to prototype clustering
code before the real cleaned dataset is available.
"""

import os
import numpy as np
import pandas as pd

CONTRACT_COLUMNS = [
    "restaurant_id", "name", "city", "location", "rest_type",
    "category", "cuisines", "primary_cuisine", "cuisine_count",
    "online_order", "book_table", "rate", "votes", "log_votes", "cost",
]

CITIES = ["BTM", "Brigade Road", "Indiranagar", "Koramangala 5th Block", "Whitefield", "Jayanagar"]
REST_TYPES = ["Quick Bites", "Casual Dining", "Cafe", "Delivery", "Dessert Parlor"]
CATEGORIES = ["Delivery", "Dine-out", "Cafes", "Desserts", "Buffet"]
CUISINES_LIST = [
    ("North Indian", 1),
    ("South Indian", 1),
    ("North Indian, Chinese", 2),
    ("Cafe, Continental", 2),
    ("Fast Food, Beverages", 2),
    ("Biryani, North Indian, Chinese", 3),
    ("Italian, Pizza, Desserts", 3),
]


def generate_mock(n=2000, output_path="data/processed/zomato_clean.csv"):
    np.random.seed(42)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    data = []
    for i in range(1, n + 1):
        city = np.random.choice(CITIES)
        location = city
        rest_type = np.random.choice(REST_TYPES)
        category = np.random.choice(CATEGORIES)
        cuisine_tuple = CUISINES_LIST[np.random.choice(len(CUISINES_LIST))]
        cuisines = cuisine_tuple[0]
        primary_cuisine = cuisines.split(",")[0].strip()
        cuisine_count = cuisine_tuple[1]

        online_order = int(np.random.rand() > 0.4)
        book_table = int(np.random.rand() > 0.8)
        rate = round(float(np.clip(np.random.normal(3.7, 0.4), 2.5, 4.9)), 1)
        votes = int(np.random.exponential(scale=200))
        log_votes = round(float(np.log1p(votes)), 4)
        cost = float(np.random.choice(range(100, 2501, 50)))

        data.append({
            "restaurant_id": i,
            "name": f"Mock Restaurant {i}",
            "city": city,
            "location": location,
            "rest_type": rest_type,
            "category": category,
            "cuisines": cuisines,
            "primary_cuisine": primary_cuisine,
            "cuisine_count": cuisine_count,
            "online_order": online_order,
            "book_table": book_table,
            "rate": rate,
            "votes": votes,
            "log_votes": log_votes,
            "cost": cost,
        })

    df = pd.DataFrame(data)[CONTRACT_COLUMNS]
    df.to_csv(output_path, index=False)
    print(f"[OK] Generated {n} mock rows -> {output_path}")


if __name__ == "__main__":
    generate_mock()
