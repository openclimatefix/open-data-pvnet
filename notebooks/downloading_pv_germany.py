# SMARD API Data Downloader for Germany PV Generation
# Downloads solar PV generation data from Bundesnetzagentur SMARD APIimport requests
import pandas as pd
import time
from datetime import datetime
from pathlib import Path
import requests

BASE_URL = "https://www.smard.de/app/chart_data"
FILTER_ID = 4068
REGION = "DE"
OUTPUT_DIR = Path("./germany_pv_data")
OUTPUT_DIR.mkdir(exist_ok=True)

START_DATE = "2021-01-01"
END_DATE   = "2021-12-31"

def get_timestamps(res="quarterhour"):
    url = f"{BASE_URL}/{FILTER_ID}/{REGION}/index_{res}.json"
    r = requests.get(url)
    r.raise_for_status()
    return r.json()["timestamps"]

def get_chunk(ts, res="quarterhour"):
    url = f"{BASE_URL}/{FILTER_ID}/{REGION}/{FILTER_ID}_{REGION}_{res}_{ts}.json"
    r = requests.get(url)
    r.raise_for_status()
    return r.json()["series"]

start_ts = int(datetime.strptime(START_DATE, "%Y-%m-%d").timestamp() * 1000)
end_ts   = int(datetime.strptime(END_DATE, "%Y-%m-%d").timestamp() * 1000)

timestamps = get_timestamps()
timestamps = [ts for ts in timestamps if start_ts <= ts <= end_ts]

data = []
for i, ts in enumerate(timestamps, 1):
    print(f"{i}/{len(timestamps)}", end=" ")
    chunk = get_chunk(ts)
    if chunk:
        data.extend(chunk)
        print("Downloaded")
    else:
        print("No data")
    time.sleep(0.3)

df = pd.DataFrame(data, columns=["timestamp_ms", "generation_mw"])
df["time_utc"] = pd.to_datetime(df["timestamp_ms"], unit="ms")
df = df.drop_duplicates("timestamp_ms").sort_values("time_utc")

df.to_csv(OUTPUT_DIR / "germany_pv.csv", index=False)

print("Saved:", OUTPUT_DIR / "germany_pv.csv")
