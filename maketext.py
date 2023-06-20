import time
from datetime import datetime

filename = "20230620data.txt"
target_time = datetime(2023, 6, 20, 8, 30)

while datetime.now() < target_time:
    time.sleep(1)

with open(filename, "w") as f:
    f.write("")