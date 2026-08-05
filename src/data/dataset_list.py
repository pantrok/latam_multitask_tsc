from __future__ import annotations

import csv
from pathlib import Path


def get_sensor_datasets(csv_path: str = "assets/DataSummary.csv") -> list[str]:
    csv_file = Path(csv_path)
    if not csv_file.exists():
        raise FileNotFoundError(
            f"Missing {csv_path}. Place the UCR 2018 DataSummary.csv there."
        )

    sensor = []
    with csv_file.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("Type", "").strip().lower() == "sensor":
                sensor.append(row["Name"].strip())

    return sorted(set(sensor))


def main() -> None:
    datasets = get_sensor_datasets()
    print(f"Sensor datasets: {len(datasets)}")
    for name in datasets:
        print(name)


if __name__ == "__main__":
    main()
