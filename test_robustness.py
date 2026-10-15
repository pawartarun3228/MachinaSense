
import json
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
PYTHON = BASE_DIR / ".venv312" / "Scripts" / "python.exe"
DATA_PATH = BASE_DIR / "data" / "ai4i_dataset.csv"
MODEL_PATH = BASE_DIR / "models" / "machinasense.joblib"
BASE_URL = "http://127.0.0.1:5000"


def check(name, passed, details=""):
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {name}")
    if details:
        print(f"       {details}")


def post_json(path, payload):
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        BASE_URL + path,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def main():
    print("\nMachinaSense Robustness Tests\n")

    try:
        with urllib.request.urlopen(BASE_URL + "/api/health", timeout=5) as r:
            health = json.loads(r.read())
        check("Backend health", r.status == 200, str(health))
    except Exception as exc:
        check("Backend health", False, str(exc))
        print("\nStart Flask with: python app.py")
        return

    check("Dataset exists", DATA_PATH.exists())
    check("Trained model exists", MODEL_PATH.exists())

    if not DATA_PATH.exists() or not MODEL_PATH.exists():
        return

    # Use a real row from the project's dataset.
    df = pd.read_csv(DATA_PATH)
    features = [
        "Type",
        "Air_temperature",
        "Process_temperature",
        "Rotational_speed",
        "Torque",
        "Tool_wear",
    ]

    missing = [c for c in features if c not in df.columns]
    if missing:
        check("Dataset feature availability", False, str(missing))
        return

    row = df.iloc[0][features].to_dict()

    for col in features:
        if col != "Type":
            row[col] = float(row[col])

    # 1. Valid input
    status, result = post_json("/api/predict", row)
    check(
        "Valid sensor input",
        status == 200 and "failure_probability" in result,
        f"HTTP {status}",
    )

    # 2. Missing required feature
    incomplete = dict(row)
    incomplete.pop("Torque")
    status, result = post_json("/api/predict", incomplete)
    check(
        "Missing feature rejected",
        status == 400 and "error" in result,
        f"HTTP {status}: {result.get('error')}",
    )

    # 3. Invalid numeric value
    invalid = dict(row)
    invalid["Torque"] = "not-a-number"
    status, result = post_json("/api/predict", invalid)
    check(
        "Invalid numeric value rejected",
        status == 400 and "error" in result,
        f"HTTP {status}: {result.get('error')}",
    )

    # 4. Missing numeric value
    empty = dict(row)
    empty["Torque"] = None
    status, result = post_json("/api/predict", empty)
    check(
        "Empty numeric value rejected",
        status == 400 and "error" in result,
        f"HTTP {status}: {result.get('error')}",
    )

    # 5. Non-JSON body is tested using the health endpoint separately.
    # This tests a small sensor perturbation, not model accuracy.
    noisy = dict(row)
    noisy["Air_temperature"] += 0.1
    status, result = post_json("/api/predict", noisy)
    check(
        "Small sensor perturbation handled",
        status == 200 and "failure_probability" in result,
        f"HTTP {status}",
    )

    print("\nTests completed. Review any FAIL results above.")


if __name__ == "__main__":
    main()
