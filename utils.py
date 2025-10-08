import os
import json
from pathlib import Path
from dotenv import load_dotenv
from typing import List

def load_env(path: str = ".env"):
    if Path(path).exists():
        load_dotenv(path)

def ensure_dirs(paths: List[Path]):
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)

def write_json(path: Path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def read_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def log(level: str, msg: str):
    print(f"[{level}] {msg}")