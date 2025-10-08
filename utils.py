import os
import json
from pathlib import Path
from dotenv import load_dotenv
from typing import List
from datetime import datetime
from zoneinfo import ZoneInfo

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

def to_local_time(utc_iso: str, tz_name: str = "Europe/Budapest") -> str:
    """
    Convert UTC ISO timestamp to local timezone.
    
    Args:
        utc_iso: UTC timestamp in ISO format (e.g., "2024-01-15T18:00:00+00:00")
        tz_name: Target timezone name (default: "Europe/Budapest")
    
    Returns:
        Formatted local time string (e.g., "2024-01-15 19:00:00 CET")
    """
    try:
        # Parse the UTC timestamp
        utc_dt = datetime.fromisoformat(utc_iso.replace('Z', '+00:00'))
        
        # Convert to local timezone
        local_tz = ZoneInfo(tz_name)
        local_dt = utc_dt.astimezone(local_tz)
        
        # Format the output
        return local_dt.strftime("%Y-%m-%d %H:%M:%S %Z")
    except Exception as e:
        log("WARNING", f"Failed to convert timezone for {utc_iso}: {e}")
        return utc_iso