#!/usr/bin/env python3
"""
Entrypoint CLI for the automated betting analyzer.

Usage examples (see README.md for full list):
  python betting.py --fetch --analyze
  python betting.py --analyze --fixture-ids 12345,67890
"""
from __future__ import annotations
import argparse
import asyncio
import os
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from typing import List, Optional

from apifootball_client import APIFootballClient
from analyzer import Analyzer
from predictor import Predictor
from utils import load_env, ensure_dirs, read_json, write_json, log

# new summarizer module
from summarizer import summarize_results

load_env()  # load .env into environment if present

DATA_DIR = Path("data")
FIXTURES_DIR = DATA_DIR / "fixtures"
ANALYSIS_DIR = DATA_DIR / "analysis"
DAILY_DIR = Path("daily_reports")
CONFIG_DIR = Path("config")
ensure_dirs([DATA_DIR, FIXTURES_DIR, ANALYSIS_DIR, CONFIG_DIR, DAILY_DIR])

DEFAULT_FETCH_DAYS = int(os.getenv("FETCH_DAYS_AHEAD", "3"))

async def fetch_fixtures(client: APIFootballClient, days_ahead: int, limit: Optional[int]=None, refetch_missing: bool=False):
    # Use timezone-aware now in UTC
    today = datetime.now(timezone.utc).date()
    fixtures = []
    day = 0
    while day < days_ahead:
        target = today + timedelta(days=day)
        day += 1
        try:
            res = await client.get_fixtures_by_date(target.isoformat())
        except Exception as e:
            log("ERROR", f"Failed to fetch fixtures for {target}: {e}")
            continue
        for f in res:
            fixtures.append(f)
            fid = f.get("fixture", {}).get("id")
            if not fid:
                continue
            path = FIXTURES_DIR / f"{fid}.json"
            if path.exists() and not refetch_missing:
                continue
            write_json(path, f)
            if limit and len(fixtures) >= limit:
                return fixtures
    return fixtures

async def analyze_fixtures(client: APIFootballClient, fixture_ids: Optional[List[int]] = None, limit: Optional[int]=None, use_ml: bool = False, use_mc: bool = False, mc_iters: int = 10000, include_past: bool = False):
    # Analyzer currently doesn't require client; instantiate accordingly
    analyzer = Analyzer()
    predictor = Predictor()

    to_analyze = []
    if fixture_ids:
        for fid in fixture_ids:
            path = FIXTURES_DIR / f"{fid}.json"
            if not path.exists():
                log("WARNING", f"Fixture {fid} not found in local cache; attempting to download")
                fdata = await client.get_fixture_by_id(fid)
                if not fdata:
                    log("ERROR", f"Could not retrieve fixture {fid}")
                    continue
                write_json(path, fdata)
            to_analyze.append(read_json(path))
    else:
        files = sorted(FIXTURES_DIR.glob("*.json"))
        for p in files[:limit]:
            to_analyze.append(read_json(p))

    results = []
    now_utc = datetime.now(timezone.utc)
    
    for f in to_analyze:
        try:
            fixture_meta = analyzer.extract_fixture_meta(f)
            
            # Skip past fixtures unless --include-past is set
            if not include_past:
                kickoff_str = fixture_meta.get("kickoff_utc")
                if kickoff_str:
                    try:
                        kickoff_dt = datetime.fromisoformat(kickoff_str.replace('Z', '+00:00'))
                        if kickoff_dt < now_utc:
                            log("INFO", f"Skipping past fixture {fixture_meta['fixture_id']} (kickoff: {kickoff_str})")
                            continue
                    except Exception as e:
                        log("WARNING", f"Could not parse kickoff time for fixture {fixture_meta['fixture_id']}: {e}")
            
            home_team = fixture_meta["home_team_id"]
            away_team = fixture_meta["away_team_id"]

            # collect recent fixtures for both teams
            home_recent = await client.get_team_recent_fixtures(home_team, limit=10)
            away_recent = await client.get_team_recent_fixtures(away_team, limit=10)

            home_form = analyzer.calculate_last_n_form(home_recent, team_id=home_team, n=5)
            away_form = analyzer.calculate_last_n_form(away_recent, team_id=away_team, n=5)

            # Get head-to-head data
            h2h_fixtures = await client.get_head_to_head(home_team, away_team, last=10)
            h2h_summary = []
            for h2h_fix in h2h_fixtures[:10]:  # Limit to last 10
                h2h_teams = h2h_fix.get("teams", {})
                h2h_goals = h2h_fix.get("goals", {})
                h2h_fixture_info = h2h_fix.get("fixture", {})
                h2h_summary.append({
                    "fixture_id": h2h_fixture_info.get("id"),
                    "date": h2h_fixture_info.get("date"),
                    "home_team": h2h_teams.get("home", {}).get("name"),
                    "away_team": h2h_teams.get("away", {}).get("name"),
                    "home_goals": h2h_goals.get("home"),
                    "away_goals": h2h_goals.get("away"),
                })

            lambda_h, lambda_a = predictor.compute_lambdas(home_form, away_form)

            # Pass use_mc flag and mc_iters to predictor
            try:
                model_probs, matrix = predictor.match_probabilities(
                    lambda_h, lambda_a, 
                    home_form=home_form, 
                    away_form=away_form,
                    use_mc=use_mc,
                    mc_iters=mc_iters
                )
            except TypeError:
                # Fallback for older predictor signature
                model_probs, matrix = predictor.match_probabilities(lambda_h, lambda_a)

            market = {}
            odds_data = await client.get_odds_for_fixture(fixture_meta["fixture_id"])
            if odds_data:
                market = predictor.parse_market_odds(odds_data)

            edge_kelly = predictor.compute_edges_and_kelly(model_probs, market)

            # Import timezone conversion utility
            from utils import to_local_time
            kickoff_local = to_local_time(fixture_meta["kickoff_utc"])

            analysis = {
                "fixture_id": fixture_meta["fixture_id"],
                "kickoff_utc": fixture_meta["kickoff_utc"],
                "kickoff_local": kickoff_local,
                "league_id": fixture_meta["league_id"],
                "league_name": fixture_meta["league_name"],
                "home_team": fixture_meta["home_team_name"],
                "away_team": fixture_meta["away_team_name"],
                "model_probs": model_probs,
                "lambda_home": lambda_h,
                "lambda_away": lambda_a,
                "market": market,
                "edge_kelly": edge_kelly,
                "home_recent": home_form,
                "away_recent": away_form,
                "h2h": h2h_summary,
            }

            results.append(analysis)
            out_path = ANALYSIS_DIR / f"{fixture_meta['fixture_id']}.analysis.json"
            write_json(out_path, analysis)
            log("INFO", f"Analyzed fixture {fixture_meta['fixture_id']} -> {out_path}")
        except Exception as e:
            log("ERROR", f"Error analyzing fixture {f.get('fixture', {}).get('id')}: {e}")

    return results

async def main_async(args):
    key = os.getenv("API_FOOTBALL_KEY")
    if not key:
        log("ERROR", "API_FOOTBALL_KEY not set in environment (.env or env variable)")
        return

    client = APIFootballClient(api_key=key)

    try:
        # Handle Tippmix scraping
        if args.tippmix_scrape:
            try:
                from tippmix_scraper import scrape_tippmix
                count = scrape_tippmix()
                log("INFO", f"Tippmix scraper completed: {count} matches found")
            except Exception as e:
                log("ERROR", f"Tippmix scraper failed: {e}")
        
        # Handle ML training
        if args.train_models:
            try:
                from models import ModelTrainer
                
                # Parse training parameters
                train_from = args.train_from
                train_to = args.train_to
                leagues_str = args.leagues or "39,61"
                league_ids = [int(x.strip()) for x in leagues_str.split(",") if x.strip().isdigit()]
                
                if not train_from or not train_to:
                    log("ERROR", "Training requires --train-from and --train-to dates (YYYY-MM-DD)")
                    return
                
                log("INFO", f"Starting ML training: {train_from} to {train_to}, leagues: {league_ids}")
                trainer = ModelTrainer(client)
                await trainer.train(
                    start_date=train_from,
                    end_date=train_to,
                    league_ids=league_ids
                )
                log("INFO", "ML training completed")
                return
            except Exception as e:
                log("ERROR", f"ML training failed: {e}")
                import traceback
                traceback.print_exc()
                return
        
        if args.reload_leagues:
            # placeholder: download leagues tiers, not implemented in depth
            leagues = await client.get_leagues()
            write_json(CONFIG_DIR / "leagues_tiers.json", leagues)
            log("INFO", f"Reloaded {len(leagues.get('response',[]))} leagues to config/leagues_tiers.json")

        if args.fetch:
            days = args.days_ahead or DEFAULT_FETCH_DAYS
            await fetch_fixtures(client, days, limit=args.limit, refetch_missing=args.refetch_missing)

        if args.analyze:
            fids = None
            if args.fixture_ids:
                fids = [int(x.strip()) for x in args.fixture_ids.split(",") if x.strip().isdigit()]
            
            # Pass include_past and require_tippmix flags
            results = await analyze_fixtures(
                client, 
                fixture_ids=fids, 
                limit=args.limit, 
                use_ml=args.use_ml, 
                use_mc=args.use_mc, 
                mc_iters=args.mc_iters,
                include_past=args.include_past
            )
            log("INFO", f"Completed analysis for {len(results)} fixtures")

            # Print full analysis JSON (existing behaviour)
            print(json.dumps(results, indent=2, ensure_ascii=False))

            # Summarize top markets: prints and writes a daily report in daily_reports/
            try:
                report = summarize_results(results, top_n=3, require_tippmix=args.require_tippmix)
                # persist the report
                now = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                report_path = DAILY_DIR / f"top_markets_{now}.json"
                write_json(report_path, report)
                log("INFO", f"Top markets report written to {report_path}")
            except Exception as e:
                log("ERROR", f"Failed to summarize top markets: {e}")
    finally:
        # ensure session is closed to avoid warnings
        try:
            await client.close()
        except Exception:
            pass

def parse_args():
    p = argparse.ArgumentParser(description="Automated betting analysis (API-Football)")
    p.add_argument("--fetch", action="store_true", help="Fetch fixtures from API-Football")
    p.add_argument("--analyze", action="store_true", help="Run analysis on cached or fetched fixtures")
    p.add_argument("--fixture-ids", type=str, help="Comma-separated fixture ids to analyze")
    p.add_argument("--reload-leagues", action="store_true", help="Reload leagues classification")
    p.add_argument("--cleanup-stale", action="store_true", help="Cleanup stale fixture files (not implemented)")
    p.add_argument("--refetch-missing", action="store_true", help="Refetch missing fixtures even if local exists")
    p.add_argument("--days-ahead", type=int, help="Override FETCH_DAYS_AHEAD")
    p.add_argument("--limit", type=int, help="Limit number of fixtures to fetch/analyze")
    p.add_argument("--use-ml", action="store_true", help="Use ML models if available (fallback to Poisson if not)")
    p.add_argument("--use-mc", action="store_true", help="Use Monte Carlo simulation for probability calculations")
    p.add_argument("--mc-iters", type=int, default=10000, help="Number of Monte Carlo iterations (default: 10000)")
    
    # Tippmix integration
    p.add_argument("--tippmix-scrape", action="store_true", help="Run Tippmix scraper before analysis")
    p.add_argument("--require-tippmix", action="store_true", help="Only include tips that are available on Tippmix")
    
    # ML training
    p.add_argument("--train-models", action="store_true", help="Train ML models on historical data")
    p.add_argument("--train-from", type=str, help="Start date for training data (YYYY-MM-DD)")
    p.add_argument("--train-to", type=str, help="End date for training data (YYYY-MM-DD)")
    p.add_argument("--leagues", type=str, help="Comma-separated league IDs for training (default: 39,61)")
    
    # Time filtering
    p.add_argument("--include-past", action="store_true", help="Include past fixtures (by default only future fixtures)")
    
    return p.parse_args()

def main():
    args = parse_args()
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        log("WARNING", "Interrupted by user")

if __name__ == "__main__":
    main()
