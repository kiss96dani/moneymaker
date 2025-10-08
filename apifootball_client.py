import aiohttp
import asyncio
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

API_BASE = "https://v3.football.api-sports.io"

from utils import log

class APIFootballClient:
    def __init__(self, api_key: str, session: Optional[aiohttp.ClientSession]=None):
        self.api_key = api_key
        self._session = session
        self._local_session = False
        self._team_fixtures_cache = {}

    async def _get_session(self):
        if self._session:
            return self._session
        self._session = aiohttp.ClientSession(headers={"x-apisports-key": self.api_key})
        self._local_session = True
        return self._session

    async def close(self):
        if self._local_session and self._session:
            await self._session.close()
            self._session = None

    async def _get(self, path: str, params: dict = None) -> dict:
        session = await self._get_session()
        url = f"{API_BASE}{path}"
        try:
            async with session.get(url, params=params or {}) as resp:
                data = await resp.json()
                if resp.status != 200:
                    log("WARNING", f"API returned {resp.status} for {url} params={params} body={data}")
                return data
        except Exception as e:
            log("ERROR", f"HTTP error for {url}: {e}")
            raise

    async def get_fixtures_by_date(self, date_iso: str) -> List[dict]:
        # GET /fixtures?date=YYYY-MM-DD
        res = await self._get("/fixtures", params={"date": date_iso})
        return res.get("response", [])

    async def get_fixture_by_id(self, fixture_id: int) -> Optional[dict]:
        res = await self._get("/fixtures", params={"id": fixture_id})
        arr = res.get("response", [])
        return arr[0] if arr else None

    async def get_team_recent_fixtures(self, team_id: int, limit: int = 10) -> List[dict]:
        # Cache per runtime to reduce rate usage
        key = f"{team_id}:{limit}"
        if key in self._team_fixtures_cache:
            return self._team_fixtures_cache[key]
        # GET /fixtures?team={team}&last={limit}
        res = await self._get("/fixtures", params={"team": team_id, "last": limit})
        fixtures = res.get("response", [])
        self._team_fixtures_cache[key] = fixtures
        await asyncio.sleep(0.1)  # small throttle
        return fixtures

    async def get_odds_for_fixture(self, fixture_id: int) -> Optional[List[dict]]:
        # GET /odds?fixture={fixture}
        res = await self._get("/odds", params={"fixture": fixture_id})
        return res.get("response", [])

    async def get_leagues(self) -> Dict[str, Any]:
        res = await self._get("/leagues")
        return res

    async def get_fixtures_by_league_and_date(
        self, 
        league_id: int, 
        from_date: str, 
        to_date: str
    ) -> List[dict]:
        """
        Get fixtures for a specific league within a date range.
        
        Args:
            league_id: League ID
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)
            
        Returns:
            List of fixture dictionaries
        """
        res = await self._get(
            "/fixtures", 
            params={
                "league": league_id,
                "from": from_date,
                "to": to_date
            }
        )
        return res.get("response", [])

    # Note: add more helper methods as needed for players/injuries/... later.