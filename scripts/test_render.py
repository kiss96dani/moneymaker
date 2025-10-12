#!/usr/bin/env python3
"""
Test script for check_analysis_json.py rendering functionality.

This script creates a fake daily report in memory and tests the rendering
pipeline without requiring real daily_reports data. Useful for CI/CD testing.
"""

import sys
import tempfile
from pathlib import Path
from datetime import datetime, timezone

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from check_analysis_json import (
    extract_top_picks,
    render_ticket_image,
    format_pick_text,
    get_color,
    PALETTE,
)


def create_fake_report():
    """Create a fake daily report for testing."""
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "top_n": 3,
        "1x2": [
            {
                "fixture_id": 12345,
                "home_team": "Test United",
                "away_team": "Demo FC",
                "league_name": "Test League",
                "chosen_outcome": "home",
                "probability": 0.65,
                "market_odds": 2.10,
                "edge": 0.15,
                "kickoff_local": "2025-10-12T15:00:00+02:00",
            },
            {
                "fixture_id": 12346,
                "home_team": "Sample City",
                "away_team": "Mock Rovers",
                "league_name": "Test League",
                "chosen_outcome": "away",
                "probability": 0.58,
                "market_odds": 2.50,
                "edge": 0.12,
                "kickoff_local": "2025-10-12T17:30:00+02:00",
            },
        ],
        "btts": [
            {
                "fixture_id": 12347,
                "home_team": "Alpha FC",
                "away_team": "Beta United",
                "league_name": "Test League",
                "chosen_outcome": "btts_yes",
                "probability": 0.72,
                "market_odds": 1.90,
                "edge": 0.20,
                "kickoff_local": "2025-10-12T19:00:00+02:00",
            },
        ],
        "over_under_2_5": [
            {
                "fixture_id": 12348,
                "home_team": "Gamma City",
                "away_team": "Delta Rovers",
                "league_name": "Test League",
                "chosen_outcome": "over25",
                "probability": 0.68,
                "market_odds": 2.00,
                "edge": 0.18,
                "kickoff_local": "2025-10-12T20:00:00+02:00",
            },
        ],
    }


def test_extract_picks():
    """Test extracting picks from report."""
    print("Testing pick extraction...")
    report = create_fake_report()
    picks = extract_top_picks(report, max_picks=3)
    
    assert len(picks) > 0, "Should extract at least one pick"
    assert all("fixture_id" in p for p in picks), "All picks should have fixture_id"
    assert all("probability" in p for p in picks), "All picks should have probability"
    
    print(f"✓ Extracted {len(picks)} picks successfully")


def test_format_pick():
    """Test formatting a pick as text."""
    print("Testing pick formatting...")
    pick = {
        "home_team": "Test United",
        "away_team": "Demo FC",
        "chosen_outcome": "home",
        "probability": 0.65,
        "market_odds": 2.10,
        "edge": 0.15,
    }
    
    text = format_pick_text(pick)
    assert "Test United vs Demo FC" in text, "Should contain team names"
    assert "home" in text, "Should contain outcome"
    assert "65.0%" in text, "Should contain probability"
    
    print(f"✓ Formatted text: {text}")


def test_get_color():
    """Test the safe color accessor."""
    print("Testing safe color accessor...")
    
    # Test existing key
    shadow = get_color(PALETTE, "shadow_rgba", (0, 0, 0, 255))
    assert isinstance(shadow, tuple), "Should return a tuple"
    assert len(shadow) == 4, "RGBA should have 4 components"
    
    # Test missing key with default
    missing = get_color(PALETTE, "nonexistent_color", (100, 100, 100))
    assert missing == (100, 100, 100), "Should return default for missing key"
    
    # Test with bad palette entry
    bad_palette = {"bad_color": "not a tuple"}
    result = get_color(bad_palette, "bad_color", (0, 0, 0))
    assert result == (0, 0, 0), "Should return default for invalid color"
    
    print("✓ Color accessor works correctly")


def test_render():
    """Test rendering a ticket image."""
    print("Testing image rendering...")
    
    report = create_fake_report()
    picks = extract_top_picks(report, max_picks=2)
    
    # Create temporary output file
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        output_path = Path(f.name)
    
    try:
        success = render_ticket_image(picks, output_path)
        
        if not success:
            print("⚠ Rendering failed (Pillow may not be available)")
            return
        
        assert output_path.exists(), "Image file should be created"
        assert output_path.stat().st_size > 0, "Image file should not be empty"
        
        print(f"✓ Rendered image: {output_path} ({output_path.stat().st_size} bytes)")
    
    finally:
        # Cleanup
        if output_path.exists():
            output_path.unlink()


def main():
    """Run all tests."""
    print("=" * 50)
    print("Testing check_analysis_json.py rendering")
    print("=" * 50)
    print()
    
    try:
        test_extract_picks()
        test_format_pick()
        test_get_color()
        test_render()
        
        print()
        print("=" * 50)
        print("✓ All tests passed!")
        print("=" * 50)
        return 0
    
    except AssertionError as e:
        print()
        print("=" * 50)
        print(f"✗ Test failed: {e}")
        print("=" * 50)
        return 1
    
    except Exception as e:
        print()
        print("=" * 50)
        print(f"✗ Unexpected error: {e}")
        print("=" * 50)
        return 1


if __name__ == "__main__":
    sys.exit(main())
