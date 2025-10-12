#!/usr/bin/env python3
"""
Check and render betting analysis results from daily_reports JSON files.

This script reads daily reports from the daily_reports/ directory and can:
- List the top picks from the analysis
- Render ticket images with the picks for sharing
- Send the images (when --send is specified)

CLI Options:
  --max-picks N    Maximum number of picks to include (default: 3)
  --send           Send the rendered images (e.g., via email/webhook)
  --date           Specific date to process (YYYY-MM-DD), defaults to today
  --font           Path to custom TTF font file for rendering
  --template       Path to custom template image for ticket background

Palette Keys:
  - shadow_rgba: Shadow color for ticket image rendering (RGBA tuple)
  - outer_rgba: Outer border/background color (RGBA tuple)
  - inner_rgba: Inner content area color (RGBA tuple)
  - text_color: Main text color (RGB tuple)
  - header_color: Header text color (RGB tuple)

Usage examples:
  python check_analysis_json.py
  python check_analysis_json.py --max-picks 5
  python check_analysis_json.py --max-picks 3 --send
  python check_analysis_json.py --date 2025-10-12 --font /path/to/font.ttf
"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("Warning: Pillow not available. Image rendering disabled.", file=sys.stderr)

from utils import log, read_json, ensure_dirs

# Directories
DAILY_DIR = Path("daily_reports")
OUTPUT_DIR = Path("rendered_tickets")
ensure_dirs([DAILY_DIR, OUTPUT_DIR])

# Color palette for ticket rendering
# Each color should be a tuple: RGBA for colors with alpha, RGB for opaque colors
PALETTE = {
    "shadow_rgba": (0, 0, 0, 210),      # Semi-transparent black shadow
    "outer_rgba": (255, 255, 255, 255),  # White outer background
    "inner_rgba": (240, 248, 255, 255),  # Alice blue inner area
    "text_color": (0, 0, 0),             # Black text
    "header_color": (25, 25, 112),       # Midnight blue headers
    "highlight_color": (34, 139, 34),    # Forest green for highlights
}


def get_color(palette: Dict[str, Any], key: str, default: Tuple[int, ...]) -> Tuple[int, ...]:
    """
    Safely retrieve a color from the palette with fallback.
    
    Args:
        palette: Color palette dictionary
        key: Color key to retrieve
        default: Default color tuple to return if key is missing
    
    Returns:
        Color tuple (RGB or RGBA)
    """
    color = palette.get(key, default)
    
    # Validate it's a tuple of integers
    if not isinstance(color, (tuple, list)):
        log("WARNING", f"Invalid color format for '{key}', using default")
        return default
    
    # Ensure all elements are integers
    try:
        return tuple(int(c) for c in color)
    except (ValueError, TypeError):
        log("WARNING", f"Invalid color values for '{key}', using default")
        return default


def find_daily_report(date_str: Optional[str] = None) -> Optional[Path]:
    """
    Find the daily report file for a given date.
    
    Args:
        date_str: Date string in YYYY-MM-DD format, or None for today
    
    Returns:
        Path to the report file, or None if not found
    """
    if date_str is None:
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    
    report_path = DAILY_DIR / f"top_markets_{date_str}.json"
    
    if not report_path.exists():
        log("WARNING", f"Report not found: {report_path}")
        return None
    
    return report_path


def load_report(report_path: Path) -> Optional[Dict[str, Any]]:
    """
    Load a daily report JSON file.
    
    Args:
        report_path: Path to the report file
    
    Returns:
        Report data as dictionary, or None if loading fails
    """
    try:
        return read_json(report_path)
    except Exception as e:
        log("ERROR", f"Failed to load report {report_path}: {e}")
        return None


def extract_top_picks(report: Dict[str, Any], max_picks: int = 3) -> List[Dict[str, Any]]:
    """
    Extract top picks from the daily report.
    
    Args:
        report: Daily report dictionary
        max_picks: Maximum number of picks to extract from each category
    
    Returns:
        List of pick dictionaries with all relevant information
    """
    picks = []
    
    # Extract from each category
    for category in ["1x2", "btts", "over_under_2_5"]:
        category_picks = report.get(category, [])
        for pick in category_picks[:max_picks]:
            pick_info = {
                "category": category,
                "fixture_id": pick.get("fixture_id"),
                "home_team": pick.get("home_team"),
                "away_team": pick.get("away_team"),
                "league_name": pick.get("league_name"),
                "chosen_outcome": pick.get("chosen_outcome"),
                "probability": pick.get("probability"),
                "market_odds": pick.get("market_odds"),
                "edge": pick.get("edge"),
                "kickoff_local": pick.get("kickoff_local"),
            }
            picks.append(pick_info)
    
    return picks


def format_pick_text(pick: Dict[str, Any]) -> str:
    """
    Format a pick as a human-readable string.
    
    Args:
        pick: Pick dictionary
    
    Returns:
        Formatted string
    """
    teams = f"{pick['home_team']} vs {pick['away_team']}"
    outcome = pick['chosen_outcome']
    prob = pick['probability'] * 100
    
    parts = [f"{teams} - {outcome} ({prob:.1f}%)"]
    
    if pick.get('market_odds'):
        parts.append(f"odds: {pick['market_odds']:.2f}")
    
    if pick.get('edge') is not None:
        parts.append(f"edge: {pick['edge']:.1%}")
    
    return " | ".join(parts)


def print_picks(picks: List[Dict[str, Any]]) -> None:
    """
    Print picks to console in a formatted manner.
    
    Args:
        picks: List of pick dictionaries
    """
    if not picks:
        print("No picks found.")
        return
    
    print("\n=== Top Picks ===")
    for i, pick in enumerate(picks, 1):
        print(f"{i}. {format_pick_text(pick)}")
    print("==================\n")


def render_ticket_image(
    picks: List[Dict[str, Any]],
    output_path: Path,
    font_path: Optional[str] = None,
    template_path: Optional[str] = None
) -> bool:
    """
    Render picks as a ticket image.
    
    Args:
        picks: List of pick dictionaries
        output_path: Where to save the rendered image
        font_path: Optional path to custom TTF font
        template_path: Optional path to template image
    
    Returns:
        True if rendering succeeded, False otherwise
    """
    if not PIL_AVAILABLE:
        log("ERROR", "Pillow is required for image rendering")
        return False
    
    if not picks:
        log("WARNING", "No picks to render")
        return False
    
    try:
        # Image dimensions
        outer_w, outer_h = 800, 600
        margin = 20
        
        # Get colors using safe accessor
        shadow_color = get_color(PALETTE, "shadow_rgba", (0, 0, 0, 210))
        outer_color = get_color(PALETTE, "outer_rgba", (255, 255, 255, 255))
        inner_color = get_color(PALETTE, "inner_rgba", (240, 248, 255, 255))
        text_color = get_color(PALETTE, "text_color", (0, 0, 0))
        header_color = get_color(PALETTE, "header_color", (25, 25, 112))
        
        # Create image with shadow
        osh = Image.new("RGBA", (outer_w, outer_h), shadow_color)
        img = Image.new("RGBA", (outer_w, outer_h), outer_color)
        
        # Draw inner rectangle
        draw = ImageDraw.Draw(img)
        draw.rectangle(
            [(margin, margin), (outer_w - margin, outer_h - margin)],
            fill=inner_color,
            outline=header_color,
            width=2
        )
        
        # Setup font
        try:
            if font_path and Path(font_path).exists():
                title_font = ImageFont.truetype(font_path, 24)
                text_font = ImageFont.truetype(font_path, 14)
            else:
                # Try common system fonts
                title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
                text_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        except Exception:
            # Fallback to default font
            title_font = ImageFont.load_default()
            text_font = ImageFont.load_default()
        
        # Draw title
        title = "Top Betting Picks"
        title_y = margin + 10
        draw.text((outer_w // 2, title_y), title, fill=header_color, font=title_font, anchor="mt")
        
        # Draw picks
        y_offset = title_y + 40
        line_height = 25
        
        for i, pick in enumerate(picks, 1):
            pick_text = f"{i}. {format_pick_text(pick)}"
            
            # Wrap text if too long
            if len(pick_text) > 70:
                # Simple wrapping: split at 70 chars
                lines = [pick_text[i:i+70] for i in range(0, len(pick_text), 70)]
                for line in lines:
                    draw.text((margin + 10, y_offset), line, fill=text_color, font=text_font)
                    y_offset += line_height
            else:
                draw.text((margin + 10, y_offset), pick_text, fill=text_color, font=text_font)
                y_offset += line_height
            
            # Add spacing between picks
            y_offset += 5
        
        # Composite shadow and main image
        final = Image.alpha_composite(osh, img)
        
        # Save image
        final.convert("RGB").save(output_path, "PNG")
        log("INFO", f"Rendered ticket image: {output_path}")
        return True
        
    except Exception as e:
        log("ERROR", f"Failed to render ticket image: {e}")
        return False


def send_image(image_path: Path) -> bool:
    """
    Send the rendered image (placeholder for email/webhook integration).
    
    Args:
        image_path: Path to the image file
    
    Returns:
        True if sending succeeded, False otherwise
    """
    # Placeholder implementation
    log("INFO", f"Would send image: {image_path}")
    log("WARNING", "Actual sending not implemented yet (placeholder)")
    return True


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Check and render betting analysis results from daily reports",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--max-picks",
        type=int,
        default=3,
        help="Maximum number of picks to include per category (default: 3)"
    )
    
    parser.add_argument(
        "--send",
        action="store_true",
        help="Send the rendered images via configured method"
    )
    
    parser.add_argument(
        "--date",
        type=str,
        help="Specific date to process (YYYY-MM-DD), defaults to today"
    )
    
    parser.add_argument(
        "--font",
        type=str,
        help="Path to custom TTF font file for rendering"
    )
    
    parser.add_argument(
        "--template",
        type=str,
        help="Path to custom template image for ticket background"
    )
    
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Skip image rendering, only print picks to console"
    )
    
    args = parser.parse_args()
    
    # Find and load report
    report_path = find_daily_report(args.date)
    if not report_path:
        log("ERROR", "No report found for the specified date")
        sys.exit(1)
    
    report = load_report(report_path)
    if not report:
        log("ERROR", "Failed to load report")
        sys.exit(1)
    
    # Extract picks
    picks = extract_top_picks(report, max_picks=args.max_picks)
    
    # Print to console
    print_picks(picks)
    
    # Render image if requested
    if not args.no_render:
        date_str = args.date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        output_path = OUTPUT_DIR / f"ticket_{date_str}.png"
        
        success = render_ticket_image(
            picks,
            output_path,
            font_path=args.font,
            template_path=args.template
        )
        
        if not success:
            log("ERROR", "Image rendering failed")
            sys.exit(1)
        
        # Send if requested
        if args.send:
            if not send_image(output_path):
                log("ERROR", "Failed to send image")
                sys.exit(1)
    
    log("INFO", "Processing complete")


if __name__ == "__main__":
    main()
