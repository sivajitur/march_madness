"""
Shared utility functions for the March Madness prediction system.

Handles team name normalization, seed parsing, and round labeling.
"""


# ── Team Name Normalization Map ───────────────────────────────────────────────
# Maps variant names found across datasets to a single canonical name.
# The canonical name should match the primary dataset (Andrew Sundberg's CBB).
TEAM_NAME_MAP = {
    # A
    "Alabama-Birmingham": "UAB",
    "Albany": "Albany",
    "Appalachian St.": "Appalachian State",
    "Arizona St.": "Arizona State",
    "Arkansas-Little Rock": "Little Rock",
    "Arkansas Little Rock": "Little Rock",
    "Arkansas Pine Bluff": "Arkansas-Pine Bluff",

    # B
    "Boise St.": "Boise State",
    "Bowling Green St.": "Bowling Green",
    "BYU": "Brigham Young",
    "Brigham Young": "Brigham Young",

    # C
    "Cal St. Bakersfield": "CSU Bakersfield",
    "Cal St. Fullerton": "Cal State Fullerton",
    "Cal State Fullerton": "Cal State Fullerton",
    "Central Connecticut": "Central Connecticut State",
    "Central Connecticut St.": "Central Connecticut State",
    "Central Florida": "UCF",
    "Charleston": "College of Charleston",
    "Col. of Charleston": "College of Charleston",
    "Colorado St.": "Colorado State",
    "Connecticut": "UConn",
    "UCONN": "UConn",
    "Coastal Car.": "Coastal Carolina",
    "Coastal Carolina": "Coastal Carolina",

    # D
    "Detroit": "Detroit Mercy",
    "Detroit Mercy": "Detroit Mercy",

    # E
    "East Tennessee St.": "ETSU",
    "East Tennessee State": "ETSU",
    "Eastern Washington": "Eastern Washington",

    # F
    "Fairleigh Dickinson": "Fairleigh Dickinson",
    "FDU": "Fairleigh Dickinson",
    "Florida Atlantic": "FAU",
    "Florida Gulf Coast": "FGCU",
    "Florida Int'l": "FIU",
    "Florida St.": "Florida State",
    "Fresno St.": "Fresno State",
    "Fort Wayne": "Purdue Fort Wayne",

    # G
    "George Mason": "George Mason",
    "Georgia St.": "Georgia State",
    "Grand Canyon": "Grand Canyon",
    "Green Bay": "Green Bay",

    # H
    "Hawaii": "Hawai'i",

    # I
    "Illinois-Chicago": "UIC",
    "Indiana St.": "Indiana State",
    "Iowa St.": "Iowa State",

    # J
    "Jacksonville St.": "Jacksonville State",

    # K
    "Kansas St.": "Kansas State",
    "Kennesaw St.": "Kennesaw State",
    "Kent St.": "Kent State",

    # L
    "Long Beach St.": "Long Beach State",
    "Long Island University": "LIU",
    "Louisiana Lafayette": "Louisiana",
    "UL Lafayette": "Louisiana",
    "Louisiana-Lafayette": "Louisiana",
    "Loyola (IL)": "Loyola Chicago",
    "Loyola Chicago": "Loyola Chicago",
    "Loyola-Chicago": "Loyola Chicago",
    "Loyola (MD)": "Loyola Maryland",

    # M
    "Miami (FL)": "Miami FL",
    "Miami FL": "Miami FL",
    "Miami (OH)": "Miami OH",
    "Miami OH": "Miami OH",
    "Michigan St.": "Michigan State",
    "Middle Tennessee": "Middle Tennessee State",
    "Middle Tennessee St.": "Middle Tennessee State",
    "Milwaukee": "Milwaukee",
    "Mississippi": "Ole Miss",
    "Mississippi St.": "Mississippi State",
    "Mississippi Valley St.": "Mississippi Valley State",
    "Montana St.": "Montana State",
    "Morehead St.": "Morehead State",
    "Murray St.": "Murray State",
    "McNeese St.": "McNeese State",
    "McNeese": "McNeese State",

    # N
    "N.C. State": "NC State",
    "NC State": "NC State",
    "North Carolina St.": "NC State",
    "N.C. A&T": "North Carolina A&T",
    "N.C. Central": "North Carolina Central",
    "New Mexico St.": "New Mexico State",
    "Norfolk St.": "Norfolk State",
    "North Dakota St.": "North Dakota State",
    "Northern Kentucky": "Northern Kentucky",
    "Northwestern St.": "Northwestern State",

    # O
    "Ohio St.": "Ohio State",
    "Oklahoma St.": "Oklahoma State",
    "Ole Miss": "Ole Miss",
    "Oregon St.": "Oregon State",

    # P
    "Penn": "Pennsylvania",
    "Penn St.": "Penn State",
    "Portland St.": "Portland State",
    "Prairie View A&M": "Prairie View",

    # S
    "Sacramento St.": "Sacramento State",
    "Saint Francis (PA)": "Saint Francis",
    "Saint Joseph's": "Saint Joseph's",
    "St. Joseph's": "Saint Joseph's",
    "Saint Louis": "Saint Louis",
    "St. Louis": "Saint Louis",
    "Saint Mary's": "Saint Mary's",
    "St. Mary's": "Saint Mary's",
    "Saint Mary's (CA)": "Saint Mary's",
    "Saint Peter's": "Saint Peter's",
    "St. Peter's": "Saint Peter's",
    "Sam Houston St.": "Sam Houston State",
    "Sam Houston": "Sam Houston State",
    "San Diego St.": "San Diego State",
    "San Jose St.": "San Jose State",
    "SE Missouri St.": "Southeast Missouri State",
    "South Carolina St.": "South Carolina State",
    "South Dakota St.": "South Dakota State",
    "Southern Illinois": "Southern Illinois",
    "SIU Edwardsville": "SIU Edwardsville",
    "Southern Miss": "Southern Mississippi",
    "Southern Mississippi": "Southern Mississippi",
    "Southwest Texas St.": "Texas State",
    "St. Bonaventure": "St. Bonaventure",
    "St. John's": "St. John's",
    "Saint John's": "St. John's",

    # T
    "TAM C. Christi": "Texas A&M Corpus Christi",
    "Texas A&M Corpus Christi": "Texas A&M Corpus Christi",
    "Texas A&M-CC": "Texas A&M Corpus Christi",
    "Texas Southern": "Texas Southern",

    # U
    "UC Davis": "UC Davis",
    "UC Irvine": "UC Irvine",
    "UC Riverside": "UC Riverside",
    "UC San Diego": "UC San Diego",
    "UC Santa Barbara": "UC Santa Barbara",
    "UCSB": "UC Santa Barbara",
    "UNC": "North Carolina",
    "UNC Asheville": "UNC Asheville",
    "UNC Greensboro": "UNC Greensboro",
    "UNC Wilmington": "UNC Wilmington",
    "UNLV": "UNLV",
    "UT Arlington": "UT Arlington",
    "UT Martin": "UT Martin",
    "Utah St.": "Utah State",
    "Utah Valley": "Utah Valley",

    # V
    "Virginia Commonwealth": "VCU",
    "VCU": "VCU",
    "Virginia Tech": "Virginia Tech",

    # W
    "Washington St.": "Washington State",
    "Weber St.": "Weber State",
    "Western Kentucky": "Western Kentucky",
    "Western Michigan": "Western Michigan",
    "Wichita St.": "Wichita State",
    "William & Mary": "William & Mary",
    "Wisconsin-Green Bay": "Green Bay",
    "Wisconsin-Milwaukee": "Milwaukee",
    "Wright St.": "Wright State",

    # Numbers / misc
    "LIU Brooklyn": "LIU",
}


def normalize_team_name(name: str) -> str:
    """
    Normalize a team name to its canonical form.

    Strips whitespace, checks the TEAM_NAME_MAP, and returns
    the canonical version. If no mapping exists, returns the
    original name (stripped).

    Args:
        name: Raw team name from any data source.

    Returns:
        Canonical team name string.
    """
    stripped = name.strip()
    return TEAM_NAME_MAP.get(stripped, stripped)


def seed_to_int(seed) -> int:
    """
    Convert a seed value to an integer.

    Handles various formats:
      - int/float: 1, 16.0
      - str with play-in suffix: "16a", "16b", "11a", "11b"
      - str: "1", "16"
      - NaN/None: returns 0

    Args:
        seed: Seed value in any format.

    Returns:
        Integer seed number (1-16), or 0 if invalid/missing.
    """
    if seed is None:
        return 0

    # Handle numeric types (int, float, numpy types)
    try:
        import math
        if isinstance(seed, (int, float)):
            if math.isnan(seed):
                return 0
            return int(seed)
    except (TypeError, ValueError):
        pass

    # Handle string seeds like "16a", "11b", "3"
    seed_str = str(seed).strip().lower()
    if not seed_str or seed_str == "nan":
        return 0

    # Strip any letter suffix (play-in indicator)
    numeric_part = "".join(c for c in seed_str if c.isdigit())
    if numeric_part:
        return int(numeric_part)

    return 0


def round_label(round_code: str) -> str:
    """
    Convert a round code to a human-readable label.

    Args:
        round_code: Short code like "R64", "S16", "F4", etc.

    Returns:
        Human-readable round name.
    """
    labels = {
        "R68": "First Four",
        "R64": "Round of 64",
        "R32": "Round of 32",
        "S16": "Sweet 16",
        "E8": "Elite 8",
        "F4": "Final Four",
        "Championship": "Championship",
        "2ND": "Runner-Up",
        "Champions": "Champion",
    }
    return labels.get(round_code, round_code)


def win_percentage(wins: int, games: int) -> float:
    """
    Calculate win percentage.

    Args:
        wins: Number of wins.
        games: Total games played.

    Returns:
        Win percentage as a float (0.0 to 1.0), or 0.0 if no games.
    """
    if games == 0:
        return 0.0
    return wins / games


def net_efficiency_margin(adjoe: float, adjde: float) -> float:
    """
    Calculate net adjusted efficiency margin.

    Higher is better: a team that scores efficiently and prevents
    efficient scoring has a large positive margin.

    Args:
        adjoe: Adjusted offensive efficiency.
        adjde: Adjusted defensive efficiency.

    Returns:
        Net efficiency margin (ADJOE - ADJDE).
    """
    return adjoe - adjde
