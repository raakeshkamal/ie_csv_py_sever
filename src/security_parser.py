"""
Security parser module for InvestEngine CSV Server.
Handles Security/ISIN string parsing.
"""

import re
from typing import Optional, Tuple


def extract_security_and_isin(text: str) -> Tuple[str, Optional[str]]:
    """
    Extract security name and ISIN from text like "Vanguard ... ETF / ISIN GB00B3XXRP09"
    Returns (security_name, isin) where isin may be None
    """
    match = re.search(r"(.*?)\s*/\s*ISIN\s+([A-Z]{2}[A-Z0-9]{9}[0-9])", str(text))
    if match:
        return match.group(1).strip(), match.group(2)
    return str(text).strip(), None
