# enums.py
from enum import Enum

class CategoryEnum(str, Enum):
    OFFERS = "offers"
    CONVENTIONS = "conventions"
    GUIDE = "guide"
    DEPOT = "depot"
    UNKNOWN = "unknown"

# A simple dictionary to map the ID from JSON to the Enum
# --- FIX: Change keys to match the uppercase strings generated in main.py ---
CATEGORY_MAP = {
    "OFFERS": CategoryEnum.OFFERS,       # Now looks for "OFFERS"
    "CONVENTIONS": CategoryEnum.CONVENTIONS, # Now looks for "CONVENTIONS"
    "GUIDE": CategoryEnum.GUIDE,         # Now looks for "GUIDE"
    "DEPOT": CategoryEnum.DEPOT          # Now looks for "DEPOT"
}