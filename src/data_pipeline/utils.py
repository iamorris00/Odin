import re

def normalize_well_name(raw_name: str) -> str:
    """
    Normalizes well names from various sources (WITSML, DDR, EDM) into a canonical format.
    E.g.:
    "15/9-F-5  W-508420" -> "15/9-F-5"
    "NO 15/9-F-1 C  1bf1cc58-83af-4e13-9696-4fae2f9294ae" -> "15/9-F-1 C"
    "15-9-F-1" -> "15/9-F-1"
    "15_9-F-1" -> "15/9-F-1"
    "15_9_F_1_C" -> "15/9-F-1 C"
    """
    if not isinstance(raw_name, str) or not raw_name.strip():
        return "UNKNOWN"
        
    s = raw_name.strip()
    
    # Remove leading "NO " or "NO-"
    s = re.sub(r'^NO[\s\-]+', '', s, flags=re.IGNORECASE)
    
    # Remove UUIDs or trailing IDs (e.g. "  W-508420" or "  1bf1cc58...")
    # Usually separated by double spaces in WITSML
    if "  " in s:
        s = s.split("  ")[0]
        
    # Standardize the block/quadrant: 15_9 or 15-9 -> 15/9
    s = re.sub(r'^(\d+)[_\-](\d+)', r'\1/\2', s)
    
    # If the format is entirely separated by underscores, try to fix it (e.g., 15_9_F_1_C)
    if '_' in s and '/' in s:
        # e.g., 15/9_F_1_C -> 15/9-F-1 C
        parts = re.split(r'[_\-]+', s)
        if len(parts) >= 3:
            # Reconstruct
            base = f"{parts[0]}-{parts[1]}-{parts[2]}"
            if len(parts) > 3:
                base += f" {' '.join(parts[3:])}"
            s = base
            
    # Also standardize typical "15/9-F-11_A" -> "15/9-F-11 A"
    s = re.sub(r'_([A-Z])$', r' \1', s)
    # And "15/9-F-1_C" -> "15/9-F-1 C"
    s = re.sub(r'_(ST\d+|T\d+)$', r' \1', s)
    
    # Replace remaining underscores with spaces or dashes appropriately?
    # Usually we want 15/9-19 A or 15/9-F-1 C.
    s = s.replace('_', ' ')
    
    # Squeeze multiple spaces
    s = re.sub(r'\s+', ' ', s)
    
    return s.strip()

def safe_filename(name: str) -> str:
    """Converts a canonical name to a safe filename string."""
    return name.replace("/", "_").replace(" ", "_").replace("-", "_")

