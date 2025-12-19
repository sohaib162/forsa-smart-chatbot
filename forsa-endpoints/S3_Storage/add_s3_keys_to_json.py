"""
Add S3 keys to JSON file
=========================
Updates a JSON file with s3_key fields based on filename mapping.

Usage:
  python add_s3_keys_to_json.py [input_json_path]

If no input path is provided, the script does nothing (safe default).
"""
import json
import os
import sys
from pathlib import Path

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent.absolute()


def load_s3_keys_mapping():
    """Load the filename -> s3_key mapping from documents_s3_keys_multi.json"""
    mapping_file = SCRIPT_DIR / 'documents_s3_keys_multi.json'
    if not mapping_file.exists():
        print(f"⚠ Warning: {mapping_file} not found. Trying legacy documents_s3_keys.json...")
        legacy_file = SCRIPT_DIR / 'documents_s3_keys.json'
        if not legacy_file.exists():
            print(f"✗ ERROR: Neither documents_s3_keys_multi.json nor documents_s3_keys.json found.")
            print(f"  Run upload_docs_and_index.py first to generate the mapping.")
            return None

        with open(legacy_file, 'r', encoding='utf-8') as f:
            legacy_data = json.load(f)
            # Convert to multi format: filename -> [s3_key]
            return {k: [v] if isinstance(v, str) else v for k, v in legacy_data.items()}

    with open(mapping_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def normalize_quotes(text):
    """Normalize different quote characters to standard apostrophe"""
    # Replace curly quotes with straight quotes
    # U+2019 RIGHT SINGLE QUOTATION MARK -> '
    # U+2018 LEFT SINGLE QUOTATION MARK -> '
    return text.replace("\u2019", "'").replace("\u2018", "'").replace("`", "'")


def get_s3_key(filename, s3_keys_mapping, existing_s3_key=None):
    """
    Search for a matching s3_key in the mapping.

    Strategy:
    1. If existing_s3_key is provided and exists in mapping, use it (most reliable)
    2. Try exact filename match
    3. Try normalized filename match
    4. Try case-insensitive match
    5. Try basename match ONLY if it maps to exactly one s3_key (avoid collisions)

    Args:
        filename: The filename to search for
        s3_keys_mapping: Dict mapping filename -> list of s3_keys
        existing_s3_key: Optional existing s3_key from the record

    Returns:
        s3_key string or None if not found
    """
    # Strategy 1: Use existing s3_key if it's still valid
    if existing_s3_key:
        # Check if this s3_key exists anywhere in the mapping
        for keys_list in s3_keys_mapping.values():
            if existing_s3_key in keys_list:
                return existing_s3_key

    # Strategy 2: Exact match
    if filename in s3_keys_mapping:
        keys = s3_keys_mapping[filename]
        if len(keys) == 1:
            return keys[0]
        else:
            print(f"⚠ Warning: Multiple S3 keys found for '{filename}': {keys}")
            print(f"  Using first one: {keys[0]}")
            return keys[0]

    # Strategy 3: Normalized match
    filename_norm = normalize_quotes(filename)
    for key, value in s3_keys_mapping.items():
        if normalize_quotes(key) == filename_norm:
            if len(value) == 1:
                return value[0]
            else:
                print(f"⚠ Warning: Multiple S3 keys found for '{filename}': {value}")
                return value[0]

    # Strategy 4: Case-insensitive match
    filename_lower = filename_norm.lower()
    for key, value in s3_keys_mapping.items():
        if normalize_quotes(key).lower() == filename_lower:
            if len(value) == 1:
                return value[0]
            else:
                print(f"⚠ Warning: Multiple S3 keys found for '{filename}': {value}")
                return value[0]

    # Strategy 5: Basename match (ONLY if unique)
    filename_base = os.path.basename(filename_norm).lower()
    candidates = []
    for key, value in s3_keys_mapping.items():
        if os.path.basename(normalize_quotes(key)).lower() == filename_base:
            candidates.extend(value)

    if len(candidates) == 1:
        return candidates[0]
    elif len(candidates) > 1:
        print(f"⚠ Warning: Ambiguous match for '{filename}'. Multiple files with same basename found.")
        print(f"  Cannot safely determine s3_key. Skipping.")
        return None

    return None


def main():
    # Check for input JSON path
    if len(sys.argv) < 2:
        print("Usage: python add_s3_keys_to_json.py <input_json_path>")
        print("\nNo input file specified. Exiting.")
        return 0

    input_json_path = Path(sys.argv[1])

    if not input_json_path.exists():
        print(f"✗ ERROR: Input file not found: {input_json_path}")
        return 1

    if not os.access(input_json_path, os.W_OK):
        print(f"✗ ERROR: Input file is not writable: {input_json_path}")
        return 1

    # Load S3 keys mapping
    print(f"\nLoading S3 keys mapping...")
    s3_keys_mapping = load_s3_keys_mapping()
    if not s3_keys_mapping:
        return 1

    print(f"  ✓ Found {len(s3_keys_mapping)} filename mappings")

    # Load input JSON
    print(f"\nLoading input JSON: {input_json_path}")
    try:
        with open(input_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"✗ ERROR: Failed to load JSON: {e}")
        return 1

    matched = 0
    not_matched = []

    # Process guides (adapt structure as needed)
    guides = data.get('guides', [])
    if not guides:
        print("⚠ Warning: No 'guides' array found in JSON. Looking for other structures...")
        # Could add support for other data structures here
        print("  Supported structure: {'guides': [...]}")
        return 1

    print(f"  Found {len(guides)} guides to process")

    # Add s3_key to each guide
    for guide in guides:
        filename = guide.get('filename', '').strip()
        existing_s3_key = guide.get('s3_key')

        if not filename:
            continue

        s3_key = get_s3_key(filename, s3_keys_mapping, existing_s3_key)

        if s3_key:
            guide['s3_key'] = s3_key
            matched += 1
        else:
            guide['s3_key'] = None
            not_matched.append(filename)

    # Save updated file
    print(f"\nSaving updated JSON to {input_json_path}...")
    try:
        with open(input_json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"✗ ERROR: Failed to save JSON: {e}")
        return 1

    print(f"\n{'='*80}")
    print(f"✓ Done! Updated {len(guides)} guides.")
    print(f"  - Matched: {matched}")
    print(f"  - Not found: {len(not_matched)}")

    if not_matched:
        print(f"\n⚠ Files not found in S3 mapping:")
        for f in not_matched[:10]:
            print(f"    - {f}")
        if len(not_matched) > 10:
            print(f"    ... and {len(not_matched) - 10} more")
    print(f"{'='*80}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
