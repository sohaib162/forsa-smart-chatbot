
import json
import os
import sys

# -------------------------------
# Configuration
# -------------------------------
# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# File paths
S3_KEYS_FILE = os.path.join(SCRIPT_DIR, 'documents_s3_keys.json')
PRODUCTS_FILE = os.path.join(PROJECT_ROOT, 'data', 'products.json')  # Your product data
OUTPUT_FILE = os.path.join(PROJECT_ROOT, 'data', 'products_with_s3_keys.json')

# -------------------------------
# 1. Load S3 keys mapping from JSON
# -------------------------------
def load_s3_keys_mapping():
    """Load the filename -> s3_key mapping from documents_s3_keys.json"""
    if not os.path.exists(S3_KEYS_FILE):
        print(f"❌ Error: S3 keys file not found: {S3_KEYS_FILE}")
        sys.exit(1)
    
    print(f"📂 Loading S3 keys from: {S3_KEYS_FILE}")
    with open(S3_KEYS_FILE, 'r', encoding='utf-8') as f:
        mapping = json.load(f)
    print(f"   ✅ Found {len(mapping)} S3 key mappings")
    return mapping

# -------------------------------
# 2. Function: get s3_key from mapping
# -------------------------------
def get_s3_key(document_title, s3_keys_mapping):
    """
    Search for a matching s3_key in the mapping.
    Tries multiple strategies:
    1. Exact match on document title
    2. Case-insensitive match
    3. Match on filename (basename only)
    4. Partial match (contains)
    """
    if not document_title:
        return None
    
    # Strategy 1: Exact match
    if document_title in s3_keys_mapping:
        return s3_keys_mapping[document_title]
    
    # Strategy 2: Case-insensitive match
    doc_lower = document_title.lower().strip()
    for key, value in s3_keys_mapping.items():
        if key.lower().strip() == doc_lower:
            return value
    
    # Strategy 3: Match basename (in case paths differ)
    doc_base = os.path.basename(document_title).lower()
    for key, value in s3_keys_mapping.items():
        key_base = os.path.basename(key).lower()
        if key_base == doc_base:
            return value
    
    # Strategy 4: Partial match (key contains document title or vice versa)
    for key, value in s3_keys_mapping.items():
        key_clean = key.lower().strip()
        # Check if one contains the other
        if doc_lower in key_clean or key_clean in doc_lower:
            # Additional check: similarity should be high
            if len(doc_lower) > 5 or len(key_clean) > 5:
                return value
    
    return None

# -------------------------------
# 3. Load product data
# -------------------------------
def load_products():
    """Load product data from JSON"""
    if not os.path.exists(PRODUCTS_FILE):
        print(f"❌ Error: Products file not found: {PRODUCTS_FILE}")
        sys.exit(1)
    
    print(f"\n📦 Loading products from: {PRODUCTS_FILE}")
    with open(PRODUCTS_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Handle different JSON structures
    if isinstance(data, list):
        products = data
    elif isinstance(data, dict) and 'products' in data:
        products = data['products']
    elif isinstance(data, dict) and 'data' in data:
        products = data['data']
    else:
        products = [data]
    
    print(f"   ✅ Loaded {len(products)} products")
    return data, products

# -------------------------------
# 4. Add S3 keys to products
# -------------------------------
def add_s3_keys_to_products(products, s3_keys_mapping):
    """
    Add s3_key field to each product by matching document_title
    """
    matched = 0
    not_matched = []
    
    print(f"\n🔍 Matching products to S3 keys...")
    
    for i, product in enumerate(products):
        # Try to find document title in metadata
        metadata = product.get('metadata', {})
        document_title = metadata.get('document_title', '')
        
        # If no document_title in metadata, try other fields
        if not document_title:
            document_title = product.get('document_title', '')
        if not document_title:
            document_title = product.get('title', '')
        if not document_title:
            # Try to construct from product_info
            product_info = product.get('product_info', {})
            document_title = product_info.get('name', '')
        
        # Try to find matching S3 key
        s3_key = get_s3_key(document_title, s3_keys_mapping)
        
        if s3_key:
            # Add s3_key to metadata
            if 'metadata' not in product:
                product['metadata'] = {}
            product['metadata']['s3_key'] = s3_key
            matched += 1
            
            if (i + 1) % 10 == 0:
                print(f"   Processed {i + 1}/{len(products)} products...")
        else:
            product['metadata']['s3_key'] = None
            not_matched.append(document_title if document_title else f"Product #{i+1}")
    
    print(f"\n✅ Matching complete!")
    print(f"   - Matched: {matched}/{len(products)}")
    print(f"   - Not matched: {len(not_matched)}/{len(products)}")
    
    return matched, not_matched

# -------------------------------
# 5. Save updated products
# -------------------------------
def save_products(original_data, products):
    """Save updated products to JSON"""
    # Reconstruct the original structure
    if isinstance(original_data, list):
        output_data = products
    elif isinstance(original_data, dict) and 'products' in original_data:
        output_data = original_data.copy()
        output_data['products'] = products
    elif isinstance(original_data, dict) and 'data' in original_data:
        output_data = original_data.copy()
        output_data['data'] = products
    else:
        output_data = products
    
    print(f"\n💾 Saving updated products to: {OUTPUT_FILE}")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"   ✅ Saved successfully!")
    
    # Also update the original file (optional - comment out if you don't want this)
    print(f"\n💾 Updating original file: {PRODUCTS_FILE}")
    with open(PRODUCTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f"   ✅ Original file updated!")

# -------------------------------
# Main
# -------------------------------
def main():
    print("="*80)
    print("🚀 ADD S3 KEYS TO PRODUCT DATASET")
    print("="*80)
    
    # Step 1: Load S3 keys mapping
    s3_keys_mapping = load_s3_keys_mapping()
    
    # Step 2: Load products
    original_data, products = load_products()
    
    # Step 3: Add S3 keys
    matched, not_matched = add_s3_keys_to_products(products, s3_keys_mapping)
    
    # Step 4: Save updated products
    save_products(original_data, products)
    
    # Step 5: Show summary
    print("\n" + "="*80)
    print("📊 SUMMARY")
    print("="*80)
    print(f"Total products: {len(products)}")
    print(f"Matched with S3 keys: {matched} ({matched/len(products)*100:.1f}%)")
    print(f"Not matched: {len(not_matched)} ({len(not_matched)/len(products)*100:.1f}%)")
    
    if not_matched and len(not_matched) <= 20:
        print("\n⚠️ Products not matched:")
        for title in not_matched[:20]:
            print(f"   - {title}")
        if len(not_matched) > 20:
            print(f"   ... and {len(not_matched) - 20} more")
    
    print("\n✅ Done!")
    print(f"📁 Output saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
