"""
Step 1: Data Preparation
========================
Extracts documents from JSON at three granularity levels:
- Guide-level: Overall guide information
- Section-level: Main workhorse for retrieval
- Step-level: Fine-grained precision

Run independently: python -m scripts.step1_data_preparation
"""
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass, asdict
import sys

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import JSON_FILE, INDEXES_DIR, PROCESSED_DOCS_FILE


@dataclass
class Document:
    """Represents a document for indexing"""
    doc_id: str
    doc_type: str  # 'guide', 'section', 'step'
    text: str
    
    # Metadata for filtering and retrieval
    guide_id: str
    guide_title: str
    filename: str
    relative_path: str
    system: str
    tags: List[str]
    date: str
    language: List[str]
    summary: str
    business_process: str
    
    # S3 storage key for document URL generation
    s3_key: str = None
    
    # Section-specific (None for guide-level)
    section_title: str = None
    section_description: str = None
    
    # Step-specific (None for guide/section level)
    step_number: int = None
    step_action: str = None
    step_details: str = None
    step_ui: str = None


def generate_doc_id(content: str) -> str:
    """Generate unique document ID from content hash"""
    return hashlib.md5(content.encode()).hexdigest()[:12]


def create_guide_document(guide: Dict[str, Any]) -> Document:
    """Create guide-level document"""
    # Combine text fields for embedding
    text_parts = [
        f"Guide: {guide['title']}",
        f"Système: {guide['system']}",
        f"Processus métier: {guide['business_process']}",
        f"Résumé: {guide['summary']}",
        f"Tags: {', '.join(guide['tags'])}",
    ]
    
    if guide.get('prerequisites'):
        text_parts.append(f"Prérequis: {' '.join(guide['prerequisites'])}")
    
    text = "\n".join(text_parts)
    
    return Document(
        doc_id=f"guide_{guide['id']}",
        doc_type="guide",
        text=text,
        guide_id=guide['id'],
        guide_title=guide['title'],
        filename=guide['filename'],
        relative_path=guide['relative_path'],
        system=guide['system'],
        tags=guide['tags'],
        date=guide.get('date', ''),
        language=guide.get('language', ['fr']),
        summary=guide['summary'],
        business_process=guide['business_process'],
        s3_key=guide.get('s3_key'),
    )


def create_section_document(guide: Dict[str, Any], section: Dict[str, Any], section_idx: int) -> Document:
    """Create section-level document (main workhorse)"""
    # Build comprehensive text from section
    text_parts = [
        f"Guide: {guide['title']}",
        f"Section: {section['section_title']}",
        f"Description: {section.get('description', '')}",
    ]
    
    # Add all steps
    steps_text = []
    for step in section.get('steps', []):
        step_text = f"Étape {step['step_number']}: {step['action']}"
        if step.get('details'):
            step_text += f" - {step['details']}"
        if step.get('ui'):
            step_text += f" (UI: {step['ui']})"
        steps_text.append(step_text)
    
    if steps_text:
        text_parts.append("Étapes:\n" + "\n".join(steps_text))
    
    # Add tags for better matching
    text_parts.append(f"Tags: {', '.join(guide['tags'])}")
    
    text = "\n".join(text_parts)
    
    return Document(
        doc_id=f"section_{guide['id']}_{section_idx}",
        doc_type="section",
        text=text,
        guide_id=guide['id'],
        guide_title=guide['title'],
        filename=guide['filename'],
        relative_path=guide['relative_path'],
        system=guide['system'],
        tags=guide['tags'],
        date=guide.get('date', ''),
        language=guide.get('language', ['fr']),
        summary=guide['summary'],
        business_process=guide['business_process'],
        s3_key=guide.get('s3_key'),
        section_title=section['section_title'],
        section_description=section.get('description', ''),
    )


def create_step_document(
    guide: Dict[str, Any], 
    section: Dict[str, Any], 
    step: Dict[str, Any],
    section_idx: int
) -> Document:
    """Create step-level document (fine-grained)"""
    text_parts = [
        f"Guide: {guide['title']}",
        f"Section: {section['section_title']}",
        f"Étape {step['step_number']}: {step['action']}",
    ]
    
    if step.get('details'):
        text_parts.append(f"Détails: {step['details']}")
    
    if step.get('ui'):
        text_parts.append(f"Interface: {step['ui']}")
    
    text_parts.append(f"Tags: {', '.join(guide['tags'])}")
    
    text = "\n".join(text_parts)
    
    return Document(
        doc_id=f"step_{guide['id']}_{section_idx}_{step['step_number']}",
        doc_type="step",
        text=text,
        guide_id=guide['id'],
        guide_title=guide['title'],
        filename=guide['filename'],
        relative_path=guide['relative_path'],
        system=guide['system'],
        tags=guide['tags'],
        date=guide.get('date', ''),
        language=guide.get('language', ['fr']),
        summary=guide['summary'],
        business_process=guide['business_process'],
        s3_key=guide.get('s3_key'),
        section_title=section['section_title'],
        section_description=section.get('description', ''),
        step_number=step['step_number'],
        step_action=step['action'],
        step_details=step.get('details', ''),
        step_ui=step.get('ui', ''),
    )


def process_json_file(json_path: Path) -> Dict[str, List[Document]]:
    """
    Process JSON file and extract documents at all granularity levels
    
    Returns:
        Dict with keys 'guides', 'sections', 'steps' containing lists of Documents
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    guides = data.get('guides', [])
    
    result = {
        'guides': [],
        'sections': [],
        'steps': []
    }
    
    for guide in guides:
        # Guide-level document
        guide_doc = create_guide_document(guide)
        result['guides'].append(guide_doc)
        
        # Section-level documents
        for section_idx, section in enumerate(guide.get('sections', [])):
            section_doc = create_section_document(guide, section, section_idx)
            result['sections'].append(section_doc)
            
            # Step-level documents
            for step in section.get('steps', []):
                step_doc = create_step_document(guide, section, step, section_idx)
                result['steps'].append(step_doc)
    
    return result


def save_documents(documents: Dict[str, List[Document]], output_path: Path):
    """Save processed documents to JSON"""
    # Convert dataclasses to dicts
    output = {
        doc_type: [asdict(doc) for doc in docs]
        for doc_type, docs in documents.items()
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"✓ Saved documents to {output_path}")


def load_documents(input_path: Path) -> Dict[str, List[Dict]]:
    """Load processed documents from JSON"""
    with open(input_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def main():
    """Run data preparation step"""
    print("=" * 60)
    print("Step 1: Data Preparation")
    print("=" * 60)
    
    # Check input file exists
    if not JSON_FILE.exists():
        print(f"✗ Input file not found: {JSON_FILE}")
        print("  Please copy your JSON file to the data directory.")
        return None
    
    print(f"\n→ Processing: {JSON_FILE}")
    
    # Process documents
    documents = process_json_file(JSON_FILE)
    
    # Print statistics
    print(f"\n✓ Extracted documents:")
    print(f"  • Guide-level:   {len(documents['guides']):4d} documents")
    print(f"  • Section-level: {len(documents['sections']):4d} documents")
    print(f"  • Step-level:    {len(documents['steps']):4d} documents")
    print(f"  • Total:         {sum(len(d) for d in documents.values()):4d} documents")
    
    # Save to file
    save_documents(documents, PROCESSED_DOCS_FILE)
    
    # Show sample
    print("\n→ Sample section document:")
    if documents['sections']:
        sample = documents['sections'][0]
        print(f"  ID: {sample.doc_id}")
        print(f"  Guide: {sample.guide_title}")
        print(f"  Section: {sample.section_title}")
        print(f"  File: {sample.filename}")
    
    return documents


if __name__ == "__main__":
    main()
