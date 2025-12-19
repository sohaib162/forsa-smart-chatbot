"""
Step 2: Sparse Indexing (BM25)
==============================
Creates a BM25 index using SQLite FTS5 for fast keyword search.

Benefits:
- Ultra-fast keyword matching
- Excellent for acronyms (FADET, TVA, IDOOM, etc.)
- No external dependencies
- Persistent storage

Run independently: python -m scripts.step2_sparse_index
"""
import sqlite3
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import re
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import SQLITE_DB, PROCESSED_DOCS_FILE, INDEXES_DIR, BM25_TOP_K


class BM25Index:
    """SQLite FTS5-based BM25 index for fast keyword search"""
    
    def __init__(self, db_path: Path = SQLITE_DB):
        self.db_path = db_path
        self.conn = None
    
    def connect(self):
        """Establish database connection"""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        return self
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            self.conn = None
    
    def __enter__(self):
        return self.connect()
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def create_index(self):
        """Create FTS5 virtual tables for full-text search"""
        cursor = self.conn.cursor()
        
        # Drop existing tables
        cursor.execute("DROP TABLE IF EXISTS documents")
        cursor.execute("DROP TABLE IF EXISTS documents_fts")
        cursor.execute("DROP TABLE IF EXISTS document_metadata")
        
        # Main documents table
        cursor.execute("""
            CREATE TABLE documents (
                id INTEGER PRIMARY KEY,
                doc_id TEXT UNIQUE NOT NULL,
                doc_type TEXT NOT NULL,
                text TEXT NOT NULL
            )
        """)
        
        # Metadata table (separate for efficient filtering)
        cursor.execute("""
            CREATE TABLE document_metadata (
                doc_id TEXT PRIMARY KEY,
                guide_id TEXT,
                guide_title TEXT,
                section_title TEXT,
                filename TEXT,
                relative_path TEXT,
                system TEXT,
                tags TEXT,
                date TEXT,
                language TEXT,
                summary TEXT,
                business_process TEXT,
                section_description TEXT,
                step_number INTEGER,
                step_action TEXT,
                step_details TEXT,
                step_ui TEXT,
                s3_key TEXT
            )
        """)
        
        # FTS5 virtual table for full-text search
        # Using porter tokenizer for stemming + trigram for partial matching
        cursor.execute("""
            CREATE VIRTUAL TABLE documents_fts USING fts5(
                doc_id,
                doc_type,
                text,
                guide_title,
                section_title,
                tags,
                content='documents',
                content_rowid='id',
                tokenize='unicode61 remove_diacritics 2'
            )
        """)
        
        # Triggers to keep FTS in sync
        cursor.execute("""
            CREATE TRIGGER documents_ai AFTER INSERT ON documents BEGIN
                INSERT INTO documents_fts(rowid, doc_id, doc_type, text, guide_title, section_title, tags)
                SELECT NEW.id, NEW.doc_id, NEW.doc_type, NEW.text, 
                       m.guide_title, m.section_title, m.tags
                FROM document_metadata m WHERE m.doc_id = NEW.doc_id;
            END
        """)
        
        cursor.execute("""
            CREATE TRIGGER documents_ad AFTER DELETE ON documents BEGIN
                INSERT INTO documents_fts(documents_fts, rowid, doc_id, doc_type, text, guide_title, section_title, tags)
                VALUES('delete', OLD.id, OLD.doc_id, OLD.doc_type, OLD.text, '', '', '');
            END
        """)
        
        # Create indexes for efficient metadata filtering
        cursor.execute("CREATE INDEX idx_doc_type ON documents(doc_type)")
        cursor.execute("CREATE INDEX idx_guide_id ON document_metadata(guide_id)")
        cursor.execute("CREATE INDEX idx_system ON document_metadata(system)")
        
        self.conn.commit()
        print("✓ Created BM25 index schema")
    
    def index_documents(self, documents: List[Dict[str, Any]]):
        """Index a list of documents"""
        cursor = self.conn.cursor()
        
        for doc in documents:
            # Insert metadata first
            cursor.execute("""
                INSERT OR REPLACE INTO document_metadata (
                    doc_id, guide_id, guide_title, section_title, filename,
                    relative_path, system, tags, date, language, summary,
                    business_process, section_description, step_number,
                    step_action, step_details, step_ui, s3_key
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                doc['doc_id'],
                doc.get('guide_id'),
                doc.get('guide_title'),
                doc.get('section_title'),
                doc.get('filename'),
                doc.get('relative_path'),
                doc.get('system'),
                json.dumps(doc.get('tags', []), ensure_ascii=False),
                doc.get('date'),
                json.dumps(doc.get('language', []), ensure_ascii=False),
                doc.get('summary'),
                doc.get('business_process'),
                doc.get('section_description'),
                doc.get('step_number'),
                doc.get('step_action'),
                doc.get('step_details'),
                doc.get('step_ui'),
                doc.get('s3_key'),
            ))
            
            # Insert document (triggers FTS update)
            cursor.execute("""
                INSERT OR REPLACE INTO documents (doc_id, doc_type, text)
                VALUES (?, ?, ?)
            """, (doc['doc_id'], doc['doc_type'], doc['text']))
        
        self.conn.commit()
    
    def search(
        self, 
        query: str, 
        top_k: int = BM25_TOP_K,
        doc_type: Optional[str] = None,
        tag_filter: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search documents using BM25 scoring
        
        Args:
            query: Search query string
            top_k: Number of results to return
            doc_type: Filter by document type ('guide', 'section', 'step')
            tag_filter: Filter by tags (OR logic)
        
        Returns:
            List of matching documents with scores
        """
        cursor = self.conn.cursor()
        
        # Clean and prepare query for FTS5
        # Remove special characters that break FTS5
        clean_query = re.sub(r'[^\w\s]', ' ', query)
        clean_query = ' '.join(clean_query.split())  # Normalize whitespace
        
        if not clean_query:
            return []
        
        # Split into words and join with OR for better recall
        # FTS5 default is AND which is too restrictive
        words = clean_query.split()
        
        # Remove common French stopwords for better matching
        stopwords = {'le', 'la', 'les', 'un', 'une', 'des', 'de', 'du', 'et', 
                     'en', 'à', 'au', 'aux', 'pour', 'sur', 'dans', 'par', 
                     'avec', 'ce', 'cette', 'ces', 'son', 'sa', 'ses',
                     'comment', 'que', 'qui', 'quoi', 'quel', 'quelle', 'quels', 'quelles',
                     'est', 'sont', 'a', 'ont', 'faire', 'fait', 'être', 'avoir',
                     'si', 'ou', 'ne', 'pas', 'plus', 'moins', 'très', 'bien',
                     'd', 'l', 'qu', 'n', 's', 'c', 'j', 'm', 't'}
        
        # Keep meaningful words (3+ chars and not stopwords)
        meaningful_words = [w for w in words if len(w) >= 3 and w.lower() not in stopwords]
        
        # If we have no meaningful words, use original words
        if not meaningful_words:
            meaningful_words = [w for w in words if len(w) >= 2]
        
        if not meaningful_words:
            return []
        
        # Build FTS5 query with OR logic for better recall
        fts_query = ' OR '.join(meaningful_words)
        
        # Build the query
        # Using MATCH with BM25 scoring
        sql = """
            SELECT 
                d.doc_id,
                d.doc_type,
                d.text,
                m.*,
                bm25(documents_fts) as score
            FROM documents_fts fts
            JOIN documents d ON fts.rowid = d.id
            JOIN document_metadata m ON d.doc_id = m.doc_id
            WHERE documents_fts MATCH ?
        """
        
        params = [fts_query]
        
        # Add filters
        if doc_type:
            sql += " AND d.doc_type = ?"
            params.append(doc_type)
        
        if tag_filter:
            # OR logic for tags
            tag_conditions = " OR ".join(["m.tags LIKE ?" for _ in tag_filter])
            sql += f" AND ({tag_conditions})"
            params.extend([f"%{tag}%" for tag in tag_filter])
        
        sql += " ORDER BY score LIMIT ?"
        params.append(top_k)
        
        try:
            cursor.execute(sql, params)
            rows = cursor.fetchall()
        except sqlite3.OperationalError as e:
            # Handle FTS5 query errors gracefully
            print(f"⚠ BM25 search error: {e}")
            return []
        
        results = []
        for row in rows:
            result = dict(row)
            # Parse JSON fields
            result['tags'] = json.loads(result.get('tags', '[]'))
            result['language'] = json.loads(result.get('language', '[]'))
            # BM25 scores are negative (lower is better), invert for consistency
            result['bm25_score'] = -result['score']
            results.append(result)
        
        return results
    
    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a single document by ID"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT d.*, m.*
            FROM documents d
            JOIN document_metadata m ON d.doc_id = m.doc_id
            WHERE d.doc_id = ?
        """, (doc_id,))
        
        row = cursor.fetchone()
        if row:
            result = dict(row)
            result['tags'] = json.loads(result.get('tags', '[]'))
            result['language'] = json.loads(result.get('language', '[]'))
            return result
        return None
    
    def get_stats(self) -> Dict[str, int]:
        """Get index statistics"""
        cursor = self.conn.cursor()
        
        stats = {}
        cursor.execute("SELECT COUNT(*) FROM documents")
        stats['total_documents'] = cursor.fetchone()[0]
        
        cursor.execute("SELECT doc_type, COUNT(*) FROM documents GROUP BY doc_type")
        for row in cursor.fetchall():
            stats[f'{row[0]}_documents'] = row[1]
        
        return stats


def build_index():
    """Build BM25 index from processed documents"""
    print("=" * 60)
    print("Step 2: Sparse Indexing (BM25)")
    print("=" * 60)
    
    # Load processed documents
    if not PROCESSED_DOCS_FILE.exists():
        print(f"✗ Processed documents not found: {PROCESSED_DOCS_FILE}")
        print("  Please run step1_data_preparation first.")
        return None
    
    with open(PROCESSED_DOCS_FILE, 'r', encoding='utf-8') as f:
        all_docs = json.load(f)
    
    # Create and populate index
    with BM25Index() as index:
        print(f"\n→ Creating BM25 index at: {SQLITE_DB}")
        index.create_index()
        
        # Index all document types
        for doc_type in ['guides', 'sections', 'steps']:
            docs = all_docs.get(doc_type, [])
            if docs:
                print(f"→ Indexing {len(docs)} {doc_type}...")
                index.index_documents(docs)
        
        # Print stats
        stats = index.get_stats()
        print(f"\n✓ BM25 Index Statistics:")
        for key, value in stats.items():
            print(f"  • {key}: {value}")
        
        # Test search
        print("\n→ Testing BM25 search: 'TVA 2%'")
        results = index.search("TVA 2%", top_k=3, doc_type="section")
        for i, r in enumerate(results, 1):
            print(f"  {i}. [{r['bm25_score']:.2f}] {r['guide_title']} - {r.get('section_title', 'N/A')}")
    
    return SQLITE_DB


def main():
    """Run sparse indexing step"""
    return build_index()


if __name__ == "__main__":
    main()
