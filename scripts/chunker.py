import re
import json
import os
import datetime
import logging

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
input_base_dir = os.path.join(project_root, "data/processed/cleaned_documents")
output_chunks_base_dir = os.path.join(project_root, "data/processed/chunks")
output_meta_base_dir = os.path.join(project_root, "data/processed/metadata")

logs_dir = os.path.join(project_root, "logs")
os.makedirs(logs_dir, exist_ok=True)
log_file = os.path.join(logs_dir, "chunking_logs.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8', mode='a'),
        logging.StreamHandler()
    ]
)

page_pattern = re.compile(r"<!-- PAGE:\s*(\d+)\s*-->")
article_pattern = re.compile(r"^(?:(?:M\s*a\s*d\s*d\s*ə)\s+(\d+)|(\d+)\.\s+(?!\d))")
chapter_pattern = re.compile(r"^(?:(?:F\s*ə\s*s\s*i\s*l|B\s*ö\s*l\s*m\s*ə)\s+([IVXLCDM\d]+)|([IVXLCDM\d]+)\s+(?:Fəsil|Bölmə))", re.IGNORECASE)

def process_file(filepath, category):
    filename = os.path.basename(filepath)
    doc_id = filename.replace(".md", "")
    
    chunks = []
    current_page = 1
    max_page = 1

    current_chapter = None
    current_article = None

    current_chunk_lines = [] # list of (text, page)
    
    def process_and_save_current_chunk():
        nonlocal current_chunk_lines, current_article, current_chapter, chunks
        if not current_chunk_lines:
            return
            
        # Check if text is completely empty
        full_text = "\n".join(t[0] for t in current_chunk_lines).strip()
        if not full_text:
            current_chunk_lines = []
            return
            
        # Generate base chunk ID
        parts = []
        if current_chapter: parts.append(f"ch_{current_chapter}")
        if current_article: parts.append(f"art_{current_article}")
        
        h_str = "_".join(parts) if parts else "intro"
        base_chunk_id = f"{doc_id}_{h_str}_{len(chunks)+1}"
        
        # Split by size if needed
        MAX_CHARS = 4000
        OVERLAP_CHARS = 500
        chunk_parts = []
        current_part_lines = []
        current_len = 0
        
        for text, page in current_chunk_lines:
            line_len = len(text) + 1
            if current_len + line_len > MAX_CHARS and current_part_lines:
                chunk_parts.append(current_part_lines)
                
                # Compute overlap from current_part_lines
                overlap_lines = []
                overlap_len = 0
                for prev_text, prev_page in reversed(current_part_lines):
                    prev_line_len = len(prev_text) + 1
                    if overlap_len + prev_line_len > OVERLAP_CHARS:
                        if not overlap_lines:
                            overlap_lines.insert(0, (prev_text, prev_page))
                            overlap_len += prev_line_len
                        break
                    overlap_lines.insert(0, (prev_text, prev_page))
                    overlap_len += prev_line_len
                    
                current_part_lines = overlap_lines + [(text, page)]
                current_len = overlap_len + line_len
            else:
                current_part_lines.append((text, page))
                current_len += line_len
                
        if current_part_lines:
            chunk_parts.append(current_part_lines)
            
        for i, part in enumerate(chunk_parts):
            part_text = "\n".join([p[0] for p in part]).strip()
            if not part_text: continue
            
            page_start = part[0][1]
            page_end = part[-1][1]
            
            if len(chunk_parts) > 1:
                chunk_id = f"{base_chunk_id}_part{i+1}"
            else:
                chunk_id = base_chunk_id
                
            chunk = {
                "chunk_id": chunk_id,
                "document_id": doc_id,
                "title": doc_id,  # Using doc_id as title fallback
                "category": category,
                "chapter": current_chapter,
                "article": current_article,
                "section": None,
                "subsection": None,
                "page_start": page_start,
                "page_end": page_end,
                "source_file": filename,
                "text": part_text
            }
            chunks.append(chunk)
            
        current_chunk_lines = []

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            # Process page marker
            m_page = page_pattern.search(line)
            if m_page:
                current_page = int(m_page.group(1))
                max_page = max(max_page, current_page)
                line = page_pattern.sub("", line).strip()
                if not line:
                    continue
                
            # Ignore horizontal rules often accompanying page tags
            if line.strip() == "---":
                continue

            # Check hierarchy
            m_chap = chapter_pattern.match(line)
            m_art = article_pattern.match(line)
            
            if m_chap:
                process_and_save_current_chunk()
                current_chapter = m_chap.group(1) or m_chap.group(2)
                current_article = None
                current_chunk_lines.append((line.strip(), current_page))
            elif m_art:
                process_and_save_current_chunk()
                current_article = m_art.group(1) or m_art.group(2)
                current_chunk_lines.append((line.strip(), current_page))
            else:
                current_chunk_lines.append((line.rstrip('\n'), current_page))

    process_and_save_current_chunk()

    out_chunks_dir = os.path.join(output_chunks_base_dir, category)
    out_meta_dir = os.path.join(output_meta_base_dir, category)
    os.makedirs(out_chunks_dir, exist_ok=True)
    os.makedirs(out_meta_dir, exist_ok=True)

    # Write jsonl
    jsonl_path = os.path.join(out_chunks_dir, f"{doc_id}.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    # Write metadata
    meta = {
        "document_id": doc_id,
        "title": doc_id,
        "category": category,
        "source_file": filename.replace(".md", ".pdf"),
        "processed_file": filename,
        "total_pages": max_page,
        "total_chunks": len(chunks),
        "language": "az",
        "parser": "pdfplumber",
        "created_at": datetime.datetime.now().isoformat()
    }
    meta_path = os.path.join(out_meta_dir, f"{doc_id}_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    logging.info(f"Processed: {category}/{filename} -> {len(chunks)} chunks, {max_page} pages.")

def main():
    if not os.path.exists(input_base_dir):
        logging.error(f"Input directory does not exist: {input_base_dir}")
        return
        
    for category in os.listdir(input_base_dir):
        category_dir = os.path.join(input_base_dir, category)
        if not os.path.isdir(category_dir):
            continue
            
        for filename in os.listdir(category_dir):
            if not filename.endswith(".md"):
                continue
            filepath = os.path.join(category_dir, filename)
            process_file(filepath, category)

if __name__ == "__main__":
    main()
