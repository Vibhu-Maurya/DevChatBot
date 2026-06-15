import os
import time
import json
import hashlib
import datetime
import trafilatura
import tiktoken

class CleanAndChunkPipeline:
    def open_spider(self, spider):
        # Open separate output files
        self.doc_file = open("documents.jsonl", "a", encoding="utf-8")
        self.chunk_file = open("chunks.jsonl", "a", encoding="utf-8")
        
        # Telemetry
        self.start_time = time.time()
        self.item_count = 0
        self.chunk_count = 0
        self.pages_seen = 0
        self.pages_skipped = 0
        self.duplicates_skipped = 0
        
        # Ensure metrics folder exists
        os.makedirs("metrics", exist_ok=True)
        
        # Load incremental crawling history
        self.history_file = "crawl_history.json"
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, "r", encoding="utf-8") as f:
                    self.history = json.load(f)
            except Exception:
                self.history = {}
        else:
            self.history = {}

        # In-run content hash deduplication
        self.seen_hashes = set()
        self.encoder = tiktoken.get_encoding("cl100k_base")

    def close_spider(self, spider):
        self.doc_file.close()
        self.chunk_file.close()
        
        # Save telemetry stats
        duration = time.time() - self.start_time
        if self.pages_seen > 0:
            stats = {
                "source": getattr(spider, "source_arg", "unknown"),
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
                "crawl_time_seconds": round(duration, 2),
                "pages_seen": self.pages_seen,
                "pages_saved": self.item_count,
                "pages_skipped": self.pages_skipped,
                "duplicates_skipped": self.duplicates_skipped,
                "documents": self.item_count,
                "chunks": self.chunk_count,
                "avg_chunks_per_page": round(self.chunk_count / self.item_count, 2) if self.item_count > 0 else 0
            }
            try:
                with open(os.path.join("metrics", "crawl_stats.jsonl"), "a", encoding="utf-8") as sf:
                    sf.write(json.dumps(stats, ensure_ascii=False) + "\n")
            except Exception as e:
                spider.logger.error(f"Failed to write crawl stats: {e}")

        # Atomic history file writing
        tmp_file = self.history_file + ".tmp"
        try:
            with open(tmp_file, "w", encoding="utf-8") as f:
                json.dump(self.history, f, ensure_ascii=False, indent=2)
            os.replace(tmp_file, self.history_file)
        except Exception as e:
            spider.logger.error(f"Failed to write history file atomically: {e}")

    def process_item(self, item, spider):
        self.pages_seen += 1
        html_content = item.get("html")
        url = item.get("url")
        title = item.get("title")
        domain = item.get("domain")
        source = item.get("source")

        if not html_content:
            return item

        # Extract text using trafilatura
        extracted_text = trafilatura.extract(html_content)
        if not extracted_text:
            extracted_text = ""

        # Content hashing
        content_hash = hashlib.sha256(extracted_text.encode("utf-8")).hexdigest()

        # In-run deduplication
        if content_hash in self.seen_hashes:
            spider.logger.info(f"Duplicate content detected in this run for {url}. Skipping.")
            self.duplicates_skipped += 1
            return item
        self.seen_hashes.add(content_hash)

        # Incremental crawl check
        crawl_date = datetime.date.today().isoformat()
        if url in self.history:
            if self.history[url].get("content_hash") == content_hash:
                spider.logger.info(f"Unchanged content for {url} since {self.history[url].get('crawl_date')}. Skipping.")
                self.pages_skipped += 1
                return item

        # Update history
        self.history[url] = {
            "content_hash": content_hash,
            "crawl_date": crawl_date
        }

        # 1. Write parent to documents.jsonl
        doc_record = {
            "url": url,
            "domain": domain,
            "source": source,
            "title": title,
            "crawl_date": crawl_date,
            "content_hash": content_hash,
            "raw_html_length": len(html_content),
            "clean_text_length": len(extracted_text),
            "markdown": extracted_text
        }
        self.doc_file.write(json.dumps(doc_record, ensure_ascii=False) + "\n")
        self.doc_file.flush()

        # 2. Extract chunks and write child records to chunks.jsonl
        chunks = self.chunk_text_with_headings(extracted_text, title, min_tokens=300, max_tokens=500)
        
        # Telemetry updates
        self.item_count += 1
        self.chunk_count += len(chunks)
        
        for idx, chunk_data in enumerate(chunks):
            chunk_text = chunk_data["content"]
            chunk_hash = hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()
            
            chunk_record = {
                "chunk_id": idx,
                "chunk_hash": chunk_hash,
                "url": url,
                "domain": domain,
                "source": source,
                "title": title,
                "section": chunk_data["section"],
                "content": chunk_text
            }
            self.chunk_file.write(json.dumps(chunk_record, ensure_ascii=False) + "\n")
        self.chunk_file.flush()

        return item

    def chunk_text_with_headings(self, text, doc_title, min_tokens=300, max_tokens=500):
        if not text:
            return []

        paragraphs = text.split("\n")
        
        # Parse paragraphs and associate each with the active section heading
        parsed_paragraphs = []
        current_section = doc_title
        
        # Common starters for headings in docs if markdown symbols are not present
        heading_keywords = ["Introduction", "Tutorial", "Chapter", "How-To", "Section", "Setup", "Install", "Step"]
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # Heading heuristics
            is_heading = False
            if para.startswith("#"):
                is_heading = True
                heading_text = para.lstrip("#").strip()
            elif len(para) < 80 and not para.endswith((".", "?", "!")) and any(para.startswith(w) for w in heading_keywords):
                is_heading = True
                heading_text = para
            
            if is_heading:
                current_section = heading_text
            
            tokens = self.encoder.encode(para)
            parsed_paragraphs.append({
                "text": para,
                "tokens": tokens,
                "num_tokens": len(tokens),
                "section": current_section
            })

        # Group paragraphs into chunks of [min_tokens, max_tokens]
        chunks = []
        current_chunk_paras = []
        current_tokens = 0
        current_chunk_section = doc_title

        for p in parsed_paragraphs:
            num_tokens = p["num_tokens"]
            
            if current_tokens + num_tokens <= max_tokens:
                if not current_chunk_paras:
                    current_chunk_section = p["section"]
                current_chunk_paras.append(p["text"])
                current_tokens += num_tokens
            else:
                if current_tokens >= min_tokens:
                    chunks.append({
                        "section": current_chunk_section,
                        "content": "\n".join(current_chunk_paras)
                    })
                    current_chunk_paras = [p["text"]]
                    current_tokens = num_tokens
                    current_chunk_section = p["section"]
                else:
                    if num_tokens > max_tokens:
                        # Split paragraph into smaller word sets
                        words = p["text"].split(" ")
                        temp_chunk = []
                        temp_tokens = 0
                        for word in words:
                            word_tokens = len(self.encoder.encode(word + " "))
                            if temp_tokens + word_tokens > max_tokens:
                                chunks.append({
                                    "section": p["section"],
                                    "content": " ".join(temp_chunk)
                                })
                                temp_chunk = [word]
                                temp_tokens = word_tokens
                            else:
                                temp_chunk.append(word)
                                temp_tokens += word_tokens
                        if temp_chunk:
                            current_chunk_paras = [" ".join(temp_chunk)]
                            current_tokens = temp_tokens
                            current_chunk_section = p["section"]
                    else:
                        if current_chunk_paras:
                            chunks.append({
                                "section": current_chunk_section,
                                "content": "\n".join(current_chunk_paras)
                            })
                        current_chunk_paras = [p["text"]]
                        current_tokens = num_tokens
                        current_chunk_section = p["section"]

        if current_chunk_paras:
            chunks.append({
                "section": current_chunk_section,
                "content": "\n".join(current_chunk_paras)
            })

        return chunks
