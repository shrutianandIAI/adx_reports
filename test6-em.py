import os
import time
import re
import traceback
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

# PDF and text processing
import PyPDF2
import fitz  # PyMuPDF - alternative to PyPDF2

# Embeddings and vector storage
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# Original selenium imports
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException

# === CONFIGURATION ===
CHROMEDRIVER_PATH = "/home/shrutianand/Downloads/chromedriver-linux64/chromedriver"
BASE_URL = "https://www.adx.ae/main-market"
REPORT_URL_TEMPLATE = "https://www.adx.ae/main-market/company-profile/financial-reports?symbols={ticker}&secCode={ticker}"
PARENT_DIR = os.path.abspath("adx_reports")
EMBEDDINGS_DB_PATH = os.path.join(PARENT_DIR, "embeddings_db")
METADATA_FILE = os.path.join(PARENT_DIR, "documents_metadata.json")

# Embedding model configuration
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # Fast and efficient model
CHUNK_SIZE = 1000  # Characters per chunk
CHUNK_OVERLAP = 200  # Overlap between chunks

# Create directories
os.makedirs(PARENT_DIR, exist_ok=True)
os.makedirs(EMBEDDINGS_DB_PATH, exist_ok=True)

# === EMBEDDING AND VECTOR STORAGE SETUP ===
class EmbeddingManager:
    def __init__(self):
        print("Initializing embedding model...")
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        
        print("Setting up ChromaDB...")
        self.client = chromadb.PersistentClient(path=EMBEDDINGS_DB_PATH)
        self.collection = self.client.get_or_create_collection(
            name="adx_financial_reports",
            metadata={"description": "ADX Financial Reports Embeddings"}
        )
        
        self.metadata = self.load_metadata()
        print("Embedding system ready")
    
    def load_metadata(self) -> Dict:
        """Load existing metadata or create new"""
        if os.path.exists(METADATA_FILE):
            with open(METADATA_FILE, 'r') as f:
                return json.load(f)
        return {"processed_files": {}, "statistics": {"total_documents": 0, "total_chunks": 0}}
    
    def save_metadata(self):
        """Save metadata to file"""
        with open(METADATA_FILE, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF using PyMuPDF (more reliable)"""
        try:
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return text.strip()
        except Exception as e:
            print(f"    PyMuPDF failed, trying PyPDF2: {str(e)}")
            try:
                # Fallback to PyPDF2
                with open(pdf_path, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    text = ""
                    for page in reader.pages:
                        text += page.extract_text()
                return text.strip()
            except Exception as e2:
                print(f"    PyPDF2 also failed: {str(e2)}")
                return ""
    
    def chunk_text(self, text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
        """Split text into overlapping chunks"""
        if not text or len(text) < chunk_size:
            return [text] if text else []
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            
            # Try to break at sentence boundary
            if end < len(text):
                last_period = chunk.rfind('.')
                last_newline = chunk.rfind('\n')
                break_point = max(last_period, last_newline)
                
                if break_point > start + chunk_size // 2:  # Only if break point is reasonable
                    chunk = text[start:start + break_point + 1]
                    end = start + break_point + 1
            
            chunks.append(chunk.strip())
            start = end - overlap
            
            if start >= len(text):
                break
        
        return [chunk for chunk in chunks if chunk.strip()]
    
    def get_file_hash(self, file_path: str) -> str:
        """Get file hash to check if file was modified"""
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    def process_pdf_to_embeddings(self, pdf_path: str, ticker: str, report_type: str, report_date: str):
        """Process a PDF file and store its embeddings"""
        file_path = Path(pdf_path)
        file_hash = self.get_file_hash(pdf_path)
        file_key = str(file_path.relative_to(PARENT_DIR))
        
        # Check if file was already processed and hasn't changed
        if file_key in self.metadata["processed_files"]:
            if self.metadata["processed_files"][file_key]["hash"] == file_hash:
                print(f"   Already processed (unchanged): {file_path.name}")
                return
            else:
                print(f" File changed, reprocessing: {file_path.name}")
                # Remove old embeddings for this file
                old_ids = self.metadata["processed_files"][file_key].get("chunk_ids", [])
                if old_ids:
                    try:
                        self.collection.delete(ids=old_ids)
                    except:
                        pass
        
        print(f"    Processing: {file_path.name}")
        
        # Extract text from PDF
        text = self.extract_text_from_pdf(pdf_path)
        if not text:
            print(f"    No text extracted from {file_path.name}")
            return
        
        print(f" Extracted {len(text)} characters")
        
        # Split into chunks
        chunks = self.chunk_text(text)
        if not chunks:
            print(f"    No chunks created from {file_path.name}")
            return
        
        print(f"Created {len(chunks)} chunks")
        
        # Generate embeddings
        print(f" Generating embeddings...")
        embeddings = self.model.encode(chunks, show_progress_bar=False)
        
        # Prepare data for ChromaDB
        chunk_ids = []
        documents = []
        metadatas = []
        
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            chunk_id = f"{ticker}_{report_date}_{i}_{file_hash[:8]}"
            chunk_ids.append(chunk_id)
            documents.append(chunk)
            
            metadata = {
                "ticker": ticker,
                "report_type": report_type,
                "report_date": report_date,
                "file_path": file_key,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "file_hash": file_hash,
                "text_length": len(chunk)
            }
            metadatas.append(metadata)
        
        # Store in ChromaDB
        try:
            self.collection.add(
                ids=chunk_ids,
                embeddings=embeddings.tolist(),
                documents=documents,
                metadatas=metadatas
            )
            
            # Update metadata
            self.metadata["processed_files"][file_key] = {
                "hash": file_hash,
                "ticker": ticker,
                "report_type": report_type,
                "report_date": report_date,
                "chunks_count": len(chunks),
                "chunk_ids": chunk_ids,
                "processed_at": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            self.metadata["statistics"]["total_chunks"] += len(chunks)
            print(f"   Stored {len(chunks)} embeddings")
            
        except Exception as e:
            print(f"    Error storing embeddings: {str(e)}")
    
    def search_similar_documents(self, query: str, n_results: int = 5, ticker_filter: str = None):
        """Search for similar documents using embeddings"""
        query_embedding = self.model.encode([query])
        
        where_clause = {}
        if ticker_filter:
            where_clause["ticker"] = ticker_filter
        
        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=n_results,
            where=where_clause if where_clause else None
        )
        
        return results
    
    def get_collection_stats(self):
        """Get statistics about the collection"""
        count = self.collection.count()
        return {
            "total_embeddings": count,
            "total_processed_files": len(self.metadata["processed_files"]),
            "total_chunks": self.metadata["statistics"]["total_chunks"]
        }

# === ORIGINAL SELENIUM FUNCTIONS (Modified) ===
def setup_chrome_driver():
    chrome_prefs = {
        "download.default_directory": PARENT_DIR,
        "download.prompt_for_download": False,
        "plugins.always_open_pdf_externally": True,
        "download.directory_upgrade": True,
        "safebrowsing.enabled": True
    }
    
    options = Options()
    options.add_experimental_option("prefs", chrome_prefs)
    # options.add_argument("--headless")  # Uncomment for headless mode
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    
    driver = webdriver.Chrome(service=Service(CHROMEDRIVER_PATH), options=options)
    return driver

def clean_filename(name):
    """Clean filename by removing invalid characters"""
    return re.sub(r'[\\/*?:"<>|]', "_", name)

def wait_for_download(download_folder, timeout=30):
    """Wait for download to complete by checking for .crdownload files"""
    start_time = time.time()
    while time.time() - start_time < timeout:
        crdownload_files = [f for f in os.listdir(download_folder) if f.endswith('.crdownload')]
        if not crdownload_files:
            time.sleep(1)
            return True
        time.sleep(1)
    return False

def get_all_companies(driver, wait):
    """Scrape all company tickers from the main market page"""
    print("Navigating to ADX Main Market...")
    driver.get(BASE_URL)
    
    try:
        wait.until(EC.presence_of_element_located((By.XPATH, "//div[@data-column-id='Symbol']//a")))
        print("Page loaded successfully")
        
        scroll_to_load_all_companies(driver)
        
        symbols = []
        elements = driver.find_elements(By.XPATH, "//div[@data-column-id='Symbol']//a")
        
        for el in elements:
            ticker = el.text.strip()
            if ticker:
                symbols.append(ticker)
        
        print(f"Found {len(symbols)} companies: {', '.join(symbols[:5])}{'...' if len(symbols) > 5 else ''}")
        return symbols
        
    except TimeoutException:
        print("Timeout waiting for page to load")
        return []
    except Exception as e:
        print(f" Error getting companies: {str(e)}")
        return []

def scroll_to_load_all_companies(driver):
    """Scroll to bottom to load all companies"""
    print("Scrolling to load all companies...")
    last_height = driver.execute_script("return document.body.scrollHeight")
    scroll_attempts = 0
    max_attempts = 10
    
    while scroll_attempts < max_attempts:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)
        new_height = driver.execute_script("return document.body.scrollHeight")
        
        if new_height == last_height:
            break
            
        last_height = new_height
        scroll_attempts += 1
    
    print(f"Scrolling completed after {scroll_attempts} attempts")

def download_file_selenium(driver, download_url, folder_path):
    """Download file using Selenium"""
    try:
        driver.execute_cdp_cmd("Page.setDownloadBehavior", {
            "behavior": "allow",
            "downloadPath": folder_path
        })

        original_tab = driver.current_window_handle
        original_tabs = driver.window_handles

        driver.execute_script("window.open(arguments[0], '_blank');", download_url)
        time.sleep(2)

        new_tabs = driver.window_handles
        new_tab = None
        for tab in new_tabs:
            if tab not in original_tabs:
                new_tab = tab
                break

        if new_tab:
            driver.switch_to.window(new_tab)
            wait_for_download(folder_path, timeout=15)
            driver.close()

        driver.switch_to.window(original_tab)
        return True

    except Exception as e:
        print(f" Error during file download: {str(e)}")
        try:
            driver.switch_to.window(original_tab)
        except:
            pass
        return False

def download_reports_for_company(driver, wait, ticker, embedding_manager):
    """Download all financial reports for a specific company and process embeddings"""
    report_url = REPORT_URL_TEMPLATE.format(ticker=ticker)
    print(f"\n[{ticker}] Processing company reports...")
    
    try:
        driver.get(report_url)
        time.sleep(3)

        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "div.table-responsive table tbody tr")))
        
        rows = driver.find_elements(By.CSS_SELECTOR, "div.table-responsive table tbody tr")
        if not rows:
            print(f" No reports found for {ticker}")
            return 0

        company_folder = os.path.join(PARENT_DIR, clean_filename(ticker))
        os.makedirs(company_folder, exist_ok=True)
        print(f"Using folder: {company_folder}")

        downloaded_count = 0
        processed_embeddings = 0
        
        for idx, row in enumerate(rows):
            try:
                cols = row.find_elements(By.TAG_NAME, "td")
                if len(cols) < 4:
                    continue

                a_tags = cols[3].find_elements(By.TAG_NAME, "a")
                if not a_tags:
                    print(f"Row {idx+1}: No download link found. Skipping.")
                    continue

                link = a_tags[0].get_attribute("href")
                report_type = cols[1].text.strip()
                report_date = cols[2].text.strip().replace(" ", "_").replace("/", "-")
                
                filename = f"{ticker}_{report_date}_{clean_filename(report_type)}.pdf"
                file_path = os.path.join(company_folder, filename)
                
                # Download if not exists
                if not os.path.exists(file_path):
                    print(f"[{idx+1}] Downloading: {filename}")
                    
                    if download_file_selenium(driver, link, company_folder):
                        downloaded_count += 1
                        print(f"[{idx+1}] Downloaded successfully")
                    else:
                        print(f" [{idx+1}] Download failed")
                        continue
                    
                    time.sleep(1)
                else:
                    print(f" [{idx+1}] Already exists: {filename}")
                
                # Process embeddings for the file
                if os.path.exists(file_path):
                    print(f"[{idx+1}] Processing embeddings...")
                    embedding_manager.process_pdf_to_embeddings(
                        file_path, ticker, report_type, report_date
                    )
                    processed_embeddings += 1

            except Exception as e:
                print(f" Failed to process report row {idx+1}: {str(e)}")
                continue
        
        print(f"[{ticker}] Downloaded: {downloaded_count}, Processed embeddings: {processed_embeddings}")
        return downloaded_count
        
    except TimeoutException:
        print(f"Timeout loading reports page for {ticker}")
        return 0
    except Exception as e:
        print(f" Could not load reports for {ticker}: {str(e)}")
        return 0

# === SEARCH FUNCTIONALITY ===
def search_documents_interactive(embedding_manager):
    """Interactive search functionality"""
    print("\n" + "="*60)
    print("DOCUMENT SEARCH MODE")
    print("="*60)
    
    stats = embedding_manager.get_collection_stats()
    print(f"Database contains {stats['total_embeddings']} embeddings from {stats['total_processed_files']} files")
    
    while True:
        print("\nOptions:")
        print("1. Search all documents")
        print("2. Search specific ticker")
        print("3. Show statistics")
        print("4. Exit search")
        
        choice = input("\nEnter choice (1-4): ").strip()
        
        if choice == "1":
            query = input("Enter search query: ").strip()
            if query:
                results = embedding_manager.search_similar_documents(query, n_results=5)
                print_search_results(results)
        
        elif choice == "2":
            ticker = input("Enter ticker symbol: ").strip().upper()
            query = input("Enter search query: ").strip()
            if query and ticker:
                results = embedding_manager.search_similar_documents(query, n_results=5, ticker_filter=ticker)
                print_search_results(results)
        
        elif choice == "3":
            stats = embedding_manager.get_collection_stats()
            print(f"\nStatistics:")
            print(f"   Total embeddings: {stats['total_embeddings']}")
            print(f"   Total files processed: {stats['total_processed_files']}")
            print(f"   Total chunks: {stats['total_chunks']}")
            
        elif choice == "4":
            break
        
        else:
            print("Invalid choice")

def print_search_results(results):
    """Print formatted search results"""
    if not results['documents'] or not results['documents'][0]:
        print(" No results found")
        return
    
    print(f"\nFound {len(results['documents'][0])} results:")
    print("-" * 80)
    
    for i, (doc, metadata, distance) in enumerate(zip(
        results['documents'][0], 
        results['metadatas'][0], 
        results['distances'][0]
    )):
        print(f"\n Result {i+1} (Similarity: {1-distance:.3f})")
        print(f"   Company: {metadata['ticker']}")
        print(f"   Report: {metadata['report_type']}")
        print(f"   Date: {metadata['report_date']}")
        print(f"   File: {metadata['file_path']}")
        print(f"   Content: {doc[:200]}{'...' if len(doc) > 200 else ''}")
        print("-" * 80)

# === MAIN EXECUTION ===
def main():
    driver = None
    try:
        print("Starting ADX Financial Reports Scraper with Embeddings")
        print(f"Reports will be saved to: {PARENT_DIR}")
        print(f"Embeddings will be stored in: {EMBEDDINGS_DB_PATH}")
        
        # Initialize embedding manager
        embedding_manager = EmbeddingManager()
        
        # Ask user what they want to do
        print("\nWhat would you like to do?")
        print("1. Scrape new reports and create embeddings")
        print("2. Process existing PDFs to embeddings")
        print("3. Search existing embeddings")
        print("4. Full scrape + embeddings")
        
        choice = input("Enter choice (1-4): ").strip()
        
        if choice == "3":
            search_documents_interactive(embedding_manager)
            return
        
        elif choice == "2":
            # Process existing PDFs
            print("Processing existing PDFs...")
            total_processed = 0
            
            for company_folder in Path(PARENT_DIR).iterdir():
                if company_folder.is_dir() and company_folder.name != "embeddings_db":
                    ticker = company_folder.name
                    print(f"\nProcessing {ticker}...")
                    
                    for pdf_file in company_folder.glob("*.pdf"):
                        # Extract metadata from filename
                        parts = pdf_file.stem.split("_")
                        if len(parts) >= 3:
                            report_date = parts[1]
                            report_type = "_".join(parts[2:])
                        else:
                            report_date = "unknown"
                            report_type = "unknown"
                        
                        embedding_manager.process_pdf_to_embeddings(
                            str(pdf_file), ticker, report_type, report_date
                        )
                        total_processed += 1
            
            embedding_manager.save_metadata()
            print(f"Processed {total_processed} PDF files")
            
            # Offer search
            if input("\nWould you like to search the documents? (y/n): ").lower() == 'y':
                search_documents_interactive(embedding_manager)
            return
        
        # For choices 1 and 4, we need selenium
        driver = setup_chrome_driver()
        wait = WebDriverWait(driver, 15)
        
        # Get all company tickers
        tickers = get_all_companies(driver, wait)
        
        if not tickers:
            print(" No companies found. Exiting.")
            return
        
        total_downloads = 0
        processed_companies = 0
        
        # Process each company
        for i, ticker in enumerate(tickers, 1):
            print(f"\n{'='*50}")
            print(f"Processing [{i}/{len(tickers)}]: {ticker}")
            print(f"{'='*50}")
            
            downloads = download_reports_for_company(driver, wait, ticker, embedding_manager)
            total_downloads += downloads
            processed_companies += 1
            
            # Save metadata periodically
            if i % 5 == 0:
                embedding_manager.save_metadata()
            
            time.sleep(2)
        
        # Final save
        embedding_manager.save_metadata()
        
        # Show final statistics
        stats = embedding_manager.get_collection_stats()
        print(f"\n{'='*60}")
        print(f"SCRAPING COMPLETED!")
        print(f"Processed: {processed_companies} companies")
        print(f"Total downloads: {total_downloads}")
        print(f"Total embeddings: {stats['total_embeddings']}")
        print(f"Reports saved to: {PARENT_DIR}")
        print(f"Embeddings saved to: {EMBEDDINGS_DB_PATH}")
        print(f"{'='*60}")
        
        # Offer search
        if input("\nWould you like to search the documents? (y/n): ").lower() == 'y':
            search_documents_interactive(embedding_manager)

    except KeyboardInterrupt:
        print("\nScraping stopped by user (Ctrl+C)")

    except Exception as e:
        print(f" Unhandled error occurred: {str(e)}")
        traceback.print_exc()

    finally:
        if driver:
            try:
                driver.quit()
                print("Browser closed successfully")
            except:
                print("Error closing browser")

if __name__ == "__main__":
    main()