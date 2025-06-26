import os
import time
import re
import traceback
import base64
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from pymongo import MongoClient
from typing import List, Dict, Any
import uuid

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import tempfile

# === CONFIGURATION ===
CHROMEDRIVER_PATH = "/home/shrutianand/Downloads/chromedriver-linux64/chromedriver"
BASE_URL = "https://www.adx.ae/main-market"
REPORT_URL_TEMPLATE = "https://www.adx.ae/main-market/company-profile/financial-reports?symbols={ticker}&secCode={ticker}"
PARENT_DIR = os.path.abspath("adx_reports")

# MongoDB Configuration
MONGODB_URI = "mongodb://localhost:27017/"  # Update with your MongoDB URI
DATABASE_NAME = "adx_financial_reports"
COLLECTION_NAME = "reports"

# Qdrant Configuration
QDRANT_HOST = "localhost"  # Update with your Qdrant host
QDRANT_PORT = 6333  # Default Qdrant port
COLLECTION_NAME_QDRANT = "adx_embeddings"

# Chunking Configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Create parent directory if it doesn't exist
os.makedirs(PARENT_DIR, exist_ok=True)

# === MONGODB SETUP ===
def setup_mongodb():
    """Setup MongoDB connection and return collection"""
    try:
        client = MongoClient(MONGODB_URI)
        db = client[DATABASE_NAME]
        collection = db[COLLECTION_NAME]
        
        # Test connection
        client.admin.command('ping')
        print("✓ MongoDB connection established successfully")
        
        # Create indexes for better performance
        collection.create_index([("company_ticker", 1), ("report_date", 1), ("report_type", 1)])
        
        return collection
    except Exception as e:
        print(f"✗ MongoDB connection failed: {str(e)}")
        return None

# === QDRANT SETUP ===
def setup_qdrant():
    """Setup Qdrant client and vector store"""
    try:
        # Initialize Qdrant client
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        
        # Initialize embeddings model
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        # Check if collection exists, create if not
        collections = client.get_collections().collections
        collection_names = [col.name for col in collections]
        
        if COLLECTION_NAME_QDRANT not in collection_names:
            client.create_collection(
                collection_name=COLLECTION_NAME_QDRANT,
                vectors_config=VectorParams(
                    size=384,  # Size for all-MiniLM-L6-v2 model
                    distance=Distance.COSINE
                )
            )
            print(f"✓ Created Qdrant collection: {COLLECTION_NAME_QDRANT}")
        else:
            print(f"✓ Using existing Qdrant collection: {COLLECTION_NAME_QDRANT}")
        
        # Initialize LangChain Qdrant vector store
        vector_store = Qdrant(
            client=client,
            collection_name=COLLECTION_NAME_QDRANT,
            embeddings=embeddings
        )
        
        print("✓ Qdrant connection established successfully")
        return vector_store, embeddings, client
        
    except Exception as e:
        print(f"✗ Qdrant connection failed: {str(e)}")
        return None, None, None

def pdf_to_base64_pages(file_path: str) -> List[str]:
    """Convert PDF file to base64 encoded pages"""
    try:
        import PyPDF2
        
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            pages_base64 = []
            
            for page_num in range(len(pdf_reader.pages)):
                # Create a new PDF with just this page
                pdf_writer = PyPDF2.PdfWriter()
                pdf_writer.add_page(pdf_reader.pages[page_num])
                
                # Write to bytes
                import io
                page_bytes = io.BytesIO()
                pdf_writer.write(page_bytes)
                page_bytes.seek(0)
                
                # Convert to base64
                page_base64 = base64.b64encode(page_bytes.read()).decode('utf-8')
                pages_base64.append(page_base64)
            
            return pages_base64
    except Exception as e:
        print(f"Error converting PDF to base64: {str(e)}")
        # Fallback: convert entire file as single base64 string
        try:
            with open(file_path, 'rb') as file:
                file_content = file.read()
                return [base64.b64encode(file_content).decode('utf-8')]
        except Exception as fallback_error:
            print(f"Fallback conversion also failed: {str(fallback_error)}")
            return []

def extract_and_chunk_pdf(file_path: str, company_ticker: str, company_name: str, 
                         report_type: str, report_date: str) -> List[Document]:
    """Extract text from PDF and create chunks with metadata"""
    try:
        # Load PDF using LangChain
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        
        # Initialize text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Create chunks
        chunks = []
        chunk_serial = 1
        
        for page_num, page in enumerate(pages):
            page_chunks = text_splitter.split_text(page.page_content)
            
            for chunk_text in page_chunks:
                if chunk_text.strip():  # Only add non-empty chunks
                    chunk_metadata = {
                        "company_ticker": company_ticker,
                        "company_name": company_name,
                        "report_type": report_type,
                        "report_date": report_date,
                        "page_number": page_num + 1,
                        "chunk_serial": chunk_serial,
                        "chunk_id": f"{company_ticker}_{report_date}_{report_type}_chunk_{chunk_serial}",
                        "source_file": os.path.basename(file_path),
                        "chunk_size": len(chunk_text),
                        "created_at": datetime.utcnow().isoformat()
                    }
                    
                    chunk_doc = Document(
                        page_content=chunk_text,
                        metadata=chunk_metadata
                    )
                    chunks.append(chunk_doc)
                    chunk_serial += 1
        
        print(f"   Created {len(chunks)} chunks from {len(pages)} pages")
        return chunks
        
    except Exception as e:
        print(f"   Error extracting and chunking PDF: {str(e)}")
        return []

def store_chunks_in_qdrant(vector_store, chunks: List[Document], mongodb_doc_id: str) -> int:
    """Store document chunks as vector embeddings in Qdrant"""
    try:
        if not chunks:
            return 0
        
        # Add MongoDB document ID to chunk metadata
        for chunk in chunks:
            chunk.metadata["mongodb_doc_id"] = str(mongodb_doc_id)
            chunk.metadata["vector_id"] = str(uuid.uuid4())
        
        # Store chunks in Qdrant
        vector_store.add_documents(chunks)
        
        print(f"   Stored {len(chunks)} chunks in Qdrant")
        return len(chunks)
        
    except Exception as e:
        print(f"   Error storing chunks in Qdrant: {str(e)}")
        return 0

def store_in_mongodb(collection, file_path: str, company_name: str, company_ticker: str, 
                    report_type: str, report_date: str, metadata: Dict[str, Any]) -> tuple:
    """Store PDF file in MongoDB as base64 with metadata"""
    try:
        # Convert PDF to base64 pages
        print(f"   Converting PDF to base64...")
        content_pages = pdf_to_base64_pages(file_path)
        
        if not content_pages:
            print(f"   Failed to convert PDF to base64")
            return False, None
        
        # Prepare document
        document = {
            "file_name_base64": base64.b64encode(os.path.basename(file_path).encode()).decode('utf-8'),
            "company_name": company_name,
            "company_ticker": company_ticker,
            "report_type": report_type,
            "report_date": report_date,
            "content_pages": content_pages,  # List of base64 strings, one per page
            "metadata": {
                **metadata,
                "total_pages": len(content_pages),
                "file_size_bytes": os.path.getsize(file_path),
                "upload_timestamp": datetime.utcnow(),
                "original_filename": os.path.basename(file_path)
            }
        }
        
        # Check if document already exists
        existing = collection.find_one({
            "company_ticker": company_ticker,
            "report_type": report_type,
            "report_date": report_date
        })
        
        if existing:
            print(f"   Document already exists in MongoDB, skipping")
            return True, existing["_id"]
        
        # Insert document
        result = collection.insert_one(document)
        print(f"   Stored in MongoDB with ID: {result.inserted_id}")
        return True, result.inserted_id
        
    except Exception as e:
        print(f"   Error storing in MongoDB: {str(e)}")
        return False, None

# === CHROME SETUP WITH DOWNLOAD PREFS ===
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
    # Uncomment the next line to run headless (faster but no visual feedback)
    # options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    
    driver = webdriver.Chrome(service=Service(CHROMEDRIVER_PATH), options=options)
    return driver

# === UTILS ===
def clean_filename(name):
    """Clean filename by removing invalid characters"""
    return re.sub(r'[\\/*?:"<>|]', "_", name)

def wait_for_download(download_folder, timeout=30):
    """Wait for download to complete by checking for .crdownload files"""
    start_time = time.time()
    while time.time() - start_time < timeout:
        # Check if there are any incomplete downloads (.crdownload files)
        crdownload_files = [f for f in os.listdir(download_folder) if f.endswith('.crdownload')]
        if not crdownload_files:
            time.sleep(1)  # Give a moment for the file to appear
            return True
        time.sleep(1)
    return False

def get_all_companies(driver, wait):
    """Scrape all company tickers from the main market page"""
    print("🔍 Navigating to ADX Main Market...")
    driver.get(BASE_URL)
    
    try:
        # Wait for the page to load
        wait.until(EC.presence_of_element_located((By.XPATH, "//div[@data-column-id='Symbol']//a")))
        print("✓ Page loaded successfully")
        
        # Scroll to load all companies
        scroll_to_load_all_companies(driver)
        
        # Extract all company symbols and names
        companies = []
        symbol_elements = driver.find_elements(By.XPATH, "//div[@data-column-id='Symbol']//a")
        name_elements = driver.find_elements(By.XPATH, "//div[@data-column-id='Name']")
        
        for i, symbol_el in enumerate(symbol_elements):
            ticker = symbol_el.text.strip()
            company_name = name_elements[i].text.strip() if i < len(name_elements) else ticker
            
            if ticker:
                companies.append({
                    'ticker': ticker,
                    'name': company_name
                })
        
        print(f"✓ Found {len(companies)} companies")
        return companies
        
    except TimeoutException:
        print("✗ Timeout waiting for page to load")
        return []
    except Exception as e:
        print(f"✗ Error getting companies: {str(e)}")
        return []

def scroll_to_load_all_companies(driver):
    """Scroll to bottom to load all companies"""
    print("📜 Scrolling to load all companies...")
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
    
    print(f"✓ Scrolling completed after {scroll_attempts} attempts")

def download_file_selenium(driver, download_url, folder_path):
    """Download file using Selenium"""
    try:
        # Set download behavior for this specific folder
        driver.execute_cdp_cmd("Page.setDownloadBehavior", {
            "behavior": "allow",
            "downloadPath": folder_path
        })

        # Save current tab handle
        original_tab = driver.current_window_handle
        original_tabs = driver.window_handles

        # Open download in new tab
        driver.execute_script("window.open(arguments[0], '_blank');", download_url)
        time.sleep(2)

        # Find the new tab
        new_tabs = driver.window_handles
        new_tab = None
        for tab in new_tabs:
            if tab not in original_tabs:
                new_tab = tab
                break

        if new_tab:
            # Switch to new tab
            driver.switch_to.window(new_tab)
            
            # Wait for download to start/complete
            wait_for_download(folder_path, timeout=15)
            
            # Close the download tab
            driver.close()

        # Switch back to original tab
        driver.switch_to.window(original_tab)
        
        return True

    except Exception as e:
        print(f"✗ Error during file download: {str(e)}")
        # Make sure we're back on the original tab
        try:
            driver.switch_to.window(original_tab)
        except:
            pass
        return False

def download_reports_for_company(driver, wait, company_info, collection, vector_store):
    """Download all financial reports for a specific company and store in MongoDB + Qdrant"""
    ticker = company_info['ticker']
    company_name = company_info['name']
    report_url = REPORT_URL_TEMPLATE.format(ticker=ticker)
    
    print(f"\n📊 [{ticker}] Processing company reports...")
    
    try:
        driver.get(report_url)
        time.sleep(3)

        # Wait for the reports table to load
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "div.table-responsive table tbody tr")))
        
        rows = driver.find_elements(By.CSS_SELECTOR, "div.table-responsive table tbody tr")
        if not rows:
            print(f"No reports found for {ticker}")
            return 0, 0

        # Create company-specific folder
        company_folder = os.path.join(PARENT_DIR, clean_filename(ticker))
        os.makedirs(company_folder, exist_ok=True)

        processed_count = 0
        total_chunks = 0
        
        for idx, row in enumerate(rows):
            try:
                cols = row.find_elements(By.TAG_NAME, "td")
                if len(cols) < 4:
                    continue

                # Check if download link is present
                a_tags = cols[3].find_elements(By.TAG_NAME, "a")
                if not a_tags:
                    print(f"   Row {idx+1}: No download link found. Skipping.")
                    continue

                link = a_tags[0].get_attribute("href")
                report_type = cols[1].text.strip()
                report_date = cols[2].text.strip()
                report_date_clean = report_date.replace(" ", "_").replace("/", "-")
                
                # Create descriptive filename
                filename = f"{ticker}_{report_date_clean}_{clean_filename(report_type)}.pdf"
                file_path = os.path.join(company_folder, filename)
                
                print(f"   [{idx+1}] Processing: {filename}")
                
                # Download the file
                if download_file_selenium(driver, link, company_folder):
                    # Wait a moment for file to be fully written
                    time.sleep(2)
                    
                    # Check if file exists and has content
                    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                        # Store in MongoDB
                        metadata = {
                            "download_url": link,
                            "row_index": idx + 1,
                            "scraped_date": datetime.utcnow().isoformat()
                        }
                        
                        success, doc_id = store_in_mongodb(collection, file_path, company_name, ticker, 
                                                         report_type, report_date, metadata)
                        
                        if success:
                            # Extract text and create chunks
                            print(f"   Creating chunks...")
                            chunks = extract_and_chunk_pdf(file_path, ticker, company_name, 
                                                         report_type, report_date)
                            
                            if chunks:
                                # Store chunks in Qdrant
                                print(f"   Storing chunks in Qdrant...")
                                chunks_stored = store_chunks_in_qdrant(vector_store, chunks, doc_id)
                                total_chunks += chunks_stored
                                
                                processed_count += 1
                                print(f"   ✓ Successfully processed and stored ({len(chunks)} chunks)")
                            else:
                                print(f"   ⚠️  MongoDB stored but no chunks created")
                        else:
                            print(f"   ✗ Failed to store in MongoDB")
                    else:
                        print(f"   ✗ Downloaded file not found or empty")
                else:
                    print(f"   ✗ Download failed")
                
                # Brief pause between downloads
                time.sleep(1)

            except Exception as e:
                print(f"✗ Failed to process report row {idx+1}: {str(e)}")
                continue
        
        print(f"[{ticker}] Processed {processed_count} reports, Created {total_chunks} chunks")
        return processed_count, total_chunks
        
    except TimeoutException:
        print(f"✗ Timeout loading reports page for {ticker}")
        return 0, 0
    except Exception as e:
        print(f"✗ Could not load reports for {ticker}: {str(e)}")
        return 0, 0

# === MAIN EXECUTION ===
def main():
    driver = None
    collection = None
    vector_store = None
    
    try:
        # Setup MongoDB
        collection = setup_mongodb()
        if not collection:
            print("Cannot proceed without MongoDB connection")
            return
        
        # Setup Qdrant
        vector_store, embeddings, qdrant_client = setup_qdrant()
        if not vector_store:
            print("Cannot proceed without Qdrant connection")
            return
        
        # Setup Chrome driver
        driver = setup_chrome_driver()
        wait = WebDriverWait(driver, 15)
        
        print("🚀 Starting ADX Financial Reports Scraper with MongoDB + Qdrant Storage")
        print(f"📁 Temporary files will be saved to: {PARENT_DIR}")
        print(f"🗄️  MongoDB storage: {DATABASE_NAME}.{COLLECTION_NAME}")
        print(f"🔍 Qdrant storage: {COLLECTION_NAME_QDRANT}")
        print(f"📝 Chunk settings: {CHUNK_SIZE} chars, {CHUNK_OVERLAP} overlap")
        
        # Get all companies
        companies = get_all_companies(driver, wait)
        
        if not companies:
            print("No companies found. Exiting.")
            return
        
        total_processed = 0
        total_chunks_created = 0
        
        # Process each company
        for i, company_info in enumerate(companies, 1):
            ticker = company_info['ticker']
            print(f"\n{'='*70}")
            print(f"Processing [{i}/{len(companies)}]: {ticker}")
            print(f"{'='*70}")
            
            processed, chunks_created = download_reports_for_company(
                driver, wait, company_info, collection, vector_store
            )
            total_processed += processed
            total_chunks_created += chunks_created
            
            # Brief pause between companies
            time.sleep(2)
        
        print(f"\n{'='*80}")
        print(f"🎉 SCRAPING COMPLETED!")
        print(f"📊 Processed companies: {len(companies)}")
        print(f"📄 Total reports stored: {total_processed}")
        print(f"🧩 Total chunks created: {total_chunks_created}")
        print(f"🗄️  MongoDB: {DATABASE_NAME}.{COLLECTION_NAME}")
        print(f"🔍 Qdrant: {COLLECTION_NAME_QDRANT}")
        print(f"{'='*80}")

    except KeyboardInterrupt:
        print("\n⏹️  Scraping stopped by user (Ctrl+C)")

    except Exception as e:
        print(f"💥 Unhandled error occurred: {str(e)}")
        traceback.print_exc()

    finally:
        if driver:
            try:
                driver.quit()
                print("🔒 Browser closed successfully")
            except:
                print("⚠️  Error closing browser")

if __name__ == "__main__":
    main()