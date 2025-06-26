import os
import time
import re
import traceback
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

# Create parent directory if it doesn't exist
os.makedirs(PARENT_DIR, exist_ok=True)

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
    print(" Navigating to ADX Main Market...")
    driver.get(BASE_URL)
    
    try:
        # Wait for the page to load
        wait.until(EC.presence_of_element_located((By.XPATH, "//div[@data-column-id='Symbol']//a")))
        print(" Page loaded successfully")
        
        # Scroll to load all companies
        scroll_to_load_all_companies(driver)
        
        # Extract all company symbols
        symbols = []
        elements = driver.find_elements(By.XPATH, "//div[@data-column-id='Symbol']//a")
        
        for el in elements:
            ticker = el.text.strip()
            if ticker:
                symbols.append(ticker)
        
        print(f"Found {len(symbols)} companies: {', '.join(symbols[:5])}{'...' if len(symbols) > 5 else ''}")
        return symbols
        
    except TimeoutException:
        print(" Timeout waiting for page to load")
        return []
    except Exception as e:
        print(f" Error getting companies: {str(e)}")
        return []

def scroll_to_load_all_companies(driver):
    """Scroll to bottom to load all companies"""
    print(" Scrolling to load all companies...")
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
    
    print(f" Scrolling completed after {scroll_attempts} attempts")

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
        print(f" Error during file download: {str(e)}")
        # Make sure we're back on the original tab
        try:
            driver.switch_to.window(original_tab)
        except:
            pass
        return False

def download_reports_for_company(driver, wait, ticker):
    """Download all financial reports for a specific company"""
    report_url = REPORT_URL_TEMPLATE.format(ticker=ticker)
    print(f"\n [{ticker}] Processing company reports...")
    
    try:
        driver.get(report_url)
        time.sleep(3)

        # Wait for the reports table to load
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "div.table-responsive table tbody tr")))
        
        rows = driver.find_elements(By.CSS_SELECTOR, "div.table-responsive table tbody tr")
        if not rows:
            print(f"No reports found for {ticker}")
            return 0

        # Create company-specific folder
        company_folder = os.path.join(PARENT_DIR, clean_filename(ticker))
        os.makedirs(company_folder, exist_ok=True)
        print(f" Created/Using folder: {company_folder}")

        downloaded_count = 0
        
        for idx, row in enumerate(rows):
            try:
                cols = row.find_elements(By.TAG_NAME, "td")
                if len(cols) < 4:
                    continue

                # Check if download link is present
                a_tags = cols[3].find_elements(By.TAG_NAME, "a")
                if not a_tags:
                    print(f" Row {idx+1}: No download link found. Skipping.")
                    continue

                link = a_tags[0].get_attribute("href")
                report_type = cols[1].text.strip()
                report_date = cols[2].text.strip().replace(" ", "_").replace("/", "-")
                
                # Create descriptive filename
                filename = f"{ticker}_{report_date}_{clean_filename(report_type)}.pdf"
                
                # Check if file already exists
                file_path = os.path.join(company_folder, filename)
                if os.path.exists(file_path):
                    print(f" [{idx+1}] {filename} - Already exists, skipping")
                    continue

                print(f" [{idx+1}] Downloading: {filename}")
                
                if download_file_selenium(driver, link, company_folder):
                    downloaded_count += 1
                    print(f" [{idx+1}] Downloaded successfully")
                else:
                    print(f" [{idx+1}] Download failed")
                
                # Brief pause between downloads
                time.sleep(1)

            except Exception as e:
                print(f"Failed to process report row {idx+1}: {str(e)}")
                continue
        
        print(f"[{ticker}] Downloaded {downloaded_count} reports")
        return downloaded_count
        
    except TimeoutException:
        print(f" Timeout loading reports page for {ticker}")
        return 0
    except Exception as e:
        print(f" Could not load reports for {ticker}: {str(e)}")
        return 0

# === MAIN EXECUTION ===
def main():
    driver = None
    try:
        # Setup Chrome driver
        driver = setup_chrome_driver()
        wait = WebDriverWait(driver, 15)
        
        print(" Starting ADX Financial Reports Scraper")
        print(f" Reports will be saved to: {PARENT_DIR}")
        
        # Get all company tickers
        tickers = get_all_companies(driver, wait)
        
        if not tickers:
            print("No companies found. Exiting.")
            return
        
        total_downloads = 0
        processed_companies = 0
        
        # Process each company
        for i, ticker in enumerate(tickers, 1):
            print(f"\n{'='*50}")
            print(f"Processing [{i}/{len(tickers)}]: {ticker}")
            print(f"{'='*50}")
            
            downloads = download_reports_for_company(driver, wait, ticker)
            total_downloads += downloads
            processed_companies += 1
            
            # Brief pause between companies
            time.sleep(2)
        
        print(f"\n{'='*60}")
        print(f" SCRAPING COMPLETED!")
        print(f" Processed: {processed_companies} companies")
        print(f" Total downloads: {total_downloads}")
        print(f" Reports saved to: {PARENT_DIR}")
        print(f"{'='*60}")

    except KeyboardInterrupt:
        print("\n Scraping stopped by user (Ctrl+C)")

    except Exception as e:
        print(f" Unhandled error occurred: {str(e)}")
        traceback.print_exc()

    finally:
        if driver:
            try:
                driver.quit()
                print(" Browser closed successfully")
            except:
                print(" Error closing browser")

if __name__ == "__main__":
    main()