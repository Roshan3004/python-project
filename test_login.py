from config import ScraperConfig
from scraper import setup_driver, login_to_site
import time

def test_login():
    print("Starting login test...")
    cfg = ScraperConfig()
    
    # For testing, you can override credentials here if needed:
    # cfg.phone_number = "your_real_number"
    # cfg.password = "your_real_password"
    
    driver = setup_driver(cfg)
    try:
        print(f"Opening {cfg.base_url}...")
        driver.get(cfg.base_url)
        
        # Optional: Add delay if site loads slowly
        time.sleep(3)  
        
        print("Attempting login...")
        if login_to_site(driver, cfg):
            print("✓ Login successful!")
            print("Current URL:", driver.current_url)
            
            # Keep browser open for inspection
            input("Press Enter to close browser...")  
        else:
            print("✗ Login failed - check login_failed.png")
    finally:
        driver.quit()
        print("Browser closed")

if __name__ == "__main__":
    test_login()