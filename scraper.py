import time
import logging
import json
import os
from datetime import datetime
from bs4 import BeautifulSoup
try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.common.exceptions import TimeoutException
except ImportError:
    webdriver = None
    By = None
    WebDriverWait = None
    EC = None
    class TimeoutException(Exception):
        pass
from config import ScraperConfig
import psycopg2
import requests

# Configure logging with UTF-8 encoding
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("scraper.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def setup_driver(cfg):
    """Setup Chrome driver with minimal stable options"""
    try:
        options = webdriver.ChromeOptions()
        options.add_argument('--disable-blink-features=AutomationControlled')
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option('useAutomationExtension', False)
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        
        # Remove persistent profile to avoid corruption issues
        # options.add_argument(f'--user-data-dir={user_data_dir}')
        
        driver = webdriver.Chrome(options=options)
        driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        
        if not cfg.headless:
            driver.set_window_size(1200, 800)
            
        return driver
    except Exception as e:
        logger.error(f"Driver setup failed: {str(e)}")
        raise

def direct_login(driver, cfg):
    """Direct login at the login page URL - new approach"""
    try:
        logger.info("=== STARTING DIRECT LOGIN APPROACH ===")
        
        # Navigate directly to login page
        login_url = "https://okwin7.in/#/login"
        logger.info(f"Navigating to login page: {login_url}")
        driver.get(login_url)
        time.sleep(3)
        
        # Wait for login form to load with better error handling
        try:
            WebDriverWait(driver, 15).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='password']"))
            )
            logger.info("✓ Login page loaded")
        except TimeoutException:
            logger.error("✗ Login page failed to load - password input not found")
            driver.save_screenshot("login_page_timeout.png")
            return False
        
        # Fill phone number
        phone_input = WebDriverWait(driver, 5).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, cfg.phone_input_selector))
        )
        phone_input.clear()
        phone_input.send_keys(cfg.phone_number)
        logger.info("✓ Phone number entered")
        
        # Fill password
        password_input = WebDriverWait(driver, 5).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, cfg.password_input_selector))
        )
        password_input.clear()
        password_input.send_keys(cfg.password)
        logger.info("✓ Password entered")
        
        # Click login button using JavaScript
        login_script = """
        var loginBtn = document.querySelector('button.van-button--large') || 
                      document.querySelector('button[data-v-33f88764]') ||
                      document.evaluate("//button[contains(text(), 'Log in')]", document, null, XPathResult.FIRST_ORDERED_NODE_TYPE, null).singleNodeValue;
        if (loginBtn) {
            loginBtn.click();
            return 'clicked';
        }
        return 'not_found';
        """
        
        result = driver.execute_script(login_script)
        if result == 'clicked':
            logger.info("✓ Login button clicked")
        else:
            logger.error("✗ Login button not found")
            return False
        
        # Wait for login to complete
        time.sleep(5)
        
        # Check if we're redirected away from login page
        current_url = driver.current_url
        if "login" not in current_url:
            logger.info("✓ Login successful - redirected from login page")
            handle_post_login_checkboxes(driver)
            return True
        else:
            logger.error("✗ Login failed - still on login page")
            return False
            
    except Exception as e:
        logger.error(f"Direct login failed: {str(e)}")
        driver.save_screenshot("direct_login_failure.png")
        return False

def handle_post_login_checkboxes(driver):
    """Handle Welcome to OK.Win confirmation dialog"""
    try:
        logger.info("Checking for Welcome dialog...")
        time.sleep(3)
        
        # Check for the Welcome dialog
        welcome_dialog = driver.find_elements(By.XPATH, "//div[contains(text(), 'Welcome to OK.Win')]")
        if welcome_dialog:
            logger.info("✓ Found Welcome to OK.Win dialog")
            
            # Click the Confirm button directly using JavaScript
            confirm_script = """
            var confirmBtn = document.querySelector('button.van-dialog__confirm') || 
                           document.querySelector('button.van-button--large.van-dialog__confirm') ||
                           document.evaluate("//button[contains(text(), 'Confirm')]", document, null, XPathResult.FIRST_ORDERED_NODE_TYPE, null).singleNodeValue;
            if (confirmBtn) {
                confirmBtn.click();
                return 'clicked';
            }
            return 'not_found';
            """
            
            result = driver.execute_script(confirm_script)
            if result == 'clicked':
                logger.info("✓ Clicked Confirm button on Welcome dialog")
                time.sleep(3)
            else:
                logger.warning("Confirm button not found, trying alternative approach")
                
                # Fallback: try to find and click any visible confirm button
                confirm_buttons = driver.find_elements(By.XPATH, "//button[contains(@class, 'van-dialog__confirm')]")
                if confirm_buttons:
                    driver.execute_script("arguments[0].click();", confirm_buttons[0])
                    logger.info("✓ Clicked Confirm button via fallback")
                    time.sleep(3)
        else:
            logger.info("No Welcome dialog found")
            
    except Exception as e:
        logger.info(f"Welcome dialog handling: {str(e)}")

def navigate_after_login(driver, cfg):
    """Navigate to WinGo 1Min after login"""
    try:
        logger.info("=== NAVIGATING TO WINGO 1MIN ===")
        
        # Go to main page
        driver.get("https://okwin7.in/#/")
        time.sleep(3)
        
        # Click Lottery - use JavaScript to avoid click interception
        lottery_script = """
        var lotteryTab = document.evaluate("//div[contains(text(), 'Lottery')]", document, null, XPathResult.FIRST_ORDERED_NODE_TYPE, null).singleNodeValue;
        if (lotteryTab) {
            lotteryTab.click();
            return 'clicked';
        }
        return 'not_found';
        """
        
        result = driver.execute_script(lottery_script)
        if result == 'clicked':
            logger.info("✓ Clicked Lottery tab via JavaScript")
        else:
            logger.error("✗ Lottery tab not found")
            return False
        time.sleep(3)
        
        # Click WinGo - use JavaScript to avoid click interception
        wingo_script = """
        var wingoGame = document.evaluate("//span[contains(text(), 'Win Go')]", document, null, XPathResult.FIRST_ORDERED_NODE_TYPE, null).singleNodeValue;
        if (wingoGame) {
            wingoGame.click();
            return 'clicked';
        }
        return 'not_found';
        """
        
        result = driver.execute_script(wingo_script)
        if result == 'clicked':
            logger.info("✓ Clicked WinGo game via JavaScript")
        else:
            logger.error("✗ WinGo game not found")
            return False
        time.sleep(3)
        
        # Switch to WinGo 1 Min tab - target the exact element from HTML
        variant_script = """
        // Target the WinGo 1 Min tab specifically from the HTML structure
        var winGo1MinTab = document.querySelector('div[data-v-840acc2][class*="timer-card"]:nth-child(2)');
        if (winGo1MinTab && winGo1MinTab.textContent.includes('WinGo 1') && winGo1MinTab.textContent.includes('Min')) {
            winGo1MinTab.click();
            return 'clicked';
        }
        
        // Fallback: look for any element with "WinGo 1 Min" text
        var allElements = document.querySelectorAll('*');
        for (var i = 0; i < allElements.length; i++) {
            var el = allElements[i];
            var text = el.textContent || '';
            if (text.includes('WinGo 1') && text.includes('Min') && !text.includes('30') && !text.includes('3') && !text.includes('5')) {
                el.click();
                return 'clicked';
            }
        }
        return 'not_found';
        """
        
        result = driver.execute_script(variant_script)
        if result == 'clicked':
            logger.info("✓ Clicked 1Min tab, waiting for page update...")
            time.sleep(5)  # Longer wait for page to update
            
            # Verify the switch worked
            verify_script = """
            var currentUrl = window.location.href;
            var pageText = document.body.textContent;
            return {
                url: currentUrl,
                has_1min: pageText.includes('1 Min'),
                has_30sec: pageText.includes('30'),
                current_period: pageText.match(/\\b(2025\\d{12})\\b/)?.[1] || null
            };
            """
            verification = driver.execute_script(verify_script)
            logger.info(f"Tab switch verification: {verification}")
        else:
            logger.warning("1Min tab not found - may already be selected or different structure")
            
        # Verify we're on the right variant and get current game info
        game_info_script = """
        var info = {
            current_tab: 'unknown',
            period_id: null,
            time_remaining: null,
            last_result: null
        };
        
        // Check which tab is active
        var tabs = document.querySelectorAll('div[data-v-840acc2]');
        tabs.forEach(function(tab) {
            if (tab.textContent.includes('WinGo')) {
                if (tab.classList.contains('active') || tab.style.background || tab.style.color) {
                    info.current_tab = tab.textContent.trim();
                }
            }
        });
        
        // Get current period ID
        var periodElements = document.querySelectorAll('*');
        for (var i = 0; i < periodElements.length; i++) {
            var text = periodElements[i].textContent || '';
            var match = text.match(/\\b(2025\\d{12})\\b/);
            if (match) {
                info.period_id = match[1];
                break;
            }
        }
        
        // Get time remaining
        var timeElement = document.querySelector('.van-count-down') || 
                         document.querySelector('[class*="countdown"]');
        if (timeElement) {
            info.time_remaining = timeElement.textContent.trim();
        }
        
        return info;
        """
        
        game_info = driver.execute_script(game_info_script)
        logger.info(f"Game Info - Tab: {game_info.get('current_tab')}, Period: {game_info.get('period_id')}, Time: {game_info.get('time_remaining')}")
        
        return True
        
    except Exception as e:
        logger.error(f"Navigation failed: {str(e)}")
        return False
        
    except Exception as e:
        logger.error(f"Login failed at step: {str(e)}")
        driver.save_screenshot(f"login_error_{int(time.time())}.png")
        return False
    
def is_login_required(driver):
    """Check if login is needed by looking for specific elements"""
    try:
        # Check for login container or login-specific elements
        login_indicators = [
            driver.find_elements(By.CSS_SELECTOR, "div.signIn_container"),
            driver.find_elements(By.CSS_SELECTOR, "input[name='userNumber']"),
            driver.find_elements(By.CSS_SELECTOR, "div.signIn_container-button"),
            driver.find_elements(By.XPATH, "//*[contains(text(), 'Log in')]"),
            driver.find_elements(By.XPATH, "//*[contains(text(), 'Please log in')]"),
            driver.find_elements(By.XPATH, "//*[contains(text(), 'Token has expired')]"),
            driver.find_elements(By.XPATH, "//*[contains(text(), 'please login again')]"),
        ]
        return any(login_indicators)
    except:
        return False
    
def check_login_required(driver, cfg):
    """Check if login is required without failing"""
    try:
        # Check for login form elements
        login_elements = driver.find_elements(By.CSS_SELECTOR, cfg.login_form_selector)
        return len(login_elements) > 0
    except:
        return False

def handle_login_if_required(driver, cfg):
    """Only attempts login if login form is present"""
    if check_login_required(driver, cfg):
        logger.info("Login page detected - attempting login")
        return handle_login_page(driver, cfg)
    logger.info("No login required - continuing")
    return True
    
def handle_login_page(driver, cfg):
    """Focused login handler only called when login is needed"""
    try:
        # 1. Country Code (if exists)
        try:
            country_code = WebDriverWait(driver, 3).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, cfg.country_code_selector))
            )
            if "+91" not in country_code.text:
                country_code.click()
                WebDriverWait(driver, 2).until(
                    EC.element_to_be_clickable((By.XPATH, "//*[contains(text(),'+91')]"))
                ).click()
        except:
            logger.debug("Country code already +91 or not found")

        # 2. Phone Number (only digits)
        phone = WebDriverWait(driver, 5).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, cfg.phone_input_selector))
        )
        phone.clear()
        phone.send_keys(cfg.phone_number)  # Just the 10 digits

        # 3. Password
        logger.info("Entering password...")
        pwd = WebDriverWait(driver, 5).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, cfg.password_input_selector))
        )
        pwd.clear()
        pwd.send_keys(cfg.password)
        logger.info("Password entered successfully")

        # 4. Login Button - using simple approach to avoid WebDriver crash
        logger.info("Clicking login button...")
        time.sleep(1)
        
        # Use JavaScript to click the login button directly
        login_script = """
        var loginBtn = document.querySelector('button.van-button--large') || 
                      document.evaluate("//button[contains(text(), 'Log in')]", document, null, XPathResult.FIRST_ORDERED_NODE_TYPE, null).singleNodeValue;
        if (loginBtn) {
            loginBtn.click();
            return 'clicked';
        }
        return 'not_found';
        """
        
        result = driver.execute_script(login_script)
        if result == 'clicked':
            logger.info("✓ Login button clicked via JavaScript")
        else:
            logger.error("✗ Login button not found via JavaScript")
            driver.save_screenshot("login_button_error.png")
            return False

        # 5. Wait for login to complete
        logger.info("Waiting for login completion...")
        time.sleep(5)
        return True

    except Exception as e:
        logger.error(f"Login failed: {e}")
        driver.save_screenshot("login_failure.png")
        return False
    
def navigate_to_wingo_1min(driver, cfg):
    """Navigate to WinGo 1min game - uses the complete flow function"""
    return navigate_to_wingo(driver, cfg)

def navigate_to_lottery_section(driver, cfg):
    """Navigate to the lottery section from main page"""
    try:
        # Check for token expiration first
        if driver.find_elements(By.XPATH, "//*[contains(text(), 'Token has expired')]"):
            logger.info("Token expired - need to refresh and re-login")
            driver.refresh()
            time.sleep(3)
            if is_login_required(driver):
                if not perform_login(driver, cfg):
                    return False
        
        # Check if we're already on lottery page
        lottery_indicators = [
            "div.lottery-container", 
            "div[class*='lottery']",
            "div.lotterySlotItem",
            "div[data-v-acadf81]"
        ]
        
        for indicator in lottery_indicators:
            if driver.find_elements(By.CSS_SELECTOR, indicator):
                logger.info(f"Already on lottery page (found: {indicator})")
                return True
            
        # Method 1: Try direct URL navigation
        try:
            current_url = driver.current_url
            if "/lottery" not in current_url:
                driver.get(f"{cfg.base_url.rstrip('/')}/lottery")
                time.sleep(5)
                logger.info("Direct navigation to lottery page")
                
                # Check if login is required after direct navigation
                if is_login_required(driver):
                    logger.info("Login required after direct navigation")
                    if not perform_login(driver, cfg):
                        return False
                return True
        except Exception as e:
            logger.warning(f"Direct navigation failed: {str(e)}")
            
        # Method 2: Use sidebar navigation
        try:
            # Wait for sidebar to be present
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "div.van-sidebar, div.mySideBar"))
            )
            
            # Click lottery tab using the exact selector from DOM
            lottery_tab = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, cfg.lottery_tab_selector))
            )
            driver.execute_script("arguments[0].click();", lottery_tab)
            logger.info("Clicked lottery tab from sidebar")
            time.sleep(3)
            
            # Check if login is required after sidebar navigation
            if is_login_required(driver):
                logger.info("Login required after sidebar navigation")
                if not perform_login(driver, cfg):
                    return False
            return True
            
        except Exception as e:
            logger.error(f"Sidebar navigation failed: {str(e)}")
            return False
    except Exception as e:
        logger.error(f"Lottery section navigation failed: {str(e)}")
        return False

def navigate_to_wingo(driver, cfg):
    """Complete navigation flow: Main page → Lottery → WinGo → Login → WinGo 1min"""
    try:
        logger.info("=== STARTING COMPLETE NAVIGATION FLOW ===")
        logger.info(f"Current URL: {driver.current_url}")
        
        # Step 1: Check if we're on main page (Popular tab should be visible)
        logger.info("Step 1: Checking if on main page...")
        main_page_indicators = [
            "//div[contains(text(), 'Popular')]",
            "//div[contains(@class, 'sidebar')]//div[contains(text(), 'Popular')]",
            "//div[contains(@class, 'van-sidebar-item')]//div[contains(text(), 'Popular')]"
        ]
        
        on_main_page = False
        for selector in main_page_indicators:
            if driver.find_elements(By.XPATH, selector):
                logger.info(f"✓ On main page (found Popular tab: {selector})")
                on_main_page = True
                break
        
        if not on_main_page:
            logger.warning("Not on main page, may need to navigate back")
        
        # Step 2: Click Lottery tab
        logger.info("Step 2: Navigating to Lottery tab...")
        lottery_selectors = [
            "//div[contains(@class, 'van-sidebar-item')]//div[contains(text(), 'Lottery')]",
            "//div[contains(text(), 'Lottery')]",
            "//span[contains(text(), 'Lottery')]",
            "//button[contains(text(), 'Lottery')]"
        ]
        
        lottery_clicked = False
        for selector in lottery_selectors:
            try:
                elements = driver.find_elements(By.XPATH, selector)
                if elements:
                    driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", elements[0])
                    time.sleep(1)
                    elements[0].click()
                    logger.info(f"✓ Clicked Lottery tab using: {selector}")
                    lottery_clicked = True
                    time.sleep(3)
                    break
            except Exception as e:
                logger.debug(f"Lottery selector failed: {str(e)}")
                continue
        
        if not lottery_clicked:
            logger.error("✗ Failed to click Lottery tab")
            return False
        
        # Step 3: Find and click WinGo game
        logger.info("Step 3: Finding WinGo game...")
        wingo_selectors = [
            "//div[contains(text(), 'Win Go')]",
            "//span[contains(text(), 'Win Go')]",
            "//div[contains(text(), 'WinGo')]",
            "//span[contains(text(), 'WinGo')]",
            "//div[contains(translate(text(), 'WINGO', 'wingo'), 'win go')]",
            "//img[contains(@src, 'wingo')]",
            "//img[contains(@alt, 'wingo')]",
            "//div[contains(@class, 'lotterySlotItem')]",
            "//div[contains(@data-v-acadf81, '')]"
        ]
        
        wingo_clicked = False
        for selector in wingo_selectors:
            try:
                elements = driver.find_elements(By.XPATH, selector)
                logger.info(f"Trying WinGo selector: {selector} - Found {len(elements)} elements")
                
                for element in elements[:5]:  # Check first 5 elements
                    if element.is_displayed():
                        element_text = element.text.strip()
                        logger.info(f"WinGo candidate: {element_text[:50]}")
                        
                        # Check if this looks like WinGo
                        if ('win' in element_text.lower() and 'go' in element_text.lower()) or 'wingo' in element_text.lower() or 'lotterySlotItem' in selector or 'acadf81' in selector:
                            driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", element)
                            time.sleep(1)
                            element.click()
                            logger.info(f"✓ Clicked WinGo using: {selector}")
                            logger.info(f"Element text: {element_text}")
                            wingo_clicked = True
                            time.sleep(3)
                            break
                
                if wingo_clicked:
                    break
                    
            except Exception as e:
                logger.debug(f"WinGo selector failed: {str(e)}")
                continue
        
        if not wingo_clicked:
            logger.error("✗ Failed to find/click WinGo game")
            driver.save_screenshot("wingo_not_found.png")
            return False
        
        # Step 4: Check if we're on login page and handle login
        logger.info("Step 4: Checking if login is required...")
        time.sleep(2)
        
        login_indicators = [
            "//input[@type='password']",
            "//input[contains(@placeholder, 'password')]",
            "//input[contains(@placeholder, 'Password')]",
            "//button[contains(text(), 'Login')]",
            "//button[contains(text(), 'Sign')]",
            "//div[contains(text(), 'Login')]"
        ]
        
        needs_login = False
        for selector in login_indicators:
            if driver.find_elements(By.XPATH, selector):
                logger.info(f"✓ Login page detected: {selector}")
                needs_login = True
                break
        
        if needs_login:
            logger.info("Performing login...")
            login_success = perform_login(driver, cfg)
            if not login_success:
                logger.error("✗ Login failed")
                return False
            logger.info("✓ Login completed")
            time.sleep(3)
        else:
            logger.info("✓ No login required")
        
        # Step 5: Switch from WinGo 30sec to WinGo 1min
        logger.info("Step 5: Switching to WinGo 1min...")
        variant_selectors = [
            "//div[contains(text(), '1Min')]",
            "//span[contains(text(), '1Min')]",
            "//button[contains(text(), '1Min')]",
            "//div[contains(text(), '1 Min')]",
            "//span[contains(text(), '1 Min')]",
            "//div[contains(@class, 'time')]//div[contains(text(), '1')]"
        ]
        
        variant_switched = False
        for selector in variant_selectors:
            try:
                elements = driver.find_elements(By.XPATH, selector)
                if elements:
                    for element in elements:
                        if element.is_displayed():
                            driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", element)
                            time.sleep(1)
                            element.click()
                            logger.info(f"✓ Switched to 1Min variant using: {selector}")
                            variant_switched = True
                            time.sleep(2)
                            break
                if variant_switched:
                    break
            except Exception as e:
                logger.debug(f"Variant selector failed: {str(e)}")
                continue
        
        if not variant_switched:
            logger.warning("⚠ Could not find 1Min variant, may already be selected or not available")
        
        # Handle any dialogs that might appear
        handle_dialogs(driver)
        
        logger.info("=== NAVIGATION FLOW COMPLETED ===")
        return True
        
    except Exception as e:
        logger.error(f"Navigation flow failed: {str(e)}")
        driver.save_screenshot("navigation_error.png")
        save_debug_info(driver)
        return False

def select_1min_variant(driver, cfg):
    """Select the 1 minute variant of WinGo"""
    try:
        # Wait for game variants to load
        time.sleep(3)
        
        # Look for 1Min option
        min1_selectors = [
            "//button[contains(text(), '1 Min') and not(contains(text(), '5 Min'))]",
            "//div[contains(text(), '1 Min') and not(contains(text(), '5 Min'))]",
            "//span[contains(text(), '1 Min')]/ancestor::button",
            "button:contains('1 Min'):not(:contains('5 Min'))"
        ]
        
        min1_element = None
        for selector in min1_selectors:
            try:
                if selector.startswith('//'):
                    min1_element = driver.find_element(By.XPATH, selector)
                else:
                    min1_element = driver.find_element(By.CSS_SELECTOR, selector)
                logger.info(f"Found 1Min variant using: {selector}")
                break
            except:
                continue
                
        if not min1_element:
            logger.warning("1Min variant not found, proceeding anyway")
            return True
            
        # Click 1Min variant
        driver.execute_script("arguments[0].click();", min1_element)
        logger.info("1Min variant selected")
        time.sleep(2)
        return True
        
    except Exception as e:
        logger.error(f"1Min selection failed: {str(e)}")
        return False

def save_debug_info(driver):
    """Save debugging information when navigation fails"""
    try:
        with open("page_source.html", "w", encoding="utf-8") as f:
            f.write(driver.execute_script("""
                document.querySelectorAll('*').forEach(el => {
                    el.setAttribute('data-visible', el.checkVisibility());
                    el.setAttribute('data-clickable', el.getBoundingClientRect().width > 0);
                });
                return document.documentElement.outerHTML;
            """))
        logger.info("Saved enhanced page_source.html for debugging")
    except:
        pass

def analyze_page_structure(driver):
    """Analyze current page structure for debugging"""
    try:
        # Get all visible elements with text content (sanitized)
        elements_info = driver.execute_script("""
            let info = [];
            document.querySelectorAll('*').forEach(el => {
                if (el.offsetWidth > 0 && el.offsetHeight > 0) {
                    let text = el.textContent.trim().replace(/[^\x00-\x7F]/g, "?");
                    if (text && text.length < 50) {
                        info.push({
                            tag: el.tagName,
                            class: el.className,
                            id: el.id,
                            text: text,
                            clickable: el.getBoundingClientRect().width > 0
                        });
                    }
                }
            });
            return info;
        """)
        
        logger.info("=== PAGE STRUCTURE ANALYSIS ===")
        for elem in elements_info[:15]:  # Show first 15 elements
            try:
                logger.info(f"Tag: {elem['tag']}, Class: {elem['class'][:30]}, Text: {elem['text'][:30]}")
            except:
                pass
        
        # Look specifically for game-related elements
        game_elements = [e for e in elements_info if any(keyword in e['text'].lower() for keyword in ['win', 'go', 'game', 'lottery'])]
        logger.info("=== GAME-RELATED ELEMENTS ===")
        for elem in game_elements[:10]:  # Limit to 10 elements
            try:
                logger.info(f"GAME: {elem['tag']} - {elem['text'][:30]} (Class: {elem['class'][:20]})")
            except:
                pass
            
    except Exception as e:
        logger.error(f"Page analysis failed: {str(e)}")

def handle_token_expiration(driver, cfg):
    """Handle token expiration scenarios"""
    try:
        # Check for token expiration indicators
        token_expired_indicators = [
            "//*[contains(text(), 'Token has expired')]",
            "//*[contains(text(), 'please login again')]",
            "//*[contains(text(), 'Session expired')]",
            "div.van-toast--fail"
        ]
        
        for indicator in token_expired_indicators:
            if driver.find_elements(By.XPATH if indicator.startswith('//') else By.CSS_SELECTOR, indicator):
                logger.info(f"Token expiration detected: {indicator}")
                
                # Refresh page and re-login
                driver.refresh()
                time.sleep(5)
                
                if is_login_required(driver):
                    logger.info("Re-login required after token expiration")
                    return perform_login(driver, cfg)
                else:
                    logger.info("Page refreshed, no login required")
                    return True
                    
        return True
        
    except Exception as e:
        logger.error(f"Token expiration handling failed: {str(e)}")
        return False
    
def handle_post_login_dialogs(driver, cfg):
    """Handle all confirmation dialogs after login"""
    try:
        # Wait for any dialogs to appear
        time.sleep(3)
        
        # Check for token expiration toast first
        token_expired = driver.find_elements(By.XPATH, "//*[contains(text(), 'Token has expired')]")
        if token_expired:
            logger.info("Token expired toast detected - dismissing")
            # Wait for toast to disappear or click to dismiss
            time.sleep(3)
        
        # Look for visible dialogs
        dialog_selectors = [
            "div.van-popup.van-popup--center.van-dialog[style*='z-index: 2001']",
            "div[role='dialog'][tabindex='0']",
            "div.van-dialog:not([style*='display: none'])",
            "div.van-popup:not([style*='display: none'])"
        ]
        
        dialogs_handled = 0
        max_dialogs = 3  # Prevent infinite loop
        
        while dialogs_handled < max_dialogs:
            dialog_found = False
            
            for selector in dialog_selectors:
                try:
                    dialogs = driver.find_elements(By.CSS_SELECTOR, selector)
                    visible_dialogs = [d for d in dialogs if d.is_displayed()]
                    
                    if visible_dialogs:
                        dialog = visible_dialogs[0]
                        dialog_found = True
                        logger.info(f"Found dialog using selector: {selector}")
                        
                        # Look for confirm button within this dialog
                        confirm_selectors = [
                            "button.van-button.van-button--default.van-button--large.van-dialog__confirm",
                            "button[class*='confirm']",
                            "button:contains('Confirm')",
                            ".van-dialog__footer button",
                            "button.van-button"
                        ]
                        
                        button_clicked = False
                        for confirm_sel in confirm_selectors:
                            try:
                                if confirm_sel.startswith('button:contains'):
                                    confirm_btn = dialog.find_element(By.XPATH, ".//button[contains(text(), 'Confirm')]")
                                else:
                                    confirm_btn = dialog.find_element(By.CSS_SELECTOR, confirm_sel)
                                    
                                if confirm_btn.is_displayed() and confirm_btn.is_enabled():
                                    driver.execute_script("arguments[0].click();", confirm_btn)
                                    logger.info(f"Clicked confirm button: {confirm_sel}")
                                    time.sleep(2)
                                    button_clicked = True
                                    dialogs_handled += 1
                                    break
                            except:
                                continue
                        
                        if button_clicked:
                            break
                        else:
                            # Try clicking anywhere on dialog to dismiss
                            try:
                                driver.execute_script("arguments[0].click();", dialog)
                                logger.info("Clicked dialog to dismiss")
                                time.sleep(2)
                                dialogs_handled += 1
                                break
                            except:
                                pass
                                
                except Exception as e:
                    logger.debug(f"Dialog selector {selector} failed: {str(e)}")
                    continue
            
            if not dialog_found:
                break
                
        logger.info(f"Handled {dialogs_handled} dialogs")
        return True
        
    except Exception as e:
        logger.error(f"Dialog handling failed: {str(e)}")
        driver.save_screenshot("dialog_error.png")
        return False

def scrape_page(driver, url, cfg):
    """Scrape WinGo 1 Min results from the actual page structure"""
    try:
        logger.info("Starting page scrape...")
        
        # Wait up to ~8s for 1-min history rows to render
        try:
            rows_ready = False
            for _ in range(8):
                count = driver.execute_script("return document.querySelectorAll('div.record-body > div.van-row').length")
                if count and int(count) > 0:
                    rows_ready = True
                    logger.info(f"History rows detected: {count}")
                    break
                time.sleep(1)
            if not rows_ready:
                logger.warning("History rows not detected yet; proceeding anyway")
        except Exception as _e:
            logger.debug("Row readiness check failed; proceeding")

        # Get current period ID from the lottery info section
        current_period_script = """
        // Try multiple header containers to find the current period number
        var selectors = [
            'div.lottery-info.padding',
            '.lottery-info',
            'div[class*="lottery-info"]',
            '.lottery-header',
            '.lottery-top',
            '.lotteryTitle',
            '.period',
            '[class*="period"]'
        ];
        var headerText = '';
        for (var i = 0; i < selectors.length; i++) {
            var el = document.querySelector(selectors[i]);
            if (el && (el.innerText || el.textContent)) {
                headerText = (el.innerText || el.textContent).replace(/\s+/g, ' ').trim();
                if (headerText) break;
            }
        }
        // First attempt: match 2025 + 12-16 digits directly
        var match = headerText ? headerText.match(/(2025\d{12,16})/) : null;
        if (match) return match[1];
        // Fallback: search any element containing 2025 within the top area
        var xpathNode = document.evaluate("//div[contains(., '2025')]", document, null, XPathResult.FIRST_ORDERED_NODE_TYPE, null).singleNodeValue;
        if (xpathNode && (xpathNode.innerText || xpathNode.textContent)) {
            var t = (xpathNode.innerText || xpathNode.textContent).replace(/\s+/g, ' ').trim();
            var m2 = t.match(/(2025\d{12,16})/);
            if (m2) return m2[1];
        }
        // Final fallback: search whole document text
        var docText = (document.body.innerText || document.body.textContent || '').replace(/\s+/g, ' ');
        var m3 = docText.match(/(2025\d{12,16})/);
        return m3 ? m3[1] : null;
        """
        
        current_period = driver.execute_script(current_period_script)
        logger.info(f"Current period detected: {current_period}")
        
        # Target the actual history section with rows inside record-body
        results_script = """
        var results = [];
        
        // Target rows under the history container
        var historyRows = document.querySelectorAll('div.record-body > div.van-row');
        
        var rowDebug = [];
        historyRows.forEach(function(row, idx) {
            var numberBadge = row.querySelector('.record-body-num') || row.querySelector('[class*="record-body-num"]');
            var sizeCell = row.querySelector('div.van-col--5');
            var rowText = (row.innerText || row.textContent || '').replace(/\s+/g,' ').trim();
            var sizeText = sizeCell ? (sizeCell.innerText || sizeCell.textContent || '') : '';

            // Period: anywhere in the row text (2025 + 12-16 digits), no word boundaries
            var periodMatch = rowText.match(/(2025\d{12,16})/);
            // Fallback: compact digits and try to find 2025 + 13 digits (17 total) from the compacted string
            if (!periodMatch) {
                var compact = rowText.replace(/[^0-9]/g, '');
                var alt = compact.match(/(2025\d{13})/);
                if (alt) periodMatch = alt;
            }
            // Number: prefer badge text; otherwise fallback to first standalone digit in row
            var numberText = numberBadge ? (numberBadge.innerText || numberBadge.textContent || '').trim() : '';
            var numberMatch = numberText.match(/\b([0-9])\b/) || rowText.match(/\b([0-9])\b/) || numberText.match(/([0-9])/);

            if (periodMatch && numberMatch) {
                var period = periodMatch[1];
                var number = numberMatch[1];
                var n = parseInt(number);
                var color = (n === 0 || n === 5) ? 'VIOLET' : (n % 2 === 0 ? 'GREEN' : 'RED');
                var size = /\bBig\b/i.test(sizeText) ? 'Big' : (/\bSmall\b/i.test(sizeText) ? 'Small' : (n >= 5 ? 'Big' : 'Small'));
                results.push({ period_id: period, number: number, color: color, big_small: size });
            } else if (idx < 5) {
                rowDebug.push({
                    rowText: rowText.slice(0,160),
                    numberText: numberText,
                    sizeText: sizeText,
                    periodFound: !!periodMatch,
                    numberFound: !!numberMatch
                });
            }
        });

        if (results.length === 0) { return { debug: rowDebug }; }
        return results;
        """
        
        records = driver.execute_script(results_script) or []
        
        # If no records found via JavaScript, try BeautifulSoup fallback
        if isinstance(records, dict) and records.get('debug') is not None:
            logger.info(f"Row debug (first rows): {json.dumps(records.get('debug')[:3])}")
            records = []
        if not records:
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            
            # Look for any elements containing period IDs (17-digit numbers starting with 2025)
            for element in soup.find_all(string=lambda s: s and len(str(s).strip()) == 17 and str(s).strip().startswith('2025')):
                period_id = str(element).strip()
                parent = element.parent
                
                # Try to find associated number in nearby elements
                number_text = None
                for sibling in parent.find_all():
                    text = sibling.get_text(strip=True)
                    if text.isdigit() and len(text) == 1:
                        number_text = text
                        break
                
                if number_text:
                    records.append({
                        "period_id": period_id,
                        "number": number_text,
                        "color": cfg.color_from_number(number_text),
                        "scraped_at": datetime.utcnow().isoformat()
                    })
        
        # Add timestamp to JavaScript-scraped records
        for record in records:
            if "scraped_at" not in record:
                record["scraped_at"] = datetime.utcnow().isoformat()
        
        if records:
            latest = records[0]
            logger.info(f"Successfully scraped {len(records)} records | Latest: period={latest.get('period_id')} number={latest.get('number')} color={latest.get('color')} big_small={latest.get('big_small')}")
            if current_period:
                logger.info(f"Current running period: {current_period}")
        else:
            logger.info("Successfully scraped 0 records")
            try:
                driver.save_screenshot("wingo_1min_page.png")
                save_debug_info(driver)
            except Exception:
                pass
        return records
        
    except Exception as e:
        logger.error(f"Scraping failed: {str(e)}")
        return []

def save_to_neon(records, conn_str):
    """Save records to Neon PostgreSQL"""
    if not records:
        return 0
        
    conn = None
    try:
        conn = psycopg2.connect(conn_str)
        cursor = conn.cursor()
        
        # Create table if not exists
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS game_history (
                period_id VARCHAR(50) PRIMARY KEY,
                number VARCHAR(10),
                color VARCHAR(10),
                scraped_at TIMESTAMP
            )
        """)
        
        # Insert records
        inserted = 0
        for record in records:
            cursor.execute("""
                INSERT INTO game_history (period_id, number, color, scraped_at)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (period_id) DO NOTHING
            """, (
                record['period_id'],
                record['number'],
                record['color'],
                record['scraped_at']
            ))
            try:
                # For INSERT ... DO NOTHING, rowcount is 1 if inserted, 0 if skipped
                inserted += max(0, cursor.rowcount or 0)
            except Exception:
                pass
        
        conn.commit()
        return inserted
    except Exception as e:
        logger.error(f"Database error: {e}")
        return 0
    finally:
        if conn:
            conn.close()

def get_total_rows(conn_str) -> int:
    """Return total rows in game_history."""
    conn = None
    try:
        conn = psycopg2.connect(conn_str)
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS game_history (
                period_id VARCHAR(50) PRIMARY KEY,
                number VARCHAR(10),
                color VARCHAR(10),
                scraped_at TIMESTAMP
            )
        """)
        cur.execute("SELECT COUNT(*) FROM game_history")
        (count,) = cur.fetchone()
        return int(count or 0)
    except Exception as e:
        logger.warning(f"Total rows check failed: {e}")
        return 0
    finally:
        if conn:
            conn.close()

def _normalize_color_from_api(color_str: str, number: int) -> str:
    """Normalize color coming from API, fallback to number rule."""
    try:
        s = (color_str or "").upper()
        if "VIOLET" in s:
            return "VIOLET"
        if s in ("RED", "GREEN"):
            return s
    except Exception:
        pass
    # Fallback based on number
    return ScraperConfig.color_from_number(number)

def _http_get_with_retries(url: str, params: dict, headers: dict, timeout: int = 20, retries: int = 4):
    """GET with simple exponential backoff for 403/429/5xx."""
    session = requests.Session()
    backoff = 1.5
    for attempt in range(retries):
        try:
            r = session.get(url, params=params, headers=headers, timeout=timeout)
            if r.status_code in (403, 429) or 500 <= r.status_code < 600:
                raise requests.HTTPError(f"{r.status_code} {r.reason}")
            r.raise_for_status()
            return r
        except Exception as e:
            if attempt == retries - 1:
                raise
            # Backoff with small jitter
            sleep_s = backoff ** attempt + (attempt * 0.1)
            logger.warning(f"HTTP retry {attempt+1}/{retries} after error: {e}; sleeping {sleep_s:.1f}s")
            time.sleep(sleep_s)

def _send_discord_webhook(webhook_url: str, message: str) -> None:
    """Send a simple message to a Discord webhook (best-effort)."""
    if not webhook_url:
        return
    try:
        payload = {"content": message}
        headers = {"Content-Type": "application/json"}
        # Best-effort with short timeout; ignore failures
        requests.post(webhook_url, data=json.dumps(payload), headers=headers, timeout=8)
    except Exception:
        pass

def _send_telegram_message(bot_token: str, chat_id: str, text: str) -> None:
    """Send a message via Telegram Bot API (best-effort)."""
    if not bot_token or not chat_id:
        return
    try:
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        payload = {"chat_id": chat_id, "text": text}
        requests.post(url, json=payload, timeout=8)
    except Exception:
        pass

def fetch_history_once(cfg: ScraperConfig):
    """Fetch recent history via public API and map to records for DB."""
    url = "https://draw.ar-lottery01.com/WinGo/WinGo_1M/GetHistoryIssuePage.json"
    params = {"ts": int(time.time() * 1000)}
    headers = {
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
        "Origin": "https://okwin7.in",
        "Referer": "https://okwin7.in/",
        "Connection": "keep-alive",
        "X-Requested-With": "XMLHttpRequest",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    }
    try:
        r = _http_get_with_retries(url, params=params, headers=headers, timeout=15, retries=4)
        data = r.json()
        if not isinstance(data, dict) or data.get("code") != 0:
            logger.warning(f"API returned non-success: {data}")
            return []
        items = (data.get("data") or {}).get("list") or []
        records = []
        for it in items:
            try:
                period_id = str(it.get("issueNumber"))
                num_str = str(it.get("number"))
                if not period_id or not num_str or not num_str.isdigit():
                    continue
                n = int(num_str)
                color_api = it.get("color")
                color = _normalize_color_from_api(color_api, n)
                records.append({
                    "period_id": period_id,
                    "number": str(n),
                    "color": color,
                    "scraped_at": datetime.utcnow().isoformat()
                })
            except Exception:
                continue
        return records
    except Exception as e:
        logger.error(f"API fetch failed: {e}")
        return []

def fetch_history_page(cfg: ScraperConfig, page_no: int):
    """Fetch a specific page from history if the API supports pagination (best-effort)."""
    url = "https://draw.ar-lottery01.com/WinGo/WinGo_1M/GetHistoryIssuePage.json"
    params = {"ts": int(time.time() * 1000), "pageNo": page_no}
    headers = {
        "Accept": "application/json, text/plain, */*",
        "Origin": "https://okwin7.in",
        "Referer": "https://okwin7.in/",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    }
    try:
        r = _http_get_with_retries(url, params=params, headers=headers, timeout=20, retries=4)
        data = r.json()
        if not isinstance(data, dict) or data.get("code") != 0:
            return []
        items = (data.get("data") or {}).get("list") or []
        mapped = []
        for it in items:
            period_id = str(it.get("issueNumber"))
            num_str = str(it.get("number"))
            if not period_id or not num_str or not num_str.isdigit():
                continue
            n = int(num_str)
            color = _normalize_color_from_api(it.get("color"), n)
            mapped.append({
                "period_id": period_id,
                "number": str(n),
                "color": color,
                "scraped_at": datetime.utcnow().isoformat()
            })
        return mapped
    except Exception:
        return []

def get_db_last_seen(conn_str: str) -> str | None:
    """Return the max period_id from DB, or None if table empty/not exists."""
    try:
        conn = psycopg2.connect(conn_str)
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS game_history (
                period_id VARCHAR(50) PRIMARY KEY,
                number VARCHAR(10),
                color VARCHAR(10),
                scraped_at TIMESTAMP
            )
        """)
        conn.commit()
        cur.execute("SELECT MAX(period_id) FROM game_history")
        row = cur.fetchone()
        cur.close()
        conn.close()
        return row[0] if row and row[0] else None
    except Exception as e:
        logger.warning(f"Could not read last_seen from DB: {e}")
        return None

def backfill_from_history(cfg: ScraperConfig, last_seen: str | None, max_pages: int = 50) -> int:
    """Backfill older pages until we reach last_seen or hit max_pages. Returns inserted count."""
    if last_seen is None:
        # No prior state; do not backfill deep by default
        return 0
    inserted_total = 0
    for page in range(1, max_pages + 1):
        page_records = fetch_history_page(cfg, page)
        if not page_records:
            break
        # Keep only records newer than last_seen
        new_recs = [r for r in page_records if r["period_id"] > last_seen]
        if not new_recs:
            # We've reached or passed last_seen; stop
            break
        # Save and continue until no more new
        inserted_total += save_to_neon(new_recs, cfg.neon_conn_str)
        logger.info(f"Backfill page {page}: fetched {len(page_records)}, new {len(new_recs)}, inserted {inserted_total}")
        # Update sentinel to the max we just saved to avoid re-saving across pages
        last_seen = max(r["period_id"] for r in new_recs)
    return inserted_total

def _compute_next_tick(now_epoch: float, offset_seconds: int) -> float:
    """Return epoch seconds for the next minute boundary + offset.

    Example: if now=12:34:18 and offset=10 → next tick = 12:35:10.
    """
    next_minute = int(now_epoch // 60 + 1) * 60
    return float(next_minute + max(0, int(offset_seconds)))

def api_poll_loop(cfg: ScraperConfig):
    """Poll the API aligned to wall clock: every minute at mm:+offset.

    This avoids drift from variable fetch durations. If a fetch takes 8–12s,
    the next cycle still begins at the next minute boundary + offset.
    """
    logger.info("=== Starting API-based WinGo 1-Min poller ===")
    # Initialize from DB so we can recover from downtime
    last_seen = get_db_last_seen(cfg.neon_conn_str)
    if last_seen:
        logger.info(f"DB last_seen period: {last_seen}; attempting backfill...")
        try:
            inserted = backfill_from_history(cfg, last_seen)
            logger.info(f"Backfill completed, inserted {inserted} missed rows")
        except Exception as e:
            logger.warning(f"Backfill skipped due to error: {e}")
    # Alert tracking
    last_insert_ts = time.time()
    alert_sent = False

    while True:
        try:
            # Calculate next tick and sleep until then
            now = time.time()
            next_tick = _compute_next_tick(now, cfg.scrape_offset_seconds)
            
            # If we're already past the next tick, skip ahead to the next interval
            if now > next_tick:
                next_tick = _compute_next_tick(next_tick, cfg.scrape_offset_seconds)
            
            # Sleep until the next tick
            sleep_time = max(0.1, next_tick - time.time())
            logger.debug(f"Next run in {sleep_time:.2f}s at {datetime.fromtimestamp(next_tick).strftime('%H:%M:%S.%f')[:-3]}")
            time.sleep(sleep_time)

            # Fetch and process data
            recs = fetch_history_once(cfg)
            if recs:
                # Keep only unseen periods (assuming list is latest-first)
                if last_seen:
                    new_recs = [r for r in recs if r["period_id"] > last_seen]
                else:
                    new_recs = recs
                
                attempted = len(new_recs)
                saved = save_to_neon(new_recs, cfg.neon_conn_str) if new_recs else 0
                
                if new_recs:
                    last_seen = max(r["period_id"] for r in new_recs)
                
                duplicates = max(0, attempted - saved)
                logger.info(f"Fetched {len(recs)}, new {len(new_recs)}; attempted {attempted}, saved {saved}, duplicates {duplicates}. Last seen: {last_seen}")
                
                # Update alert state if we saved new data
                if saved > 0:
                    last_insert_ts = time.time()
                    alert_sent = False
            else:
                logger.warning("No records returned from API this round")
                
            # Log total rows periodically
            try:
                total = get_total_rows(cfg.neon_conn_str)
                logger.info(f"Total rows so far: {total}")
            except Exception as e:
                logger.warning(f"Could not get total rows: {e}")
                
            # Check for alert conditions
            try:
                if cfg.alert_no_new_mins > 0 and cfg.telegram_bot_token and cfg.telegram_chat_id:
                    minutes_since = (time.time() - last_insert_ts) / 60.0
                    if minutes_since >= cfg.alert_no_new_mins and not alert_sent:
                        _send_telegram_message(
                            cfg.telegram_bot_token,
                            cfg.telegram_chat_id,
                            f"WinGo scraper alert: no new rows inserted for {int(minutes_since)} minutes (threshold {cfg.alert_no_new_mins}m)."
                        )
                        alert_sent = True
            except Exception as e:
                logger.warning(f"Alert check failed: {e}")
                
        except Exception as e:
            logger.error(f"Unexpected error in poll loop: {e}")
            # Add a small delay before retrying to prevent tight error loops
            time.sleep(5)

def main():
    """Main entrypoint: use lightweight API poller (no Selenium)."""
    cfg = ScraperConfig()
    # Support one-shot execution for cron/CI runners
    oneshot = os.getenv("ONESHOT", "").lower() in ("1", "true", "yes")
    if oneshot:
        try:
            last_seen = get_db_last_seen(cfg.neon_conn_str)
            if last_seen:
                logger.info(f"[One-shot] DB last_seen: {last_seen}; attempting backfill...")
                try:
                    inserted = backfill_from_history(cfg, last_seen)
                    logger.info(f"[One-shot] Backfill inserted {inserted} rows")
                    total = get_total_rows(cfg.neon_conn_str)
                    logger.info(f"[One-shot] Total rows after backfill: {total}")
                except Exception as e:
                    logger.warning(f"[One-shot] Backfill skipped: {e}")
            recs = fetch_history_once(cfg)
            if recs:
                new_recs = [r for r in recs if (not last_seen) or r["period_id"] > last_seen]
                saved = save_to_neon(new_recs, cfg.neon_conn_str) if new_recs else 0
                logger.info(f"[One-shot] fetched {len(recs)}, new {len(new_recs)}, saved {saved}")
                total = get_total_rows(cfg.neon_conn_str)
                logger.info(f"[One-shot] Total rows after save: {total}")
            else:
                logger.info("[One-shot] no records fetched")
        except Exception as e:
            logger.error(f"[One-shot] error: {e}")
        return
    # Default long-running poller
    try:
        api_poll_loop(cfg)
    except KeyboardInterrupt:
        logger.info("Shutting down poller...")

def perform_login(driver, cfg):
    """Perform login with enhanced error handling"""
    try:
        # Phone input
        phone = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, cfg.phone_input_selector))
        )
        phone.clear()
        phone.send_keys(cfg.phone_number)
        logger.info("Phone number entered")
        
        # Password input
        pwd = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, cfg.password_input_selector))
        )
        pwd.clear()
        pwd.send_keys(cfg.password)
        logger.info("Password entered")
        
        # Login button
        login_btn = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, cfg.login_button_selector))
        )
        driver.execute_script("arguments[0].click();", login_btn)
        logger.info("Login button clicked")
        
        # Wait for login to complete
        WebDriverWait(driver, 15).until_not(
            EC.presence_of_element_located((By.CSS_SELECTOR, cfg.login_form_selector))
        )
        logger.info("Login successful")
        
        # Handle any immediate post-login dialogs
        time.sleep(2)
        handle_post_login_dialogs(driver, cfg)
        
        return True
        
    except Exception as e:
        logger.error(f"Login failed: {str(e)}")
        driver.save_screenshot("login_failure.png")
        return False

def navigate_to_wingo_with_full_flow(driver, cfg):
    """Complete navigation flow to WinGo with all error handling"""
    try:
        # Step 1: Handle token expiration if present
        if not handle_token_expiration(driver, cfg):
            logger.error("Token expiration handling failed")
            return False
        
        # Step 2: Ensure we're on the right starting point
        current_url = driver.current_url
        logger.info(f"Starting navigation from: {current_url}")
        
        # Step 3: Analyze page structure for debugging
        analyze_page_structure(driver)
        
        # Step 4: Navigate to lottery section
        if not navigate_to_lottery_section(driver, cfg):
            logger.error("Failed to reach lottery section")
            return False
            
        # Step 5: Navigate to WinGo (complete flow)
        if not navigate_to_wingo(driver, cfg):
            logger.error("Failed to navigate to WinGo")
            return False
            
        # Step 6: Handle any login that appears after WinGo selection
        if is_login_required(driver):
            logger.info("Login required after WinGo selection")
            if not perform_login(driver, cfg):
                return False
                
        # Step 7: Handle post-game-selection dialogs
        handle_post_login_dialogs(driver, cfg)
        
        # Step 8: Select 1Min variant
        if not select_1min_variant(driver, cfg):
            logger.warning("Could not select 1Min variant, continuing anyway")
            
        logger.info("WinGo navigation completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Full navigation flow failed: {str(e)}")
        save_debug_info(driver)
        return False

def prepare_for_scraping(driver, cfg):
    """Prepare the page for scraping"""
    try:
        # Scroll to history section
        driver.execute_script("""
            let historySection = document.querySelector('div.record-body, div[class*="history"], div[class*="record"]');
            if (historySection) {
                historySection.scrollIntoView({
                    behavior: 'smooth',
                    block: 'center'
                });
            }
        """)
        time.sleep(2)
        logger.info("Page prepared for scraping")
    except Exception as e:
        logger.warning(f"Page preparation failed: {str(e)}")

if __name__ == "__main__":
    main()