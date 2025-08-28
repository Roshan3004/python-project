import os
from dataclasses import dataclass, field
from typing import List

@dataclass
class ScraperConfig:
    # ====== CORE SETTINGS ======
    base_url: str = "https://okwin7.in/"
    headless: bool = False
    
    # ====== CREDENTIALS (from environment only) ======
    phone_number: str = os.getenv("OKWIN_PHONE", "")
    password: str = os.getenv("OKWIN_PASSWORD", "")
    neon_conn_str: str = os.getenv("NEON_CONN_STR", "")

    # ====== LOGIN SELECTORS ======
    login_form_selector = "div.signIn_container"  # Main container
    country_code_selector: str = "div.dropdown > div.van-dropdown-menu_title"
    phone_input_selector: str = "input[name='userNumber'], input[type='tel']"
    password_input_selector: str = "input[type='password'][maxlength='32']"
    login_button_selector = "div.signIn_container-button > button.van-button--large"  # Exact from HTML
    remember_checkbox_selector = "div.van-checkbox"  # Remember me checkbox
    logged_in_selector: str = "div.user-avatar, img.account-icon"

    # ====== NAVIGATION SELECTORS (UPDATED FROM SCREENSHOTS) ======
    sidebar_container = "div.van-sidebar, div.mySideBar"  # Based on actual DOM
    lottery_tab_selector = "div.van-sidebar-item[id='gameType-lottery']"  # Exact selector from DOM
    popular_tab_selector = "div.van-sidebar-item[id='gameType-popular']"  # Popular tab selector
    lottery_container: str = "div.lottery-container, div.lotterySlotItem"  
    wingo_game_selector: str = "div[data-v-acadf81][class*='lotterySlotItem']"  # WinGo game container
    wingo_text_selector: str = "span[data-v-acadf81]:contains('Win Go')"  # WinGo text element
    
    # Time variant selectors
    wingo_1min_selector: str = "div.van-sidebar-item:contains('1 Min'), button:contains('1 Min')"

    # ====== WINGO 1MIN SELECTORS (Updated from HTML) ======
    history_container_selector: str = "div.lottery-info.padding"
    row_selector: str = "div.timer-card"
    period_selector: str = "div.lottery-info.padding" 
    number_selector: str = "div.timer-card"
    color_selector: str = "div.timer-card"
    
    # Current game selectors
    current_period_selector: str = "div.lottery-info.padding"
    timer_selector: str = "div.van-count-down"

    # ====== TIMING ======
    page_load_timeout: int = 20  # Increased for slow loading
    element_timeout: int = 15   # More generous waits
    scrape_interval: int = int(os.getenv("SCRAPE_INTERVAL", "60"))
    # Wall-clock alignment: seconds after the minute to poll (e.g., 10 => mm:+10s)
    scrape_offset_seconds: int = int(os.getenv("SCRAPE_OFFSET_SECONDS", "10"))

    # ====== ALERTING ======
    # Telegram Bot settings
    telegram_bot_token: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    telegram_chat_id: str = os.getenv("TELEGRAM_CHAT_ID", "")
    # Minutes with no successful inserts before sending alert (e.g., 120 or 180)
    alert_no_new_mins: int = int(os.getenv("ALERT_NO_NEW_MINS", "180"))

    # ====== DIALOG HANDLING (UPDATED) ======
    dialog_overlay_selector: str = "div.van-popup.van-popup--center.van-dialog[style*='z-index: 2001']"
    dialog_content_selector: str = "div.van-dialog__content"
    dialog_confirm_text: str = "span.van-button__text:contains('Confirm')"
    confirm_button_selector: str = "button.van-button.van-button--default.van-button--large.van-dialog__confirm"
    welcome_dialog_selector: str = "div.promptHeader:contains('Welcome to OK.Win')"
    bonus_dialog_selector: str = "div:contains('Cumulative')"

    @staticmethod
    def color_from_number(n: int) -> str:
        """Determine color from number (0-9)"""
        try:
            n = int(n)
            if n in (0, 5):
                return "VIOLET"
            return "GREEN" if n % 2 == 0 else "RED"
        except (ValueError, TypeError):
            return "UNKNOWN"