from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import time

def execute_trade_exness(signal_type, symbol, email, password):
    chrome_options = Options()
    chrome_options.add_argument("--start-maximized")

    driver = webdriver.Chrome(options=chrome_options)
    driver.get("https://www.exness.com")

    try:
        # Login
        driver.find_element(By.LINK_TEXT, "Sign in").click()
        time.sleep(2)
        driver.find_element(By.NAME, "email").send_keys(email)
        driver.find_element(By.NAME, "password").send_keys(password)
        driver.find_element(By.CSS_SELECTOR, "button[type='submit']").click()
        time.sleep(5)

        # Navigate to trading terminal
        driver.get("https://trade.exness.com/")  # direct link
        time.sleep(10)

        # Search for market
        search_box = driver.find_element(By.XPATH, "//input[@placeholder='Search']")
        search_box.clear()
        search_box.send_keys(symbol)
        time.sleep(2)

        # Select instrument (first result)
        driver.find_element(By.XPATH, "//div[contains(text(), '" + symbol + "')]").click()
        time.sleep(3)

        # Click Buy/Sell
        if signal_type == "BUY":
            driver.find_element(By.XPATH, "//button[contains(text(),'Buy')]").click()
        elif signal_type == "SELL":
            driver.find_element(By.XPATH, "//button[contains(text(),'Sell')]").click()
        time.sleep(2)

        print(f"✅ {signal_type} trade executed on {symbol}")
    except Exception as e:
        print("❌ Trade execution failed:", e)
    finally:
        time.sleep(5)
        driver.quit()
