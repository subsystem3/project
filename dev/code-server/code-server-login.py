from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager

# setup webdriver
s = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=s)

# open the website
driver.get("https://dev.subsystem3.ai/login")

# find the elements for password input
password_input = driver.find_element(
    By.CSS_SELECTOR, "body > div > div > div.content > form > div > input.password"
)

# enter the password and submit
password_input.send_keys("Hesitancy#Polygon0#Unless")
password_input.send_keys(Keys.RETURN)

# wait for the live share button to be clickable
wait = WebDriverWait(driver, 10)
live_share_button = wait.until(
    EC.element_to_be_clickable(
        (
            By.CSS_SELECTOR,
            "#ms-vsliveshare\.vsliveshare\.liveshare\.userStatusBarItem > a",
        )
    )
)

# click the button to start a live share session
live_share_button.click()

# click on the more info button
more_info_button = driver.find_element(
    By.CSS_SELECTOR,
    "#list_id_15_0 > div.notification-list-item.expanded > div.notification-list-item-details-row > div.notification-list-item-buttons-container > a:nth-child(2)",
)
more_info_button.click()
