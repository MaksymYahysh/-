import re

def parse_user_agent(user_agent):
    """Витягує інформацію про браузер та ОС із User-Agent."""
    browser_match = re.search(r'(Chrome|Firefox|Safari|Opera|Edg|MSIE)\/[\d.]+', user_agent)
    os_match = re.search(r'\(([^)]+)\)', user_agent)

    browser = browser_match.group(0) if browser_match else "Невідомий браузер"
    os_info = os_match.group(1) if os_match else "Невідома ОС"
    return f"{browser} на {os_info}"
