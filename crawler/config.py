# Moveek Crawler Configuration

# Base URLs
BASE_URL = "https://moveek.com"
REVIEW_URL = f"{BASE_URL}/review"
MOVIES_NOW_SHOWING_URL = f"{BASE_URL}/dang-chieu/"
MOVIES_COMING_SOON_URL = f"{BASE_URL}/sap-chieu/"

# Monthly movies URL template
# Format: /phim-thang-MM-YYYY/ (e.g., /phim-thang-12-2025/)
MONTHLY_MOVIES_URL_TEMPLATE = f"{BASE_URL}/phim-thang-{{month:02d}}-{{year}}/"

# Crawl range for monthly mode
CRAWL_CONFIG = {
    "years": [2025, 2024],  # Years to crawl (in order, will process first to last)
    "start_month": 12,      # Start from December (will go backwards: 12, 11, 10, ...)
    "end_month": 1,         # End at January
}

# CSS Selectors for review page
CSS_SELECTORS = {
    # Movie list page
    "movie_card": "a[href*='/lich-chieu/']",
    
    # Review page
    "review_container": "div.card.card-sm.article.mb-3",
    "username": "a.is-phone-verification-needed",
    "rating_star": "i.fa-star",
    "timestamp": "small.text-muted, div.text-muted.small",
    "review_content": "div.card-body",
    "spoiler_button": "a.btn-reveal-spoiler",
    "load_more_button": "#btn-view-more, a.btn-view-more",
    
    # Movie info
    "movie_title": "h1.movie-title, h1",
}

# Crawler settings
SETTINGS = {
    "delay_between_requests": 1.5,  # seconds
    "delay_between_movies": 2.0,   # seconds
    "max_load_more_clicks": 50,    # max times to click "Xem thêm"
    "timeout": 30000,              # ms
    "headless": True,              # run browser in headless mode
    # Retry and recovery settings
    "max_retries": 3,              # retries per failed request
    "retry_delay": 5,              # seconds between retries
    "browser_restart_threshold": 10,  # restart browser after N consecutive errors
}

# Output paths
OUTPUT = {
    "raw_data_dir": "data/raw",
    "processed_data_dir": "data/processed",
    "reviews_file": "moveek_reviews.csv",
}
