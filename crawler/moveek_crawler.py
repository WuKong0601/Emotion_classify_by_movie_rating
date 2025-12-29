"""
Moveek Movie Review Crawler
Crawl movie reviews from moveek.com for emotion classification research.

Features:
- Checkpoint saving after each movie (won't lose data if interrupted)
- Resume from last crawled movie
- Progress tracking
"""

import asyncio
import csv
import json
import os
import re
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional

# Suppress Windows asyncio pipe warnings (harmless)
warnings.filterwarnings("ignore", category=ResourceWarning)

from playwright.async_api import async_playwright, Page, Browser
from tqdm import tqdm

from config import BASE_URL, CSS_SELECTORS, SETTINGS, OUTPUT, MONTHLY_MOVIES_URL_TEMPLATE, CRAWL_CONFIG


class MoveekCrawler:
    """Crawler for Moveek movie reviews with checkpoint/resume support."""
    
    def __init__(self, headless: bool = True):
        self.headless = headless
        self.browser: Optional[Browser] = None
        self.playwright = None  # Store playwright instance for cleanup
        self.reviews_data = []
        self.consecutive_errors = 0  # Track consecutive errors for browser restart
        
        # Paths
        self.data_dir = Path(OUTPUT["raw_data_dir"])
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.output_file = self.data_dir / OUTPUT["reviews_file"]
        self.checkpoint_file = self.data_dir / "checkpoint.json"
        self.movies_file = self.data_dir / "movies_list.json"
        
    def load_checkpoint(self) -> dict:
        """Load checkpoint if exists."""
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"crawled_movies": [], "total_reviews": 0}
    
    def save_checkpoint(self, crawled_movies: list, total_reviews: int):
        """Save checkpoint after each movie."""
        checkpoint = {
            "crawled_movies": crawled_movies,
            "total_reviews": total_reviews,
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        with open(self.checkpoint_file, "w", encoding="utf-8") as f:
            json.dump(checkpoint, f, ensure_ascii=False, indent=2)
    
    def append_reviews_to_csv(self, reviews: list[dict]):
        """Append reviews to CSV file (creates if not exists)."""
        if not reviews:
            return
            
        fieldnames = ["movie_title", "username", "rating", "review_text"]
        file_exists = self.output_file.exists()
        
        with open(self.output_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerows(reviews)
    
    async def start_browser(self):
        """Start the Playwright browser."""
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(headless=self.headless)
        self.consecutive_errors = 0  # Reset error counter
        print("✓ Browser started")
    
    async def restart_browser(self):
        """Restart browser after consecutive errors."""
        print("\n⚠️  Restarting browser due to consecutive errors...")
        try:
            await self.close_browser()
        except Exception:
            pass
        await asyncio.sleep(3)  # Wait before restarting
        await self.start_browser()
        print("✓ Browser restarted successfully")
        
    async def close_browser(self):
        """Close the browser and cleanup playwright."""
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()
        print("✓ Browser closed")
            
    async def get_movie_list(self, url: str) -> list[dict]:
        """Get list of movies from a listing page."""
        page = await self.browser.new_page()
        movies = []
        
        try:
            await page.goto(url, timeout=SETTINGS["timeout"])
            await page.wait_for_load_state("networkidle")
            await asyncio.sleep(2)
            
            movies_data = await page.evaluate("""
                () => {
                    const movies = [];
                    const links = document.querySelectorAll('a[href*="/lich-chieu/"]');
                    const seen = new Set();
                    
                    links.forEach(link => {
                        const href = link.getAttribute('href');
                        const match = href.match(/\\/lich-chieu\\/([^\\/]+)\\/?/);
                        if (match && !seen.has(match[1])) {
                            const slug = match[1];
                            const title = link.textContent.trim();
                            if (title && title.length > 1) {
                                seen.add(slug);
                                movies.push({ slug: slug, title: title });
                            }
                        }
                    });
                    return movies;
                }
            """)
            
            for m in movies_data:
                movies.append({
                    "slug": m["slug"],
                    "title": m["title"],
                    "url": f"{BASE_URL}/review/{m['slug']}/"
                })
                
        except Exception as e:
            print(f"Error getting movie list: {e}")
        finally:
            await page.close()
            
        return movies
    
    async def get_all_movies(self) -> list[dict]:
        """Get all movies from now showing and coming soon pages."""
        # Check if we have cached movie list
        if self.movies_file.exists():
            print("\n📽️ Loading cached movie list...")
            with open(self.movies_file, "r", encoding="utf-8") as f:
                movies = json.load(f)
            print(f"  ✓ Loaded {len(movies)} movies from cache")
            return movies
        
        print("\n📽️ Getting movie list from Moveek...")
        
        all_movies = []
        
        # Get movies from "now showing"
        now_showing = await self.get_movie_list(f"{BASE_URL}/dang-chieu/")
        all_movies.extend(now_showing)
        print(f"  ✓ Now showing: {len(now_showing)} movies")
        
        await asyncio.sleep(SETTINGS["delay_between_requests"])
        
        # Get movies from "coming soon"
        coming_soon = await self.get_movie_list(f"{BASE_URL}/sap-chieu/")
        all_movies.extend(coming_soon)
        print(f"  ✓ Coming soon: {len(coming_soon)} movies")
        
        # Remove duplicates
        unique = {m["slug"]: m for m in all_movies}
        movies = list(unique.values())
        
        # Save movie list for resume
        with open(self.movies_file, "w", encoding="utf-8") as f:
            json.dump(movies, f, ensure_ascii=False, indent=2)
        
        print(f"  ✓ Total unique movies: {len(movies)}")
        return movies
    
    async def get_movies_by_month(self, year: int, month: int) -> list[dict]:
        """Get list of movies for a specific month.
        
        Args:
            year: The year (e.g., 2025)
            month: The month (1-12)
            
        Returns:
            List of movie dictionaries with slug, title, and review URL
        """
        url = MONTHLY_MOVIES_URL_TEMPLATE.format(month=month, year=year)
        page = await self.browser.new_page()
        movies = []
        
        try:
            print(f"    Fetching {month:02d}/{year}...", end=" ", flush=True)
            
            # Use shorter timeout and domcontentloaded instead of networkidle
            await page.goto(url, timeout=SETTINGS["timeout"], wait_until="domcontentloaded")
            
            # Wait a bit for dynamic content but don't rely on networkidle
            try:
                await page.wait_for_load_state("networkidle", timeout=5000)
            except Exception:
                # If networkidle times out, just continue - page likely has persistent connections
                pass
            
            await asyncio.sleep(1.5)
            
            # Scroll to load any lazy content
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await asyncio.sleep(0.5)
            
            # Extract movie links using improved selectors based on page analysis
            # Use title attribute which is more reliable than text content
            movies_data = await page.evaluate("""
                () => {
                    const movies = [];
                    const seen = new Set();
                    
                    // Primary: links with title attribute (most reliable)
                    const linksWithTitle = document.querySelectorAll('a[href*="/lich-chieu/"][title], a[href*="/phim/"][title]');
                    linksWithTitle.forEach(link => {
                        const href = link.getAttribute('href') || '';
                        const title = link.getAttribute('title') || '';
                        
                        // Match /lich-chieu/slug or /phim/slug
                        let match = href.match(/\\/lich-chieu\\/([^\\/\\?#]+)/);
                        if (!match) {
                            match = href.match(/\\/phim\\/([^\\/\\?#]+)/);
                        }
                        
                        if (match && title && !seen.has(match[1])) {
                            seen.add(match[1]);
                            movies.push({ slug: match[1], title: title.trim() });
                        }
                    });
                    
                    // Fallback: links without title but inside cards with h4 titles
                    if (movies.length === 0) {
                        const cards = document.querySelectorAll('.card');
                        cards.forEach(card => {
                            const link = card.querySelector('a[href*="/lich-chieu/"], a[href*="/phim/"]');
                            const titleEl = card.querySelector('h4 a, h4, .title');
                            
                            if (link && titleEl) {
                                const href = link.getAttribute('href') || '';
                                let match = href.match(/\\/lich-chieu\\/([^\\/\\?#]+)/);
                                if (!match) {
                                    match = href.match(/\\/phim\\/([^\\/\\?#]+)/);
                                }
                                
                                const title = titleEl.textContent?.trim() || '';
                                if (match && title && !seen.has(match[1])) {
                                    seen.add(match[1]);
                                    movies.push({ slug: match[1], title: title });
                                }
                            }
                        });
                    }
                    
                    return movies;
                }
            """)
            
            for m in movies_data:
                movies.append({
                    "slug": m["slug"],
                    "title": m["title"],
                    "url": f"{BASE_URL}/review/{m['slug']}/",
                    "source_month": f"{month:02d}/{year}"
                })
            
            print(f"found {len(movies)} movies")
            
        except Exception as e:
            print(f"ERROR: {str(e)[:50]}")
        finally:
            await page.close()
            
        return movies
    
    async def get_all_monthly_movies(self, 
                                      years: list[int] = None, 
                                      start_month: int = None, 
                                      end_month: int = None) -> list[dict]:
        """Get all movies from specified months/years.
        
        Args:
            years: List of years to crawl (default from config)
            start_month: Starting month (default from config, goes backwards)
            end_month: Ending month (default from config)
            
        Returns:
            List of unique movies from all specified months
        """
        # Use config defaults if not specified
        if years is None:
            years = CRAWL_CONFIG["years"]
        if start_month is None:
            start_month = CRAWL_CONFIG["start_month"]
        if end_month is None:
            end_month = CRAWL_CONFIG["end_month"]
        
        # Check for cached monthly movie list
        monthly_cache_file = self.data_dir / "movies_list_monthly.json"
        if monthly_cache_file.exists():
            print("\n📽️ Loading cached monthly movie list...")
            with open(monthly_cache_file, "r", encoding="utf-8") as f:
                movies = json.load(f)
            print(f"  ✓ Loaded {len(movies)} movies from cache")
            return movies
        
        print("\n📽️ Getting movies by month from Moveek...")
        print(f"   Years: {years}")
        print(f"   Months: {start_month} → {end_month} (backwards)")
        
        all_movies = []
        
        for year in years:
            print(f"\n  📅 Year {year}:")
            
            # Generate month range (backwards: 12, 11, 10, ... 1)
            if year == years[0]:
                # First year: start from start_month
                months = range(start_month, end_month - 1, -1)
            else:
                # Subsequent years: always 12 → 1
                months = range(12, 0, -1)
            
            for month in months:
                movies = await self.get_movies_by_month(year, month)
                all_movies.extend(movies)
                await asyncio.sleep(SETTINGS["delay_between_requests"])
        
        # Remove duplicates (keep first occurrence with source_month info)
        unique = {}
        for m in all_movies:
            if m["slug"] not in unique:
                unique[m["slug"]] = m
        movies = list(unique.values())
        
        # Save to cache
        with open(monthly_cache_file, "w", encoding="utf-8") as f:
            json.dump(movies, f, ensure_ascii=False, indent=2)
        
        print(f"\n  ✓ Total unique movies across all months: {len(movies)}")
        return movies
    
    async def click_load_more(self, page: Page) -> bool:
        """Click the 'Xem thêm' button if available."""
        try:
            button = page.locator(CSS_SELECTORS["load_more_button"])
            if await button.count() > 0 and await button.is_visible():
                await button.click()
                await asyncio.sleep(1)
                return True
        except Exception:
            pass
        return False
            
    async def crawl_movie_reviews(self, movie: dict, pbar: Optional[tqdm] = None) -> list[dict]:
        """Crawl all reviews for a single movie."""
        page = await self.browser.new_page()
        reviews = []
        
        try:
            url = movie["url"]
            await page.goto(url, timeout=SETTINGS["timeout"])
            await page.wait_for_load_state("networkidle")
            await asyncio.sleep(1)
            
            # Click "Load more" button multiple times
            load_more_clicks = 0
            while load_more_clicks < SETTINGS["max_load_more_clicks"]:
                clicked = await self.click_load_more(page)
                if not clicked:
                    break
                load_more_clicks += 1
                if pbar:
                    pbar.set_postfix_str(f"Loading... ({load_more_clicks})")
            
            # Reveal spoilers
            await page.evaluate("""
                () => {
                    document.querySelectorAll('.btn-reveal-spoiler').forEach(btn => {
                        try { btn.click(); } catch(e) {}
                    });
                }
            """)
            await asyncio.sleep(0.5)
            
            # Extract reviews using JavaScript
            reviews_data = await page.evaluate("""
                () => {
                    const reviews = [];
                    const cards = document.querySelectorAll('.card.article.mb-3');
                    
                    cards.forEach(card => {
                        try {
                            let username = '';
                            const usernameEl = card.querySelector('h4.card-title a');
                            if (usernameEl) username = usernameEl.textContent.trim();
                            
                            let rating = null;
                            const ratingEl = card.querySelector('h4.card-title span');
                            if (ratingEl) {
                                const match = ratingEl.textContent.trim().match(/(\\d+)/);
                                if (match) {
                                    const r = parseInt(match[1]);
                                    if (r >= 1 && r <= 10) rating = r;
                                }
                            }
                            
                            let reviewText = '';
                            const contentEl = card.querySelector('.review-content');
                            if (contentEl) {
                                reviewText = contentEl.textContent.trim();
                            }
                            
                            if (!reviewText) {
                                const bodyEl = card.querySelector('.card-body');
                                if (bodyEl) {
                                    reviewText = bodyEl.innerText
                                        .replace(/\\d+ (ngày|giờ|phút|tháng|năm|tuần) trước/g, '')
                                        .replace(/Thích.*$/gm, '')
                                        .replace(/Không thích.*$/gm, '')
                                        .replace(username, '')
                                        .replace(/^\\s*\\d+\\s*$/gm, '')
                                        .trim();
                                }
                            }
                            
                            if (username || reviewText || rating) {
                                reviews.push({
                                    username: username || 'Anonymous',
                                    rating: rating,
                                    reviewText: reviewText
                                });
                            }
                        } catch(e) {}
                    });
                    
                    return reviews;
                }
            """)
            
            # Convert to our format
            for r in reviews_data:
                reviews.append({
                    "movie_title": movie["title"],
                    "username": r.get("username", "Anonymous"),
                    "rating": r.get("rating"),
                    "review_text": r.get("reviewText", "")
                })
            
        except Exception as e:
            if pbar:
                pbar.set_postfix_str(f"Error: {str(e)[:30]}")
        finally:
            await page.close()
            
        return reviews
        
    async def crawl_all(self, limit: Optional[int] = None, resume: bool = True, monthly: bool = False):
        """Main crawling function with checkpoint support.
        
        Args:
            limit: Maximum number of movies to crawl
            resume: Whether to resume from checkpoint
            monthly: Use monthly crawling mode instead of now-showing/coming-soon
        """
        print("=" * 60)
        print("🎬 MOVEEK MOVIE REVIEW CRAWLER")
        print("=" * 60)
        
        # Load checkpoint
        checkpoint = self.load_checkpoint() if resume else {"crawled_movies": [], "total_reviews": 0}
        crawled_slugs = set(checkpoint["crawled_movies"])
        total_reviews = checkpoint["total_reviews"]
        
        if resume and crawled_slugs:
            print(f"\n📌 Resuming from checkpoint:")
            print(f"   Already crawled: {len(crawled_slugs)} movies")
            print(f"   Reviews collected: {total_reviews}")
        
        await self.start_browser()
        
        try:
            # Get all movies based on mode
            if monthly:
                movies = await self.get_all_monthly_movies()
            else:
                movies = await self.get_all_movies()
            
            # Filter out already crawled movies
            remaining_movies = [m for m in movies if m["slug"] not in crawled_slugs]
            
            if limit:
                remaining_movies = remaining_movies[:limit]
                print(f"\n⚠️  Limited to {limit} movies")
            
            # Show stats
            print(f"\n📊 CRAWL STATISTICS:")
            print(f"   Total movies in database: {len(movies)}")
            print(f"   Already crawled: {len(crawled_slugs)}")
            print(f"   Remaining to crawl: {len(remaining_movies)}")
            
            if not remaining_movies:
                print("\n✅ All movies have been crawled!")
                return []
            
            # Crawl reviews
            print(f"\n🔍 Crawling reviews from {len(remaining_movies)} movies...")
            
            new_reviews_count = 0
            with tqdm(remaining_movies, desc="Crawling", unit="movie") as pbar:
                for movie in pbar:
                    pbar.set_description(f"📽️ {movie['title'][:25]}")
                    
                    # Retry loop for each movie
                    reviews = []
                    for retry in range(SETTINGS["max_retries"]):
                        try:
                            reviews = await self.crawl_movie_reviews(movie, pbar)
                            self.consecutive_errors = 0  # Reset on success
                            break
                        except Exception as e:
                            self.consecutive_errors += 1
                            if retry < SETTINGS["max_retries"] - 1:
                                pbar.set_postfix_str(f"Retry {retry + 1}/{SETTINGS['max_retries']}")
                                await asyncio.sleep(SETTINGS["retry_delay"])
                                
                                # Restart browser if too many consecutive errors
                                if self.consecutive_errors >= SETTINGS["browser_restart_threshold"]:
                                    await self.restart_browser()
                            else:
                                pbar.set_postfix_str(f"Failed: {str(e)[:20]}")
                                print(f"\n⚠️  Skipping {movie['title']} after {SETTINGS['max_retries']} retries")
                    
                    # Save reviews immediately (append to CSV)
                    self.append_reviews_to_csv(reviews)
                    
                    # Update checkpoint (save even if movie failed - mark as crawled to avoid infinite loop)
                    crawled_slugs.add(movie["slug"])
                    total_reviews += len(reviews)
                    new_reviews_count += len(reviews)
                    self.save_checkpoint(list(crawled_slugs), total_reviews)
                    
                    pbar.set_postfix_str(f"Reviews: {new_reviews_count} (Total: {total_reviews})")
                    
                    await asyncio.sleep(SETTINGS["delay_between_movies"])
            
            # Print summary
            print("\n" + "=" * 60)
            print("📊 CRAWLING SUMMARY")
            print("=" * 60)
            print(f"  Movies processed this session: {len(remaining_movies)}")
            print(f"  New reviews collected: {new_reviews_count}")
            print(f"  Total reviews in database: {total_reviews}")
            print(f"  Output file: {self.output_file}")
            print("=" * 60)
            
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted by user!")
            print(f"   Progress saved. {len(crawled_slugs)} movies crawled, {total_reviews} reviews collected.")
            print("   Run again to resume from where you left off.")
        
        except Exception as e:
            print(f"\n\n❌ Unexpected error: {str(e)}")
            print(f"   Progress saved. {len(crawled_slugs)} movies crawled.")
            print("   Run again to resume from where you left off.")
            
        finally:
            await self.close_browser()
            
        return self.reviews_data


async def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Crawl movie reviews from Moveek.com")
    parser.add_argument("--test", action="store_true", help="Run in test mode (limit 3 movies)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of movies to crawl")
    parser.add_argument("--visible", action="store_true", help="Show browser window")
    parser.add_argument("--fresh", action="store_true", help="Start fresh (ignore checkpoint)")
    parser.add_argument("--monthly", action="store_true", help="Use monthly crawling mode (crawl from each month's page)")
    parser.add_argument("--start-year", type=int, default=2025, help="Start year for monthly mode (default: 2025)")
    parser.add_argument("--end-year", type=int, default=2024, help="End year for monthly mode (default: 2024)")
    
    args = parser.parse_args()
    
    # Update config if custom years specified
    if args.monthly:
        from config import CRAWL_CONFIG
        years = []
        for y in range(args.start_year, args.end_year - 1, -1):
            years.append(y)
        CRAWL_CONFIG["years"] = years
        print(f"Monthly mode: crawling years {years}")
    
    limit = args.limit
    if args.test and not limit:
        limit = 3
        
    crawler = MoveekCrawler(headless=not args.visible)
    await crawler.crawl_all(limit=limit, resume=not args.fresh, monthly=args.monthly)


if __name__ == "__main__":
    asyncio.run(main())
