"""Sitemap parser and async page downloader for urbanemissions.info."""

import asyncio
import re
from pathlib import Path
from xml.etree import ElementTree

import httpx

DATA_DIR = Path("data/raw_html")
SITEMAP_URL = "https://urbanemissions.info/page-sitemap.xml"
CONCURRENCY = 5
DELAY_BETWEEN_BATCHES = 1.0


def url_to_slug(url: str) -> str:
    """Convert URL to filesystem-safe slug."""
    slug = url.replace("https://", "").replace("http://", "")
    slug = re.sub(r"[^a-zA-Z0-9]", "_", slug)
    return slug.strip("_")


def parse_sitemap(xml_text: str) -> list[str]:
    """Extract URLs from sitemap XML."""
    root = ElementTree.fromstring(xml_text)
    ns = {"ns": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    urls = [loc.text for loc in root.findall(".//ns:loc", ns) if loc.text]
    return urls


async def download_page(
    client: httpx.AsyncClient, url: str, semaphore: asyncio.Semaphore
) -> tuple[str, bool]:
    """Download a single page, respecting semaphore. Returns (url, success)."""
    slug = url_to_slug(url)
    dest = DATA_DIR / f"{slug}.html"

    if dest.exists():
        return url, True

    async with semaphore:
        try:
            resp = await client.get(url, follow_redirects=True, timeout=30.0)
            resp.raise_for_status()
            dest.write_text(resp.text, encoding="utf-8")
            return url, True
        except Exception as e:
            print(f"  Failed: {url} -- {e}")
            return url, False


async def crawl() -> list[str]:
    """Fetch sitemap, download all pages. Returns list of downloaded URLs."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    async with httpx.AsyncClient(
        headers={"User-Agent": "urbanemissions-rag-bot/0.1 (research)"}
    ) as client:
        # Fetch sitemap
        print("Fetching sitemap...")
        resp = await client.get(SITEMAP_URL, timeout=30.0)
        resp.raise_for_status()
        urls = parse_sitemap(resp.text)
        print(f"Found {len(urls)} pages in sitemap")

        # Download pages in batches
        semaphore = asyncio.Semaphore(CONCURRENCY)
        downloaded = []
        failed = []

        # Process in batches for polite crawling
        batch_size = CONCURRENCY
        for i in range(0, len(urls), batch_size):
            batch = urls[i : i + batch_size]
            tasks = [download_page(client, url, semaphore) for url in batch]
            results = await asyncio.gather(*tasks)

            for url, success in results:
                if success:
                    downloaded.append(url)
                else:
                    failed.append(url)

            done = min(i + batch_size, len(urls))
            print(f"  Progress: {done}/{len(urls)} pages")

            if i + batch_size < len(urls):
                await asyncio.sleep(DELAY_BETWEEN_BATCHES)

        print(f"Downloaded: {len(downloaded)}, Failed: {len(failed)}")
        return downloaded


def run():
    """Entry point for crawling."""
    return asyncio.run(crawl())


if __name__ == "__main__":
    run()
