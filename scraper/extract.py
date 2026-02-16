"""HTML to clean text + metadata extraction."""

import json
import re
from pathlib import Path

from bs4 import BeautifulSoup

RAW_DIR = Path("data/raw_html")
EXTRACTED_DIR = Path("data/extracted")


def category_from_url(url: str) -> str:
    """Derive category from URL path."""
    path = url.rstrip("/").split("//")[-1]
    parts = path.split("/")
    if len(parts) >= 2:
        cat = parts[1]
        return cat.replace("-", " ").title()
    return "General"


def extract_page(html: str, url: str) -> dict | None:
    """Extract clean text and metadata from HTML."""
    soup = BeautifulSoup(html, "lxml")

    # Extract title before removing elements
    title_tag = soup.find("h1") or soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else ""

    # Find main content before removing noise
    content_el = (
        soup.find("article")
        or soup.find(class_="entry-content")
        or soup.find("main")
        or soup.find(id="content")
        or soup.body
    )

    if not content_el:
        return None

    # Remove noise elements within content
    for tag in content_el.find_all(["script", "style", "nav", "footer", "aside"]):
        tag.decompose()

    # Extract text, preserving paragraph structure
    paragraphs = []
    for el in content_el.find_all(["p", "h2", "h3", "h4", "li", "td", "blockquote"]):
        text = el.get_text(separator=" ", strip=True)
        if text and len(text) > 10:
            paragraphs.append(text)

    text = "\n\n".join(paragraphs)

    if len(text.strip()) < 50:
        return None

    # Extract PDF links
    pdf_links = []
    for a in content_el.find_all("a", href=True):
        href = a["href"]
        if href.lower().endswith(".pdf"):
            pdf_links.append(href)

    return {
        "url": url,
        "title": title,
        "category": category_from_url(url),
        "text": text,
        "pdf_links": pdf_links,
    }


def extract_all() -> list[dict]:
    """Extract text from all downloaded HTML files."""
    EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    html_files = sorted(RAW_DIR.glob("*.html"))
    print(f"Extracting text from {len(html_files)} HTML files...")

    for html_path in html_files:
        html = html_path.read_text(encoding="utf-8", errors="replace")

        # Reconstruct URL from slug (approximate -- stored in the HTML meta)
        soup = BeautifulSoup(html, "lxml")
        canonical = soup.find("link", rel="canonical")
        url = canonical["href"] if canonical and canonical.get("href") else html_path.stem.replace("_", "/")

        result = extract_page(html, url)
        if result:
            # Save individual JSON
            slug = html_path.stem
            out_path = EXTRACTED_DIR / f"{slug}.json"
            out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
            results.append(result)

    print(f"Extracted {len(results)} pages with content")
    return results


def load_extracted() -> list[dict]:
    """Load all previously extracted JSON files."""
    files = sorted(EXTRACTED_DIR.glob("*.json"))
    results = []
    for f in files:
        data = json.loads(f.read_text())
        results.append(data)
    return results


if __name__ == "__main__":
    extract_all()
