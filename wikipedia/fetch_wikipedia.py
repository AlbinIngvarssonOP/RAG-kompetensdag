import json
import re
from pathlib import Path
from urllib.parse import unquote, urlencode, urlparse
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parent
DOCUMENTS = ROOT / "documents"


def fetch_article(url):
    parsed = urlparse(url)
    if (
        parsed.scheme != "https"
        or not (parsed.hostname or "").endswith(".wikipedia.org")
        or not parsed.path.startswith("/wiki/")
    ):
        raise ValueError(f"Expected a Wikipedia article URL: {url}")

    params = urlencode({
        "action": "query",
        "format": "json",
        "formatversion": 2,
        "prop": "extracts|info",
        "inprop": "url",
        "explaintext": 1,
        "exsectionformat": "wiki",
        "redirects": 1,
        "titles": unquote(parsed.path.removeprefix("/wiki/")),
    })
    request = Request(
        f"https://{parsed.hostname}/w/api.php?{params}",
        headers={"User-Agent": "RAG-Kompetensdag/1.0 (Wikipedia text downloader)"},
    )
    with urlopen(request, timeout=30) as response:
        data = json.load(response)
    if "error" in data:
        raise ValueError(data["error"]["info"])
    page = data["query"]["pages"][0]
    if not page.get("extract", "").strip():
        raise ValueError(f"No article text found: {url}")

    article = {"title": page["title"], "url": page["fullurl"], "text": page["extract"]}
    title = re.sub(r"[^\w.-]+", "_", page["title"]).strip("._-") or "article"
    language = parsed.hostname.split(".")[0]
    return f"{title}-{language}-{page['pageid']}", article


def main():
    DOCUMENTS.mkdir(exist_ok=True)
    for line in (ROOT / "wikipedia_urls.txt").read_text(encoding="utf-8").splitlines():
        url = line.strip()
        if not url or url.startswith("#"):
            continue
        filename, article = fetch_article(url)
        json_path = DOCUMENTS / f"{filename}.json"
        json_path.write_text(json.dumps(article, ensure_ascii=False, indent=2), encoding="utf-8")

        article = json.loads(json_path.read_text(encoding="utf-8"))
        text = re.sub(
            r"^(={2,6})\s*(.*?)\s*\1\s*$",
            lambda match: f"{'#' * len(match[1])} {match[2]}",
            article["text"],
            flags=re.MULTILINE,
        )
        markdown = f"# {article['title']}\n\nSource: {article['url']}\n\n{text.strip()}\n"
        markdown_path = DOCUMENTS / f"{filename}.md"
        markdown_path.write_text(markdown, encoding="utf-8")
        print(f"Saved: {markdown_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()