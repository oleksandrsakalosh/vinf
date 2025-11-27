import requests
import time
from collections import deque
from pathlib import Path

API_URL = "https://en.wikipedia.org/w/api.php"

USER_AGENT = "TVIndex/0.1 (oleksandr.sakalosh0@gmail.com)"

ROOT_CATEGORIES = [
    "Category:Television",
    "Category:Film",
    "Category:Animation",
]

OUTPUT_FILE = "extracted/wiki_media_categories.txt"

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": USER_AGENT,
})


def fetch_subcategories(category_title: str) -> list[str]:
    """
    Fetch direct subcategories of a given category using MediaWiki API.
    Returns a list of category titles like 'Category:American television series'.
    """
    subcats: list[str] = []
    cmcontinue = None

    while True:
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": category_title,
            "cmtype": "subcat",  
            "cmlimit": "max",    
            "format": "json",
        }
        if cmcontinue:
            params["cmcontinue"] = cmcontinue

        resp = SESSION.get(API_URL, params=params, timeout=30)

        if resp.status_code == 403:
            print(f"403 Forbidden for {category_title}")
            print("Response headers:", resp.headers)
            print("Response text (truncated):", resp.text[:500])
            resp.raise_for_status()

        resp.raise_for_status()
        data = resp.json()

        for item in data.get("query", {}).get("categorymembers", []):
            title = item.get("title")
            if title and title.startswith("Category:"):
                subcats.append(title)

        cont = data.get("continue")
        if not cont:
            break
        cmcontinue = cont.get("cmcontinue")

    return subcats


def collect_all_subcategories(roots: list[str]) -> set[str]:
    """
    BFS over the category graph starting from the given root categories.
    Returns a set of all categories reachable (including the roots).
    """
    visited: set[str] = set()
    queue: deque[str] = deque()

    # Initialize BFS with roots
    for root in roots:
        visited.add(root)
        queue.append(root)

    while queue:
        current = queue.popleft()
        print(f"Processing {current} ...")

        try:
            subcats = fetch_subcategories(current)
        except Exception as e:
            print(f"  ERROR fetching subcategories for {current}: {e}")
            continue

        for sub in subcats:
            if sub not in visited:
                visited.add(sub)
                queue.append(sub)

        time.sleep(0.1)

    return visited


def main():
    print("Collecting all subcategories for:")
    for root in ROOT_CATEGORIES:
        print(f"  - {root}")

    all_cats = collect_all_subcategories(ROOT_CATEGORIES)

    # Save to file
    out_path = Path(OUTPUT_FILE)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for cat in sorted(all_cats):
            f.write(cat + "\n")

    print(f"\nDone. Saved {len(all_cats)} categories to {out_path.resolve()}")


if __name__ == "__main__":
    main()
