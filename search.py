from pathlib import Path
from indexer import Indexer

from indexer import Doc

IDX = Path("indexer")

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"

FG_GREEN = "\033[92m"
FG_CYAN = "\033[96m"
FG_YELLOW = "\033[93m"
FG_MAGENTA = "\033[95m"
FG_RED = "\033[91m"
FG_BLUE = "\033[94m"
FG_WHITE = "\033[97m"

TYPE_COLORS = {
    "show": FG_CYAN,
    "episode": FG_YELLOW,
    "actor": FG_MAGENTA,
}

def color(text, c):
    return f"{c}{text}{RESET}"

if __name__ == "__main__":
    idf_mode = input(
        "Select IDF mode (classic, smooth, prob, invdf) [classic]: "
    ).strip()
    if not idf_mode:
        idf_mode = "classic"
    idx = Indexer.load(IDX/f"index_{idf_mode}.json")
    print(color(f"Loaded {idx.N} docs from {IDX/f'index_{idf_mode}.json'}", FG_GREEN))

    while True:
        try:
            q = input(color("query> ", FG_GREEN)).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not q:
            break

        hits: list[tuple[float, Doc]] = idx.search(q, top_k=10)

        if not hits:
            print(color("No results.", FG_RED))
        else:
            print()
            for score, doc in hits[:10]:
                t = "show"
                url = doc.url or "(no url)"
                type_color = TYPE_COLORS.get(t, FG_WHITE)
                print(
                    f"{color(f'{score:0.3f}', FG_GREEN)}  "
                    f"{color(t.upper() + ':', type_color)}  "
                    f"{color(BOLD + doc.title + RESET, type_color)}  "
                    f"{color('→', FG_WHITE)} {color(url, FG_WHITE)}"
                )
                print(f"    {color(DIM + doc.description + RESET, FG_WHITE)}")
                print(f"    {color('Keywords:', FG_WHITE)} {color(doc.keywords, DIM)}")
                print(f"    {color('Genres:', FG_WHITE)} {color(doc.genres, DIM)}")
                print(f"    {color('Cast:', FG_WHITE)} {color(doc.cast, DIM)}")
                print(f"    {color('Characters:', FG_WHITE)} {color(doc.characters, DIM)}")
                print(f"    {color('Episodes:', FG_WHITE)}")
                for ep in doc.episodes.splitlines():
                    print(f"      {color(ep, DIM)}")

        print()