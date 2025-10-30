from pathlib import Path
from indexer import Indexer

IDX = Path("indexer/index.json")

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
    idx = Indexer.load(IDX)
    print(color(f"Loaded {idx.N} docs from {IDX}", FG_GREEN))
    print(color("Tip:", FG_BLUE), "filter by type, e.g.",
          color("type:show fantasy", FG_CYAN), ";",
          color("type:actor cranston", FG_MAGENTA), ";",
          color("type:episode fly", FG_YELLOW))

    while True:
        try:
            q = input(color("query> ", FG_GREEN)).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not q:
            break

        hits = idx.search(q, top_k=10)

        if not hits:
            print(color("No results.", FG_RED))
        else:
            print()
            for score, doc in hits[:10]:
                t = getattr(doc, "doc_type", "") or ""
                url = doc.url or "(no url)"
                type_color = TYPE_COLORS.get(t, FG_WHITE)
                print(
                    f"{color(f'{score:0.3f}', FG_GREEN)}  "
                    f"{color(t.upper() + ':', type_color)}  "
                    f"{color(BOLD + doc.title + RESET, type_color)}  "
                    f"{color('→', FG_WHITE)} {color(url, FG_WHITE)}"
                )
                for k, v in (doc.extra or {}).items():
                    if not v:
                        continue
                    print(f"    {color(k + ':', FG_WHITE)} {color(str(v), DIM)}")

        print()