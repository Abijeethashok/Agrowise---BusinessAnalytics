"""
Step 6 test: every page renders and the static files are served.
Run:  python test_pages.py      (no server needs to be running)
"""
from html.parser import HTMLParser

from app import app

client = app.test_client()
results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


class TagBalance(HTMLParser):
    VOID = {"meta", "link", "br", "img", "input", "hr", "source"}

    def __init__(self):
        super().__init__()
        self.stack, self.problems = [], []

    def handle_starttag(self, tag, attrs):
        if tag not in self.VOID:
            self.stack.append(tag)

    def handle_endtag(self, tag):
        if tag in self.VOID:
            return
        if not self.stack or self.stack[-1] != tag:
            self.problems.append(f"unexpected </{tag}> (open: {self.stack[-3:]})")
            if tag in self.stack:
                while self.stack and self.stack.pop() != tag:
                    pass
        else:
            self.stack.pop()


PAGES = {
    "/": ("Choose crops that fit", "Home"),
    "/recommend": ("recommend-form", "Crop Recommendation"),
    "/analytics": ('data-ov="rows_clean"', "Business Analytics"),
    "/about": ('id="references"', "About"),
}

print("=" * 72 + "\nPAGES\n" + "=" * 72)
for path, (marker, title) in PAGES.items():
    r = client.get(path)
    html = r.get_data(as_text=True)
    check(f"GET {path} returns 200", r.status_code == 200)
    check(f"{path} contains its main content", marker in html)
    check(f"{path} links the stylesheet and script",
          "css/style.css" in html and "js/app.js" in html)
    check(f"{path} has no unrendered template tags", "{{" not in html and "{%" not in html)
    tb = TagBalance()
    tb.feed(html)
    check(f"{path} HTML tags are balanced", not tb.problems and not tb.stack,
          "; ".join(tb.problems[:2]) or (f"unclosed: {tb.stack}" if tb.stack else ""))
    check(f"{path} marks its nav link active",
          'class="nav-link active"' in html)

print("\n" + "=" * 72 + "\nSTATIC FILES\n" + "=" * 72)
for path in ["/static/css/style.css", "/static/js/app.js"]:
    r = client.get(path)
    check(f"GET {path} returns 200 and is not empty", r.status_code == 200 and len(r.data) > 1000)
r = client.get("/no-such-page")
check("unknown page returns 404", r.status_code == 404)
r = client.get("/api/health")
check("API still works", r.status_code == 200 and r.get_json()["model_loaded"])

print("\n" + "=" * 72)
print(f"{sum(results)} of {len(results)} checks passed.")
if not all(results):
    print("Some checks FAILED. Fix them before continuing to Step 7.")