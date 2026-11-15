"""The course website: every page renders, every cross-link and screenshot it names exists."""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
course = pytest.importorskip("course.app", reason="the course website is not part of this checkout")


def test_every_page_renders_and_links_resolve():
    from fastapi.testclient import TestClient
    from course.content import MODULES
    from course.encyclopedia import ENCYCLOPEDIA_BY_SLUG, ENCYCLOPEDIA_ENTRIES
    c = TestClient(course.course_app)
    urls = (["/", "/primer", "/install", "/encyclopedia"] + [f"/module/{m['slug']}" for m in MODULES]
            + [f"/encyclopedia/{e['slug']}" for e in ENCYCLOPEDIA_ENTRIES])
    html = {}
    for u in urls:
        r = c.get(u)
        assert r.status_code == 200, u
        html[u] = r.text
    assert c.get("/module/nope").status_code == 404 and c.get("/encyclopedia/nope").status_code == 404
    docs = Path(__file__).parent.parent / "docs"
    for u, text in html.items():
        for img in re.findall(r'src="/docs/([\w.-]+)"', text):
            assert (docs / img).exists(), f"{u} shows a missing screenshot: {img}"
        for link in re.findall(r'href="(/(?:module|encyclopedia)/[\w-]+)"', text):
            assert link in html, f"{u} links to a page that does not exist: {link}"
    # a course, not a lab sheet: every module has all its parts
    for m in MODULES:
        for part in ("introduction", "motivation", "theory", "steps", "python_code", "results", "conclusion", "questions", "exercise"):
            assert m.get(part), f"module {m['slug']} has no {part}"
        assert all(s in ENCYCLOPEDIA_BY_SLUG for s in m.get("read_more", []))
        compile(m["python_code"], m["slug"], "exec")                 # the lab script is at least valid Python
    assert [m["number"] for m in MODULES] == list(range(1, len(MODULES) + 1))


def test_text_is_escaped_then_formatted():
    assert str(course.md("`a<b>` **x** *y*")) == "<code>a&lt;b&gt;</code> <strong>x</strong> <em>y</em>"
    assert "<ol><li>one</li><li>two</li></ol>" in str(course.md("intro\n1. one\n2. two"))
