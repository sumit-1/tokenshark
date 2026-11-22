"""Reasoning Steps (`MultiArith` 2,000+ CoT examples) & Custom Training Datasets extension (`tokenshark.extensions.datasets`).

Registers built-in step-by-step chain-of-thought reasoning datasets (`multiarith_cot` with 2,000 structured
`{question, answer}` examples) and lets users upload custom `.jsonl`, `.csv`, or `.txt` datasets into
`data/datasets/` for use in Tiny LLM and LoRA/Adaptation training.
"""
from __future__ import annotations

import csv
import io
import json
import random
import re
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ...runs import DATA
from ...sdk import (
    DatasetSpec,
    FeaturePlugin,
    get_dataset,
    list_datasets,
    register_dataset,
    register_extension,
)

router = APIRouter()
DATASETS_DIR = DATA / "datasets"

DEFAULT_COT_SYSTEM_PROMPT = "You are a helpful math reasoning assistant. Solve each problem step by step and end with 'Answer: <number>'."

SEED_MULTIARITH_ROWS = [
    {
        "question": "Julia had 12 apples. She gave 4 to her friend and bought 3 packs of 5 apples each. How many apples does she have now?",
        "answer": "Step 1: Julia starts with 12 apples and gives away 4: 12 - 4 = 8.\nStep 2: She buys 3 packs of 5 apples: 3 * 5 = 15.\nStep 3: Add the remaining and new apples: 8 + 15 = 23.\nAnswer: 23",
    },
    {
        "question": "A bookstore sold 8 notebooks in the morning and 7 in the afternoon. If each notebook costs $4, how much money did they make?",
        "answer": "Step 1: Total notebooks sold: 8 + 7 = 15.\nStep 2: Multiply by the price per notebook: 15 * 4 = 60.\nAnswer: 60",
    },
    {
        "question": "Sam baked 36 cookies. He ate 6 of them and shared the rest equally among 5 friends. How many cookies did each friend get?",
        "answer": "Step 1: Subtract the cookies Sam ate: 36 - 6 = 30.\nStep 2: Divide equally among 5 friends: 30 / 5 = 6.\nAnswer: 6",
    },
    {
        "question": "Maya read 14 pages on Monday and 18 pages on Tuesday. Her book has 80 pages. How many pages are left to read?",
        "answer": "Step 1: Add the pages read so far: 14 + 18 = 32.\nStep 2: Subtract from the total pages: 80 - 32 = 48.\nAnswer: 48",
    },
    {
        "question": "There are 4 boxes of pencils, and each box holds 9 pencils. If 11 pencils are given to students, how many remain?",
        "answer": "Step 1: Total pencils in 4 boxes: 4 * 9 = 36.\nStep 2: Subtract the 11 pencils given out: 36 - 11 = 25.\nAnswer: 25",
    },
    {
        "question": "Leo earned $9 per hour for 6 hours of work, then spent $19 on a book. How much money does he have left?",
        "answer": "Step 1: Total earnings for 6 hours: 9 * 6 = 54.\nStep 2: Subtract the cost of the book: 54 - 19 = 35.\nAnswer: 35",
    },
    {
        "question": "A classroom has 5 rows of 6 desks, plus 3 extra desks at the back. If 28 students are seated, how many desks are empty?",
        "answer": "Step 1: Desks in the 5 rows: 5 * 6 = 30.\nStep 2: Total desks including extras: 30 + 3 = 33.\nStep 3: Empty desks: 33 - 28 = 5.\nAnswer: 5",
    },
    {
        "question": "Elena collected 24 seashells on Saturday and 16 on Sunday. She put them into bags of 8 shells each. How many bags did she fill?",
        "answer": "Step 1: Total seashells collected: 24 + 16 = 40.\nStep 2: Divide into bags of 8: 40 / 8 = 5.\nAnswer: 5",
    },
    {
        "question": "A baker made 3 trays of 12 muffins and 2 trays of 8 scones. How many pastries did the baker make in total?",
        "answer": "Step 1: Muffins on 3 trays: 3 * 12 = 36.\nStep 2: Scones on 2 trays: 2 * 8 = 16.\nStep 3: Total pastries: 36 + 16 = 52.\nAnswer: 52",
    },
    {
        "question": "Noah had $50. He bought 3 tickets for $11 each. How much change did he receive?",
        "answer": "Step 1: Cost of 3 tickets: 3 * 11 = 33.\nStep 2: Subtract from $50: 50 - 33 = 17.\nAnswer: 17",
    },
    {
        "question": "A garden has 7 rows of tomato plants with 4 plants in each row. If 5 plants withered and were replaced by 9 new ones, how many plants are there now?",
        "answer": "Step 1: Initial plants in 7 rows: 7 * 4 = 28.\nStep 2: Subtract withered plants: 28 - 5 = 23.\nStep 3: Add new plants: 23 + 9 = 32.\nAnswer: 32",
    },
    {
        "question": "Clara ran 4 miles a day for 5 days, then ran 7 miles on Saturday. What is her total distance for the week?",
        "answer": "Step 1: Distance over 5 days: 4 * 5 = 20.\nStep 2: Add Saturday's run: 20 + 7 = 27.\nAnswer: 27",
    },
]


def _generate_multiarith_cot_rows(total: int = 2000, seed: int = 42) -> list[dict[str, str]]:
    rows = list(SEED_MULTIARITH_ROWS)
    rnd = random.Random(seed)
    names = ["Alice", "Ben", "Clara", "David", "Elena", "Felix", "Grace", "Henry", "Iris", "Jack",
             "Kira", "Leo", "Maya", "Noah", "Olivia", "Paul", "Quinn", "Rosa", "Sam", "Tara"]
    items = ["apples", "marbles", "stickers", "books", "pencils", "cookies", "coins", "cards", "muffins", "shells"]
    containers = ["packs", "boxes", "bags", "trays", "crates", "bundles"]

    while len(rows) < total:
        tpl = len(rows) % 8
        name = rnd.choice(names)
        item = rnd.choice(items)
        box = rnd.choice(containers)

        if tpl == 0:
            a = rnd.randint(10, 45)
            b = rnd.randint(2, a - 2)
            c = rnd.randint(2, 7)
            d = rnd.randint(3, 9)
            r1 = a - b
            r2 = c * d
            ans = r1 + r2
            q = f"{name} had {a} {item}. {name} gave {b} to a friend and bought {c} {box} of {d} {item} each. How many {item} does {name} have now?"
            a_str = f"Step 1: Subtract the {item} given away: {a} - {b} = {r1}.\nStep 2: Multiply the new {box}: {c} * {d} = {r2}.\nStep 3: Add them together: {r1} + {r2} = {ans}.\nAnswer: {ans}"
        elif tpl == 1:
            a = rnd.randint(4, 18)
            b = rnd.randint(3, 16)
            price = rnd.randint(2, 12)
            r1 = a + b
            ans = r1 * price
            q = f"A shop sold {a} {item} in the morning and {b} in the afternoon. If each costs ${price}, how much money did they make?"
            a_str = f"Step 1: Add morning and afternoon sales: {a} + {b} = {r1}.\nStep 2: Multiply by ${price} each: {r1} * {price} = {ans}.\nAnswer: {ans}"
        elif tpl == 2:
            friends = rnd.randint(3, 9)
            each = rnd.randint(4, 12)
            eaten = rnd.randint(2, 14)
            rem = friends * each
            start = rem + eaten
            q = f"{name} had {start} {item}. After setting aside {eaten} of them, {name} shared the rest equally among {friends} friends. How many did each friend get?"
            a_str = f"Step 1: Subtract the {eaten} set aside: {start} - {eaten} = {rem}.\nStep 2: Divide equally among {friends} friends: {rem} / {friends} = {each}.\nAnswer: {each}"
        elif tpl == 3:
            d1 = rnd.randint(8, 28)
            d2 = rnd.randint(9, 28)
            tot = d1 + d2 + rnd.randint(12, 60)
            r1 = d1 + d2
            ans = tot - r1
            q = f"{name} collected {d1} {item} on Monday and {d2} on Tuesday. The goal is {tot} {item}. How many more are needed?"
            a_str = f"Step 1: Add Monday and Tuesday totals: {d1} + {d2} = {r1}.\nStep 2: Subtract from the goal of {tot}: {tot} - {r1} = {ans}.\nAnswer: {ans}"
        elif tpl == 4:
            n_box = rnd.randint(3, 9)
            per_box = rnd.randint(5, 12)
            used = rnd.randint(4, n_box * per_box - 3)
            r1 = n_box * per_box
            ans = r1 - used
            q = f"There are {n_box} {box} of {item}, with {per_box} in each. If {used} {item} are used, how many remain?"
            a_str = f"Step 1: Total {item} in {n_box} {box}: {n_box} * {per_box} = {r1}.\nStep 2: Subtract the {used} used: {r1} - {used} = {ans}.\nAnswer: {ans}"
        elif tpl == 5:
            rate = rnd.randint(8, 22)
            hrs = rnd.randint(3, 8)
            spent = rnd.randint(10, rate * hrs - 5)
            r1 = rate * hrs
            ans = r1 - spent
            q = f"{name} earned ${rate} per hour for {hrs} hours, then spent ${spent}. How much money does {name} have left?"
            a_str = f"Step 1: Earnings for {hrs} hours: {rate} * {hrs} = {r1}.\nStep 2: Subtract amount spent: {r1} - {spent} = {ans}.\nAnswer: {ans}"
        elif tpl == 6:
            c1, p1 = rnd.randint(2, 6), rnd.randint(6, 14)
            c2, p2 = rnd.randint(2, 6), rnd.randint(5, 12)
            r1 = c1 * p1
            r2 = c2 * p2
            ans = r1 + r2
            q = f"{name} packed {c1} {box} of {p1} red {item} and {c2} {box} of {p2} blue {item}. How many {item} are packed in total?"
            a_str = f"Step 1: Red {item}: {c1} * {p1} = {r1}.\nStep 2: Blue {item}: {c2} * {p2} = {r2}.\nStep 3: Total {item}: {r1} + {r2} = {ans}.\nAnswer: {ans}"
        else:
            budget = rnd.randint(50, 120)
            qty = rnd.randint(3, 6)
            cost = rnd.randint(6, (budget - 5) // qty)
            r1 = qty * cost
            ans = budget - r1
            q = f"{name} had ${budget} and bought {qty} {box} for ${cost} each. How much money is left?"
            a_str = f"Step 1: Cost of {qty} {box}: {qty} * {cost} = {r1}.\nStep 2: Subtract from ${budget}: {budget} - {r1} = {ans}.\nAnswer: {ans}"

        rows.append({"question": q, "answer": a_str})
    return rows


MULTIARITH_COT_ROWS = _generate_multiarith_cot_rows(2000, seed=42)
MULTIARITH_COT_EXAMPLES = [f"Question: {r['question']}\n{r['answer']}" for r in MULTIARITH_COT_ROWS]

register_dataset(
    DatasetSpec(
        id="multiarith_cot",
        name="MultiArith Step-by-Step Reasoning (CoT)",
        description="2,000 multi-step arithmetic word problems formatted with explicit Step 1 -> Step 2 -> Answer chain-of-thought reasoning.",
        kind="reasoning",
        examples=MULTIARITH_COT_EXAMPLES,
        rows=MULTIARITH_COT_ROWS,
        system_prompt=DEFAULT_COT_SYSTEM_PROMPT,
        sample_prompt="Question: Julia had 12 apples. She gave 4 to her friend and bought 3 packs of 5 apples each. How many apples does she have now?\nStep 1:",
    )
)


def split_qa_example(text: str) -> dict[str, str]:
    """Split a single training example string into `{question, answer}` columns."""
    s = (text or "").strip()
    if " => " in s:
        q, a = s.split(" => ", 1)
        return {"question": q.strip(), "answer": a.strip()}
    if "\nStep 1:" in s:
        q, a = s.split("\nStep 1:", 1)
        q_clean = re.sub(r"^Question:\s*", "", q.strip(), flags=re.I)
        return {"question": q_clean, "answer": "Step 1:" + a}
    if "Question:" in s and "\nAnswer:" in s:
        q, a = s.split("\nAnswer:", 1)
        q_clean = re.sub(r"^Question:\s*", "", q.strip(), flags=re.I)
        return {"question": q_clean, "answer": "Answer:" + a}
    if "Q:" in s and "A:" in s:
        q, a = s.split("A:", 1)
        q_clean = re.sub(r"^Q:\s*", "", q.strip(), flags=re.I)
        return {"question": q_clean, "answer": a.strip()}
    if " = " in s and s.count(" = ") == 1:
        q, a = s.split(" = ", 1)
        return {"question": q.strip() + " =", "answer": a.strip()}
    if " is " in s:
        q, a = s.rsplit(" is ", 1)
        return {"question": q.strip() + " is", "answer": a.strip()}
    if " comes " in s:
        q, a = s.rsplit(" comes ", 1)
        return {"question": q.strip() + " comes", "answer": a.strip()}
    parts = s.splitlines()
    if len(parts) >= 2:
        return {"question": parts[0].strip(), "answer": "\n".join(parts[1:]).strip()}
    words = s.split()
    if len(words) > 2:
        return {"question": " ".join(words[:-1]), "answer": words[-1]}
    return {"question": s, "answer": ""}


def format_row_for_tiny(row: dict[str, str], system_prompt: str = "") -> str:
    q = (row.get("question") or "").strip()
    a = (row.get("answer") or "").strip()
    sys_line = f"System: {system_prompt.strip()}\n" if (system_prompt or "").strip() else ""
    if a.startswith("Step 1:") or a.startswith("Answer:"):
        return f"{sys_line}Question: {q}\n{a}".strip()
    if a:
        return f"{sys_line}{q} {a}".strip()
    return f"{sys_line}{q}".strip()


def format_row_for_adapt(row: dict[str, str]) -> str:
    q = (row.get("question") or "").strip()
    a = (row.get("answer") or "").strip()
    if a:
        return f"{q} => {a.replace(chr(10), ' ')}"
    return q


def _safe_id(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", (name or "").strip()).strip("_.") or "custom_dataset"


def _parse_dataset_content(raw_text: str, fmt: str = "auto") -> tuple[list[str], list[dict[str, str]]]:
    text = raw_text.strip()
    if not text:
        return [], []
    f = (fmt or "auto").lower()
    lines = [l for l in text.splitlines() if l.strip()]
    if f == "jsonl" or (f == "auto" and lines and lines[0].lstrip().startswith("{")):
        examples, rows = [], []
        for l in lines:
            try:
                obj = json.loads(l)
                if isinstance(obj, dict):
                    if "question" in obj and ("steps" in obj or "cot" in obj or "answer" in obj):
                        q = str(obj.get("question", "")).strip()
                        steps = obj.get("steps") or obj.get("cot") or ""
                        if isinstance(steps, list):
                            steps = "\n".join(f"Step {i + 1}: {s}" for i, s in enumerate(steps))
                        ans = str(obj.get("answer", "")).strip()
                        full_ans = (f"{steps}\nAnswer: {ans}" if steps and ans else (steps or ans)).strip()
                        rows.append({"question": q, "answer": full_ans})
                        examples.append(f"Question: {q}\n{full_ans}".strip())
                    elif ("prompt" in obj or "user" in obj or "input" in obj) and ("completion" in obj or "response" in obj or "assistant" in obj or "output" in obj):
                        q = str(obj.get("prompt") or obj.get("user") or obj.get("input") or "").strip()
                        a = str(obj.get("completion") or obj.get("response") or obj.get("assistant") or obj.get("output") or "").strip()
                        rows.append({"question": q, "answer": a})
                        examples.append(f"Question: {q}\n{a}".strip())
                    elif "text" in obj:
                        s = str(obj["text"]).strip()
                        examples.append(s)
                        rows.append(split_qa_example(s))
                    else:
                        s = json.dumps(obj)
                        examples.append(s)
                        rows.append(split_qa_example(s))
                else:
                    s = str(obj)
                    examples.append(s)
                    rows.append(split_qa_example(s))
            except Exception:
                s = l.strip()
                examples.append(s)
                rows.append(split_qa_example(s))
        return examples, rows

    if f == "csv" or (f == "auto" and "," in lines[0] and len(lines) > 1):
        try:
            reader = csv.DictReader(io.StringIO(text))
            examples, rows = [], []
            for row in reader:
                if "question" in row and "answer" in row:
                    q = str(row["question"]).strip()
                    steps = str(row.get("steps") or row.get("reasoning") or "").strip()
                    ans = str(row["answer"]).strip()
                    full_ans = (f"{steps}\nAnswer: {ans}" if steps else ans).strip()
                    rows.append({"question": q, "answer": full_ans})
                    examples.append(f"Question: {q}\n{full_ans}".strip())
                elif "prompt" in row and ("completion" in row or "answer" in row):
                    q = str(row["prompt"]).strip()
                    a = str(row.get("completion") or row.get("answer") or "").strip()
                    rows.append({"question": q, "answer": a})
                    examples.append(f"Question: {q}\n{a}".strip())
                elif "text" in row:
                    s = str(row["text"]).strip()
                    examples.append(s)
                    rows.append(split_qa_example(s))
                else:
                    s = " | ".join(f"{k}: {v}" for k, v in row.items() if v)
                    examples.append(s)
                    rows.append(split_qa_example(s))
            if examples:
                return examples, rows
        except Exception:
            pass

    raw_items = [b.strip() for b in text.split("\n\n") if b.strip()] if "\n\n" in text else [l.strip() for l in lines if l.strip()]
    rows = [split_qa_example(x) for x in raw_items]
    return raw_items, rows


def _load_saved_datasets():
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    for jf in sorted(DATASETS_DIR.glob("*.json")):
        try:
            d = json.loads(jf.read_text(encoding="utf-8"))
            exs = d.get("examples", [])
            rws = d.get("rows") or [split_qa_example(x) for x in exs]
            register_dataset(
                DatasetSpec(
                    id=d["id"],
                    name=d.get("name", d["id"]),
                    description=d.get("description", "Custom uploaded dataset"),
                    kind=d.get("kind", "custom"),
                    examples=exs,
                    rows=rws,
                    system_prompt=d.get("system_prompt", ""),
                    sample_prompt=d.get("sample_prompt", ""),
                )
            )
        except Exception:
            pass


_load_saved_datasets()


class DatasetUploadReq(BaseModel):
    name: str
    content: str
    format: str = "auto"
    description: str = ""
    system_prompt: str = ""
    sample_prompt: str = ""


@router.get("/api/datasets")
def api_list_datasets():
    return list_datasets()


@router.get("/api/datasets/{ds_id}")
def api_get_dataset(ds_id: str, offset: int = 0, limit: int = 200):
    ds = get_dataset(ds_id)
    if ds is None:
        raise HTTPException(404, f"Unknown dataset '{ds_id}'.")
    rws = ds.rows if ds.rows else [split_qa_example(x) for x in ds.examples]
    total = len(rws)
    off = max(0, min(offset, total))
    lim = max(1, min(limit, 2000))
    return {
        "id": ds.id,
        "name": ds.name,
        "description": ds.description,
        "kind": ds.kind,
        "n_examples": total,
        "offset": off,
        "limit": lim,
        "rows": rws[off: off + lim],
        "examples": ds.examples[off: off + lim],
        "system_prompt": ds.system_prompt,
        "sample_prompt": ds.sample_prompt,
    }


@router.post("/api/datasets/upload")
def api_upload_dataset(req: DatasetUploadReq):
    examples, rows = _parse_dataset_content(req.content, req.format)
    if not examples:
        raise HTTPException(400, "No examples found in the uploaded content.")
    ds_id = _safe_id(req.name)
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    spec = DatasetSpec(
        id=ds_id,
        name=req.name.strip() or ds_id,
        description=req.description.strip() or f"Custom dataset ({len(rows)} examples)",
        kind="custom",
        examples=examples,
        rows=rows,
        system_prompt=req.system_prompt.strip(),
        sample_prompt=req.sample_prompt.strip() or rows[0]["question"],
    )
    register_dataset(spec)
    (DATASETS_DIR / f"{ds_id}.json").write_text(
        json.dumps(
            {
                "id": spec.id,
                "name": spec.name,
                "description": spec.description,
                "kind": spec.kind,
                "examples": spec.examples,
                "rows": spec.rows,
                "system_prompt": spec.system_prompt,
                "sample_prompt": spec.sample_prompt,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "id": spec.id,
        "name": spec.name,
        "description": spec.description,
        "kind": spec.kind,
        "n_examples": len(spec.rows),
        "rows": spec.rows[:100],
        "examples": spec.examples[:100],
        "system_prompt": spec.system_prompt,
        "sample_prompt": spec.sample_prompt,
    }


@register_extension
class DatasetsPlugin(FeaturePlugin):
    id = "datasets"
    name = "Reasoning Steps (MultiArith CoT) & Custom Datasets"
    version = "0.2.0"
    description = "2,000+ step-by-step arithmetic reasoning (MultiArith CoT) problems and custom JSONL/CSV/text dataset upload with Question/Answer split and system prompt."
    families = ("causal_lm",)
    router = router
    ui_tabs = ()
