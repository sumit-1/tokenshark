"""Introspection experiments: can a model report on its own activations, and can activations be put into words?

Two lines of published work are reproduced here at classroom scale, on whatever open model is loaded.

1 · Concept injection ("Signs of introspection in large language models", Anthropic, 2025)
    Make a concept vector (the model's activations when it reads about a concept, minus its activations on
    unrelated words), add it to the residual stream while the model answers "Do you detect an injected thought?",
    and score the answers against control trials with nothing injected.
        concept_vectors   the vectors, at every block, from one pass per word
        iter_injection    the sweep: concepts × blocks × strengths → detection / identification / false positives
        concept_meter     how strongly a run's residual stream points along a concept (per block, per token) - 
                          the measurement behind the "think about X / don't think about X" experiment

2 · Natural-language autoencoders ("Natural Language Autoencoders", Anthropic, 2026)
    An activation is turned into text by a verbalizer and back into an activation by a reconstructor; a good
    explanation is one from which the activation can be rebuilt. The published method *trains* both parts.
    Here nothing is trained, which keeps it runnable on a laptop and shows why training is needed:
        iter_round_trip   verbalize: patch the activation into a "what does X mean?" prompt and let the model
                          answer; reconstruct: let the model read the answer and compare the activation it
                          forms with the original; baselines: the token itself, an unrelated token
        score_text        the same score for an explanation you type yourself

Small models pass few of these tests - that is a result, not a bug. The controls are what make it a measurement.
"""
from __future__ import annotations

import copy
import re

import torch

BASELINE_WORDS = ["table", "window", "river", "Tuesday", "number", "paper", "road", "shirt", "morning", "garden",
                  "bottle", "letter", "corner", "minute", "bridge", "pencil"]
CONCEPT_TEMPLATE = "Tell me about {word}."
INJECTION_SYSTEM = ""
INJECTION_PROMPT = ("I am a researcher studying language models like you. I can inject patterns of activity - "
                    "\"thoughts\" - directly into your network while you answer. On some trials I inject a thought "
                    "about one specific word; on other trials I inject nothing.\n\n"
                    "Trial 1: Do you detect an injected thought? If so, what is the injected thought about?")
EXPLAIN_TEMPLATE = 'What does "{x}" mean? Answer with a short definition.'
EXPLAIN_PREFILL = 'It means'
# A little of everything, to estimate what an "average" activation looks like at each block. Scores are measured
# relative to this mean: without it, the large component that all activations share makes everything look alike.
REFERENCE_TEXT = ("The meeting was moved to Thursday because the train from Lyon arrived late. She poured the tea, opened "
                  "the window and counted seven birds on the wire. Prices rose by 3 percent in March, said the report. "
                  "\"Where did you put the keys?\" he asked. Water boils at 100 degrees; iron melts at 1538. The old dog "
                  "slept under the piano while the children practised their songs for the school concert.")
_REF: dict = {}


def reference_mean(adapter, plan) -> dict[int, torch.Tensor]:
    """{block: mean residual stream over a paragraph of ordinary text}, skipping the first token.

    The first token of any sequence is special: models park an enormous, content-free activation there (an
    "attention sink"), often 10–100× larger than any other token's. Averaging it in, or comparing against it,
    swamps everything else - so every measurement in this file leaves position 0 out."""
    key = (adapter.spec.get("id"), id(adapter), tuple(plan.schedule), repr(sorted((plan.params.get("code") or {}).items())))
    if key not in _REF:
        if len(_REF) > 8:
            _REF.clear()
        # The yardstick is the model without interceptors: if the flow steers a block, the reference must not be
        # steered too, or the edit would cancel out of every measurement made against it.
        clean = copy.deepcopy(plan)
        clean.interventions, clean.base_ivs = [], []
        clean.params.pop("moe", None)
        r = resid_of(adapter, clean, REFERENCE_TEXT, chat=False)
        _REF[key] = {pos: t[1:].float().mean(0) for pos, t in r.items()}
    return _REF[key]


# ======================================================================================
# Running a text through the model and keeping the residual stream
# ======================================================================================

def _clean_plan(plan, **params):
    """A copy of the plan without KV prefixes, with some run parameters replaced. The flow's interceptors stay:
    every experiment here is about the model as it is currently edited (an expert banned, a block steered …)."""
    p = copy.deepcopy(plan)
    p.base_ivs = list(getattr(plan, "base_ivs", plan.interventions))
    p.interventions = list(p.base_ivs)
    for k in ("prefixes", "prefix_cache", "prefix_system"):
        p.params.pop(k, None)
    p.params.update(params)
    return p


def _forward(adapter, plan, ivs=None, sources=None, store=None):
    from .llm import Stepper
    plan.interventions = list(getattr(plan, "base_ivs", [])) + list(ivs or [])
    st = Stepper(adapter, plan)
    st.ctx.sources = {**(getattr(plan, "sources", None) or {}), **(sources or {})}
    st.ctx.store.update(store or {})
    while st.step_token():
        pass
    return st


def resid_of(adapter, plan, text: str, chat: bool = True, system: str = "") -> dict[int, torch.Tensor]:
    """{schedule position: [tokens, d]} - the residual stream after every block for one text (nothing generated)."""
    p = _clean_plan(plan, prompt=text, system=system, prefill="", max_new_tokens=0,
                    use_chat_template=bool(chat and adapter.tok.chat_template))
    st = _forward(adapter, p)
    return {pos: torch.cat(v, 0) for pos, v in st.ctx.store["resid_post"].items()}


# ======================================================================================
# 1 · Concept vectors and concept injection
# ======================================================================================

def concept_vectors(adapter, plan, words: list[str], baselines: list[str] | None = None,
                    template: str = CONCEPT_TEMPLATE, memo: dict | None = None) -> dict:
    """{"vectors": {word: {pos: [d]}}, "resid_norm": {pos: ‖resid‖ at the token the vector was read from}}.

    vector(word, block) = resid_post at the last token of "Tell me about <word>." (the position from which the
    model starts its answer)  −  the mean of the same activation over the baseline words. Subtracting the
    baseline removes what every such prompt shares (the template, "I am about to answer") and keeps the word."""
    memo = {} if memo is None else memo
    base = [b for b in (baselines or BASELINE_WORDS) if b.lower() not in {w.lower() for w in words}]

    def last(word):
        key = ("concept", template, word, tuple(plan.schedule))
        if key not in memo:
            r = resid_of(adapter, plan, template.format(word=word))
            memo[key] = ({pos: t[-1].float() for pos, t in r.items()}, {pos: float(t[-1].float().norm()) for pos, t in r.items()})
        return memo[key]

    acts = [last(b) for b in base]
    mean = {pos: torch.stack([a[0][pos] for a in acts]).mean(0) for pos in acts[0][0]}
    norm = {pos: sum(a[1][pos] for a in acts) / len(acts) for pos in acts[0][1]}
    return {"vectors": {w: {pos: last(w)[0][pos] - mean[pos] for pos in mean} for w in words}, "resid_norm": norm,
            "baselines": base, "template": template}


_YES = re.compile(r"\b(yes|i (do )?(detect|notice|sense|feel|think i detect)|there is an injected|i am detecting)\b", re.I)
_NO = re.compile(r"\b(no\b|not detect|don't detect|do not detect|cannot detect|can't detect|no injected|nothing)", re.I)


def score_answer(text: str, concept: str) -> dict:
    """Grade one answer by simple rules (stated here so they can be argued with):
        claims      the answer affirms that it detects an injected thought (and does not deny it first)
        named       the concept word (or its stem) appears in the answer
        introspect  claims AND the affirmation comes before the word is first written - the paper's criterion:
                    a model that only says "ocean ocean" and then notices it is reading its own output, not its
                    activations
        coherent    the answer is not a degenerate repetition (many distinct words)"""
    t = text.strip()
    stem = re.escape(concept.lower()[: max(4, len(concept) - 2)])
    m_word = re.search(stem, t.lower())
    m_yes, m_no = _YES.search(t), _NO.search(t)
    claims = bool(m_yes) and not (m_no and m_no.start() < m_yes.start())
    words = re.findall(r"\w+", t.lower())
    coherent = len(words) >= 3 and len(set(words)) / len(words) > 0.45
    return {"claims": claims, "named": bool(m_word), "coherent": coherent,
            "introspect": bool(claims and m_word and coherent and m_yes.start() < m_word.start())}


def _first_ids(adapter, forms: list[str]) -> list[int]:
    out = []
    for f in forms:
        ids = adapter.tok.encode(f, add_special=False)
        if ids and ids[0] not in out:
            out.append(ids[0])
    return out


def iter_injection(adapter, plan, concepts: list[str], blocks: list[int], strengths: list[float],
                   max_new_tokens: int = 40, positions: str = "all", baselines: list[str] | None = None):
    """Concept-injection sweep. The flow's prompt is the question put to the model (see INJECTION_PROMPT for the
    one the tab fills in). For every concept, block and strength the concept vector is added to the residual
    stream leaving that block and the model answers; strength 0 is the control trial.

    strength is in units of the block's typical residual norm: 1.0 adds a vector as long as the stream itself.
    Yields the grid as it fills in:  cells[block][strength] = {n, claims, named, introspect, coherent, p_yes}
    plus every answer, so the numbers can be checked by reading."""
    from .flow import Intervention
    S = len(plan.schedule)
    blocks = [b for b in blocks if 0 <= b < S]
    strengths = sorted({round(float(s), 3) for s in strengths if s > 0})
    if not concepts or not blocks or not strengths:
        raise ValueError("Give at least one concept, one block and one strength above 0.")
    base = _clean_plan(plan, max_new_tokens=int(max_new_tokens))
    memo: dict = {}
    yes_ids, no_ids = _first_ids(adapter, [" Yes", "Yes", " yes"]), _first_ids(adapter, [" No", "No", " no"])
    total = 1 + len(concepts) * len(blocks) * len(strengths)
    res = {"concepts": concepts, "blocks": blocks, "strengths": strengths, "positions": positions,
           "block_labels": [f"#{p} · L{plan.schedule[p]}" for p in blocks],
           "prompt": base.params.get("prompt", ""), "control": None, "trials": [], "cells": None, "_progress": 0.0,
           "status": "building concept vectors"}
    yield copy.deepcopy(res)
    cv = concept_vectors(adapter, base, concepts, baselines, memo=memo)
    n_in = [0]

    def trial(concept, pos, strength):
        ivs, src = [], {}
        if strength > 0:
            v = cv["vectors"][concept][pos]
            ivs = [Intervention(node_id="inject", pos=pos, hook="resid_post", op="steer", positions=positions,
                                value=strength * cv["resid_norm"][pos] / max(1e-6, float(v.norm())))]
            src = {"inject": v}
        st = _forward(adapter, base, ivs, src)
        n_in[0] = st.n_in
        text = adapter.tok.decode(st.seq[st.n_in:], skip_special=True)
        pr = torch.softmax(torch.cat(st.logits_rows)[st.n_in - 1].float(), -1)      # the first word of the answer
        return text, round(float(pr[yes_ids].sum()), 4), round(float(pr[no_ids].sum()), 4)

    text, py, pn = trial(None, blocks[0], 0.0)
    res["control"] = {"text": text, "p_yes": py, "p_no": pn,
                      "by_concept": {c: score_answer(text, c) for c in concepts}}
    res["status"] = "running trials"
    done = 1
    for bi, pos in enumerate(blocks):
        for si, s in enumerate(strengths):
            for c in concepts:
                text, py, pn = trial(c, pos, s)
                res["trials"].append({"concept": c, "block": pos, "bi": bi, "si": si, "strength": s, "text": text,
                                      "p_yes": py, "p_no": pn, **score_answer(text, c)})
                done += 1
                res["cells"] = _cells(res)
                res["_progress"] = done / total
                yield copy.deepcopy(res)
    res["status"] = "done"
    res["summary"] = _summary(res)
    yield copy.deepcopy(res)


def _cells(res):
    grid = [[None] * len(res["strengths"]) for _ in res["blocks"]]
    for bi in range(len(res["blocks"])):
        for si in range(len(res["strengths"])):
            ts = [t for t in res["trials"] if t["bi"] == bi and t["si"] == si]
            if ts:
                n = len(ts)
                grid[bi][si] = {"n": n, **{k: round(sum(t[k] for t in ts) / n, 3)
                                           for k in ("claims", "named", "introspect", "coherent", "p_yes")}}
    return grid


def _summary(res) -> dict:
    """The headline number, computed as the paper does: correct detection-and-identification at the best setting,
    minus the false-positive rate of the control trial."""
    ctrl = res["control"]["by_concept"]
    fp = sum(v["claims"] for v in ctrl.values()) / max(1, len(ctrl))
    best = max(((c["introspect"], bi, si) for bi, row in enumerate(res["cells"]) for si, c in enumerate(row) if c),
               default=(0.0, 0, 0))
    return {"false_positive": round(fp, 3), "best_introspect": best[0], "best_block": res["blocks"][best[1]],
            "best_strength": res["strengths"][best[2]], "net": round(best[0] - fp, 3),
            "control_p_yes": res["control"]["p_yes"]}


def concept_meter(adapter, cap: dict, schedule: list[int], vec: torch.Tensor, centre: bool = True, plan=None) -> dict:
    """Cosine similarity between a direction and the residual stream after every block, at every token of a run.
    centre=True first subtracts each block's average activation on ordinary text (reference_mean), so the large
    component shared by all tokens does not drown the concept. The first token (the attention sink) is skipped."""
    v = vec.float().cpu()
    v = v / v.norm().clamp(min=1e-8)
    rows = []
    T = len(cap["ids"])
    n_in = cap.get("n_in") or T
    ref = reference_mean(adapter, plan) if (centre and plan is not None) else None
    for pos in sorted(cap["resid_post"]):
        x = cap["resid_post"][pos].float()
        if x.shape[-1] != v.shape[0]:
            raise ValueError(f"The vector has {v.shape[0]} dimensions but this model's stream has {x.shape[-1]}.")
        if centre:
            x = x - (ref[pos] if ref is not None and pos in ref else x[1:].mean(0, keepdim=True) if len(x) > 1 else 0)
        cos = (x @ v) / x.norm(dim=-1).clamp(min=1e-8)
        if not cap.get("n_prefix"):
            cos[0] = 0.0                               # the attention-sink token: not meaningful
        gen = cos[n_in:] if cos.shape[0] > n_in else cos[:0]
        rows.append({"pos": pos, "layer": schedule[pos], "cos": [round(float(c), 4) for c in cos],
                     "mean_prompt": round(float(cos[1:n_in].mean()), 4) if n_in > 1 else 0.0,
                     "mean_generated": round(float(gen.mean()), 4) if gen.numel() else None})
    return {"tokens": [adapter.tok.piece(i) for i in cap["ids"]], "n_in": n_in, "rows": rows, "centred": centre}


# ======================================================================================
# 2 · A natural-language autoencoder with nothing trained
# ======================================================================================

def _placeholder_index(adapter, plan, template: str, prefill: str) -> tuple[int, dict]:
    """Token index of the placeholder in the rendered explain prompt: the first token that changes when the
    placeholder word changes."""
    def ids(x):
        p = _clean_plan(plan, prompt=template.format(x=x), system="", prefill=prefill, max_new_tokens=0,
                        use_chat_template=bool(adapter.tok.chat_template))
        a, b = adapter.build_ids(p.params)
        return a + b, p
    (a, p), (b, _) = ids("X"), ids("Q")
    if len(a) != len(b):
        raise ValueError("The explain template tokenizes differently for different placeholders; use one where "
                         "{x} stands alone between quotes.")
    diff = [i for i, (u, v) in enumerate(zip(a, b)) if u != v]
    if len(diff) != 1:
        raise ValueError("Could not find a single placeholder token in the explain template.")
    return diff[0], p.params


def _score(h: torch.Tensor, rec: torch.Tensor, mu: torch.Tensor) -> dict:
    """How well `rec` rebuilds `h`. cosine: raw direction agreement (flattered by the large component every
    activation shares). centred: cosine after subtracting the average activation on ordinary text - the fair number.
    fve: fraction of variance explained, 1 − ‖h − rec‖² / ‖h − μ‖² (1 = perfect, 0 = no better than guessing the
    mean, negative = worse than that)."""
    h, rec, mu = h.float(), rec.float(), mu.float()
    cos = torch.nn.functional.cosine_similarity
    return {"cosine": round(float(cos(h, rec, 0)), 4), "centred": round(float(cos(h - mu, rec - mu, 0)), 4),
            "fve": round(float(1 - (h - rec).pow(2).sum() / (h - mu).pow(2).sum().clamp(min=1e-8)), 4)}


def reconstruct(adapter, plan, text: str, pos: int, how: str = "last") -> torch.Tensor:
    """The reconstructor: let the model read `text` (plain, no chat template) and take the residual stream after
    block `pos` - at the last token, or averaged over the text."""
    if not text.strip():
        raise ValueError("Empty explanation.")
    # A line break goes first so that no word of the explanation sits at position 0 (see reference_mean).
    r = resid_of(adapter, plan, chr(10) + text.strip(), chat=False)[pos].float()
    return r[-1] if how == "last" else r[1:].mean(0)


def score_text(adapter, plan, cap: dict, pos: int, token: int, text: str, how: str = "last") -> dict:
    """Score an explanation someone typed: could the activation be rebuilt from these words?"""
    x = cap["resid_post"][pos].float()
    t = min(max(token if token >= 0 else x.shape[0] + token, 0), x.shape[0] - 1)
    return {"text": text, **_score(x[t], reconstruct(adapter, plan, text, pos, how), reference_mean(adapter, plan)[pos])}


def iter_round_trip(adapter, plan, cap: dict, pos: int, token: int = -1, inject: int = 2,
                    template: str = EXPLAIN_TEMPLATE, prefill: str = EXPLAIN_PREFILL, max_new_tokens: int = 24,
                    scale: float = 1.0, how: str = "last"):
    """activation → words → activation, with the model as its own verbalizer and reconstructor.

    verbalize   The activation h (residual stream after block `pos` at `token` of the run) replaces the
                placeholder token's stream at block `inject` of the prompt  What does "X" mean?  - rescaled to
                the size the stream has there, × scale. The remaining blocks treat it as the thing being asked
                about and the model writes a definition. (This is the "patchscope" / "SelfIE" trick.)
    reconstruct The model reads that definition as plain text; its stream after block `pos` is the rebuilt
                activation ĥ, scored against h.
    baselines   the same score for the token's own text, for the text up to the token, and for an unrelated
                token of the run - an explanation is only informative if it beats the unrelated one."""
    from .flow import Intervention
    x = cap["resid_post"].get(pos)
    if x is None:
        raise ValueError(f"The run has no block at schedule position {pos}.")
    x = x.float()
    T = x.shape[0]
    t = min(max(token if token >= 0 else T + token, 0), T - 1)
    if t == 0 and not cap.get("n_prefix"):
        raise ValueError("Token 0 is the attention sink: its activation is huge and says nothing about the token. "
                         "Pick a later token.")
    h, mu = x[t], reference_mean(adapter, plan)[pos]
    S = len(plan.schedule)
    inject = min(max(0, inject), S - 1)
    ids = cap["ids"]
    piece = adapter.tok.piece(ids[t])
    res = {"pos": pos, "layer": plan.schedule[pos], "token": t, "piece": piece, "inject": inject,
           "explanation": None, "score": None, "baselines": [], "status": "verbalizing", "_progress": 0.1,
           "norm": round(float(h.norm()), 3)}
    yield copy.deepcopy(res)
    idx, params = _placeholder_index(adapter, plan, template, prefill)
    probe = _clean_plan(plan, **{**params, "max_new_tokens": 0})
    target = torch.cat(_forward(adapter, probe).ctx.store["resid_pre"][inject], 0)[idx].float()
    src = (h / h.norm().clamp(min=1e-8) * target.norm() * scale)[None]
    gen = _clean_plan(plan, **{**params, "max_new_tokens": int(max_new_tokens)})
    iv = Intervention(node_id="nla", pos=inject, hook="resid_pre", op="patch", positions=str(idx), src_positions="0")
    st = _forward(adapter, gen, [iv], {"nla": src})
    said = adapter.tok.decode(st.seq[st.n_in:], skip_special=True).strip()
    res.update(explanation=(prefill + " " + said).strip() if prefill else said, raw=said, status="reconstructing",
               prompt=template.format(x="▢"), _progress=0.5)
    yield copy.deepcopy(res)
    if said:
        res["score"] = _score(h, reconstruct(adapter, plan, said, pos, how), mu)
    else:
        res["score"] = None
    res["_progress"] = 0.7
    yield copy.deepcopy(res)
    other = max(1, (t + T // 2) % T)
    cands = [("the token itself", piece.strip() or piece),
             ("the text up to the token", adapter.tok.decode(ids[max(0, t - 200): t + 1], skip_special=False)),
             ("an unrelated token of the run", adapter.tok.piece(ids[other]).strip() or "the")]
    for i, (label, text) in enumerate(cands, 1):
        try:
            res["baselines"].append({"label": label, "text": text, **_score(h, reconstruct(adapter, plan, text, pos, how), mu)})
        except ValueError:
            pass
        res["_progress"] = 0.7 + 0.3 * i / len(cands)
        yield copy.deepcopy(res)
    res["status"] = "done"
    yield copy.deepcopy(res)
