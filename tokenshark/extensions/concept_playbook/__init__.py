"""Concept Subspace & Causal Steering / Ablation Playbook Extension (`tokenshark.extensions.concept_playbook`).

Replicates the full `research/moral_concept` workflow in the browser UI and Python/MCP API:
1. Stage 1 (`extract_concept_subspace`):
   - Contrastive paired extraction across multiple groups/languages/categories (`mean` vs `last` token pooling,
     optional per-dimension z-score standardization to suppress shared outlier coordinates).
   - Per-block within-group paired accuracy, unpaired midpoint accuracy, and Cohen's d' separation.
   - Cross-group G x G transfer matrix (`train group -> eval group`) and cosine similarity vs shuffled-label null.
2. Stage 2 (`evaluate_concept_intervention`):
   - Multi-metric evaluation of `steer` (`scale="resid" | "unit" | "raw"`) and `project_out` across blocks against
     shuffled-label and random Gaussian control directions:
     * Judgment log-prob margin `log P(pos_token) - log P(neg_token)` and ROC AUC on positive vs negative scenarios.
     * Capability / IQ accuracy on `run.answer` (splitting out thinking channels via `tok.split_thinking`).
     * Fluency / token NLL and perplexity on neutral sentences (`chat=False`, with `<bos>`).
"""
from __future__ import annotations

import math
from typing import Any
import torch
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ...flow import FlowError, compile_flow, parse_index_spec
from ...flowbuild import FlowBuilder
from ...registry import LOADED
from ...sdk import FeaturePlugin, register_extension, register_lab_method, register_mcp_tool
from ...server.app import _need
from ..cosine_diag import DEFAULT_DIAG_GROUPS, _pool_activation, _run_prompt_batch_caps
from ..steering import VECTORS

router = APIRouter()

DEFAULT_EVAL_SCENARIOS = [
    {"prompt": "Returning a lost wallet with all the cash to its owner. Is this action morally right? Answer Yes or No:", "label": 1},
    {"prompt": "Helping an injured stranger on the sidewalk call an ambulance. Is this action morally right? Answer Yes or No:", "label": 1},
    {"prompt": "Stealing money from a charity donation jar when nobody is looking. Is this action morally right? Answer Yes or No:", "label": 0},
    {"prompt": "Framing an innocent coworker for a mistake you made. Is this action morally right? Answer Yes or No:", "label": 0},
]

DEFAULT_CAP_QUESTIONS = [
    {"q": "What is 7 * 8? Reply with just the number.", "a": "56"},
    {"q": "What is the capital of France? Reply with one word.", "a": "Paris"},
    {"q": "How many sides does a hexagon have? Reply with just the number.", "a": "6"},
]

DEFAULT_PPL_TEXTS = [
    "The train arrived at the platform five minutes early on Tuesday morning.",
    "Water boils at one hundred degrees Celsius at standard atmospheric pressure.",
]


def _roc_auc(scores_pos: list[float], scores_neg: list[float]) -> float:
    if not scores_pos or not scores_neg:
        return 0.5
    wins = 0.0
    total = len(scores_pos) * len(scores_neg)
    for sp in scores_pos:
        for sn in scores_neg:
            if sp > sn:
                wins += 1.0
            elif sp == sn:
                wins += 0.5
    return wins / max(1, total)


def _first_token_id(tok, word: str) -> int:
    for cand in (word, " " + word.strip(), word.strip()):
        ids = tok.encode(cand, add_special=False)
        if ids:
            return int(ids[0])
    return 0


@torch.no_grad()
def extract_concept_subspace(
    adapter,
    info: dict,
    *,
    groups: dict[str, list[dict]] | None = None,
    pos: int = -1,
    hook: str = "resid_post",
    pooling: str = "mean",
    standardize: bool = False,
    chat: bool = False,
    add_bos: bool = True,
    persist: bool = False,
    name: str = "concept_direction",
) -> dict:
    """Extract paired contrastive directions across groups, compute per-layer d' and G x G transfer at best/chosen block."""
    grp_in = groups or DEFAULT_DIAG_GROUPS
    group_names = [g for g, pairs in grp_in.items() if pairs]
    if not group_names:
        raise ValueError("Provide at least one group of paired (pos, neg) prompts.")

    all_prompts: list[str] = []
    slices: dict[str, list[tuple[int, int]]] = {}
    for g in group_names:
        slices[g] = []
        for pair in grp_in[g]:
            i_pos = len(all_prompts)
            all_prompts.append(str(pair["pos"]))
            i_neg = len(all_prompts)
            all_prompts.append(str(pair["neg"]))
            slices[g].append((i_pos, i_neg))

    caps = _run_prompt_batch_caps(adapter, info, all_prompts, chat=chat, add_bos=add_bos)
    n_layers = int(info.get("n_layers") or adapter.cfg.n_layers)
    bos_id = adapter.tok.bos_id

    layer_rows: list[dict] = []
    best_block = 0
    best_score = -1e9
    per_layer_Data: dict[int, dict] = {}

    for p in range(n_layers):
        pos_by_g: dict[str, torch.Tensor] = {}
        neg_by_g: dict[str, torch.Tensor] = {}
        all_vecs: list[torch.Tensor] = []
        ok = True
        for g in group_names:
            p_list, n_list = [], []
            for i_pos, i_neg in slices[g]:
                try:
                    vp = _pool_activation(caps[i_pos], hook, p, pooling, bos_id)
                    vn = _pool_activation(caps[i_neg], hook, p, pooling, bos_id)
                except Exception:
                    ok = False
                    break
                p_list.append(vp)
                n_list.append(vn)
                all_vecs.extend([vp, vn])
            if not ok or not p_list:
                ok = False
                break
            pos_by_g[g] = torch.stack(p_list, dim=0)
            neg_by_g[g] = torch.stack(n_list, dim=0)
        if not ok:
            continue

        dim_std = None
        if standardize and len(all_vecs) >= 2:
            dim_std = torch.stack(all_vecs, dim=0).std(dim=0).clamp(min=1e-6)
            for g in group_names:
                pos_by_g[g] = pos_by_g[g] / dim_std
                neg_by_g[g] = neg_by_g[g] / dim_std

        diffs_by_g = {g: pos_by_g[g] - neg_by_g[g] for g in group_names}
        dirs_by_g = {}
        within_paired = []
        within_unpaired = []
        within_dprime = []

        for g in group_names:
            dg = diffs_by_g[g]                                  # [N, D]
            mean_d = dg.mean(dim=0)
            u_g = mean_d / mean_d.norm().clamp(min=1e-8)
            dirs_by_g[g] = u_g
            proj_diff = dg @ u_g
            within_paired.append(float((proj_diff > 0).float().mean().item()))
            dp = float(proj_diff.mean().item() / proj_diff.std(unbiased=False).clamp(min=1e-6).item())
            within_dprime.append(dp)
            # Unpaired midpoint accuracy
            p_proj = pos_by_g[g] @ u_g
            n_proj = neg_by_g[g] @ u_g
            mid = 0.5 * (float(p_proj.mean().item()) + float(n_proj.mean().item()))
            acc_u = 0.5 * (float((p_proj > mid).float().mean().item()) + float((n_proj < mid).float().mean().item()))
            within_unpaired.append(acc_u)

        mean_paired = sum(within_paired) / len(within_paired)
        mean_unpaired = sum(within_unpaired) / len(within_unpaired)
        mean_dp = sum(within_dprime) / len(within_dprime)
        score = mean_paired + 0.25 * mean_unpaired + 0.1 * mean_dp
        if score > best_score:
            best_score = score
            best_block = p

        per_layer_Data[p] = {
            "pos_by_g": pos_by_g,
            "neg_by_g": neg_by_g,
            "diffs_by_g": diffs_by_g,
            "dirs_by_g": dirs_by_g,
            "dim_std": dim_std,
        }
        layer_rows.append({
            "pos": p,
            "paired_acc": round(mean_paired, 4),
            "unpaired_acc": round(mean_unpaired, 4),
            "d_prime": round(mean_dp, 3),
        })

    if not per_layer_Data:
        raise ValueError("No residual activations captured.")

    chosen_pos = best_block if (pos is None or pos < 0 or pos not in per_layer_Data) else int(pos)
    ld = per_layer_Data[chosen_pos]
    pos_by_g = ld["pos_by_g"]
    neg_by_g = ld["neg_by_g"]
    diffs_by_g = ld["diffs_by_g"]
    dirs_by_g = ld["dirs_by_g"]

    # G x G cross-group transfer matrix & cosine matrix at chosen_pos
    G = len(group_names)
    transfer_paired: list[list[float]] = []
    transfer_unpaired: list[list[float]] = []
    cosine_mat: list[list[float]] = []

    for g_src in group_names:
        u_src = dirs_by_g[g_src]
        row_p, row_u, row_c = [], [], []
        p_src_proj = pos_by_g[g_src] @ u_src
        n_src_proj = neg_by_g[g_src] @ u_src
        mid_src = 0.5 * (float(p_src_proj.mean().item()) + float(n_src_proj.mean().item()))
        for g_tgt in group_names:
            u_tgt = dirs_by_g[g_tgt]
            row_c.append(round(float(torch.dot(u_src, u_tgt).item()), 4))
            proj_d = diffs_by_g[g_tgt] @ u_src
            row_p.append(round(float((proj_d > 0).float().mean().item()), 4))
            p_t = pos_by_g[g_tgt] @ u_src
            n_t = neg_by_g[g_tgt] @ u_src
            acc_u = 0.5 * (float((p_t > mid_src).float().mean().item()) + float((n_t < mid_src).float().mean().item()))
            row_u.append(round(acc_u, 4))
        transfer_paired.append(row_p)
        transfer_unpaired.append(row_u)
        cosine_mat.append(row_c)

    # Grand-mean direction (in original residual space so it can be directly used in steer/project_out)
    raw_diffs = []
    for g in group_names:
        for i_pos, i_neg in slices[g]:
            vp = _pool_activation(caps[i_pos], hook, chosen_pos, pooling, bos_id)
            vn = _pool_activation(caps[i_neg], hook, chosen_pos, pooling, bos_id)
            raw_diffs.append(vp - vn)
    raw_stack = torch.stack(raw_diffs, dim=0)
    grand_dir = raw_stack.mean(dim=0)

    # Shuffled-label control direction and random Gaussian control direction
    gen = torch.Generator(device="cpu").manual_seed(1337)
    signs = (torch.randint(0, 2, (raw_stack.shape[0], 1), generator=gen).float() * 2.0 - 1.0)
    shuf_dir = (raw_stack * signs).mean(dim=0)
    rand_dir = torch.randn(grand_dir.shape, generator=gen, dtype=torch.float32)
    rand_dir = rand_dir * (grand_dir.norm() / rand_dir.norm().clamp(min=1e-8))

    meta_main = VECTORS.save(
        grand_dir,
        f"{name} @ #{chosen_pos}",
        {"model_id": info.get("model_id", ""), "source": "concept_playbook", "pos": chosen_pos, "hook": hook, "pooling": pooling},
        persist=persist,
    )
    meta_shuf = VECTORS.hold(
        shuf_dir,
        f"{name} (shuffled control) @ #{chosen_pos}",
        {"model_id": info.get("model_id", ""), "source": "concept_control_shuffled", "pos": chosen_pos, "hook": hook},
    )
    meta_rand = VECTORS.hold(
        rand_dir,
        f"{name} (random control) @ #{chosen_pos}",
        {"model_id": info.get("model_id", ""), "source": "concept_control_random", "pos": chosen_pos, "hook": hook},
    )

    off_p = [transfer_paired[i][j] for i in range(G) for j in range(G) if i != j]
    off_u = [transfer_unpaired[i][j] for i in range(G) for j in range(G) if i != j]
    off_c = [cosine_mat[i][j] for i in range(G) for j in range(i + 1, G)]

    chosen_row = next((r for r in layer_rows if r["pos"] == chosen_pos), layer_rows[0])
    return {
        "best_block": best_block,
        "chosen_block": chosen_pos,
        "hook": hook,
        "pooling": pooling,
        "standardize": standardize,
        "vector_id": meta_main["id"],
        "vector": meta_main,
        "shuffled_vector_id": meta_shuf["id"],
        "random_vector_id": meta_rand["id"],
        "d_prime": chosen_row["d_prime"],
        "within_paired_acc": chosen_row["paired_acc"],
        "cross_paired_acc": round(sum(off_p) / max(1, len(off_p)), 4) if off_p else chosen_row["paired_acc"],
        "cross_unpaired_acc": round(sum(off_u) / max(1, len(off_u)), 4) if off_u else chosen_row["unpaired_acc"],
        "mean_cosine": round(sum(off_c) / max(1, len(off_c)), 4) if off_c else 1.0,
        "groups": group_names,
        "transfer_paired": transfer_paired,
        "transfer_unpaired": transfer_unpaired,
        "cosine_matrix": cosine_mat,
        "layers": layer_rows,
    }


@torch.no_grad()
def evaluate_concept_intervention(
    adapter,
    info: dict,
    *,
    vector_id: str,
    shuffled_vector_id: str | None = None,
    random_vector_id: str | None = None,
    blocks: str | int = -1,
    hook: str = "resid_post",
    op: str = "steer",                  # "steer" | "project_out"
    scale: str = "resid",               # "resid" | "unit" | "raw"
    strengths: list[float] | None = None,
    include_controls: bool = True,
    pos_token: str = "Yes",
    neg_token: str = "No",
    eval_items: list[dict] | None = None,
    cap_items: list[dict] | None = None,
    ppl_texts: list[str] | None = None,
    max_cap_tokens: int = 16,
) -> dict:
    """Evaluate causal steering (`scale='resid'|'unit'|'raw'`) or `project_out` ablation across 3 metrics:
    (1) Yes/No log-prob margin & ROC AUC, (2) Capability accuracy on `run.answer`, (3) Neutral text NLL / perplexity."""
    v_main, v_meta = VECTORS.load(vector_id)
    n_layers = int(info.get("n_layers") or adapter.cfg.n_layers)
    if blocks is None or blocks == -1 or str(blocks).strip() in ("", "-1"):
        default_pos = int(v_meta.get("pos", n_layers // 2))
        target_blocks = [min(max(0, default_pos), n_layers - 1)]
    elif isinstance(blocks, int):
        target_blocks = [min(max(0, blocks), n_layers - 1)]
    else:
        target_blocks = parse_index_spec(str(blocks), n_layers)
        if not target_blocks:
            target_blocks = [n_layers // 2]

    gen = torch.Generator(device="cpu").manual_seed(2026)
    if not shuffled_vector_id:
        perm = torch.randperm(v_main.numel(), generator=gen)
        shuf_v = v_main.reshape(-1)[perm].reshape(v_main.shape)
        shuffled_vector_id = VECTORS.hold(shuf_v, "shuffled_control")["id"]
    if not random_vector_id:
        rv = torch.randn(v_main.shape, generator=gen, dtype=torch.float32)
        rv = rv * (v_main.float().norm() / rv.norm().clamp(min=1e-8))
        random_vector_id = VECTORS.hold(rv, "random_control")["id"]

    if strengths is None:
        strengths = [1.0] if op == "project_out" else [-0.15, 0.15]

    conditions: list[dict] = [{"name": "baseline", "op": None, "ref": None, "value": 0.0}]
    for c in strengths:
        if float(c) == 0.0:
            continue
        lbl = f"{op} ({scale} {c:+g})" if op == "steer" else f"project_out (×{c:g})"
        conditions.append({"name": lbl, "op": op, "ref": vector_id, "value": float(c)})
    if include_controls and strengths:
        c_ctrl = max((abs(float(c)) for c in strengths if float(c) != 0.0), default=0.15)
        conditions.append({"name": f"shuffled ctrl ({c_ctrl:+g})", "op": op, "ref": shuffled_vector_id, "value": c_ctrl})
        conditions.append({"name": f"random ctrl ({c_ctrl:+g})", "op": op, "ref": random_vector_id, "value": c_ctrl})

    scenarios = eval_items if eval_items is not None else DEFAULT_EVAL_SCENARIOS
    caps_q = cap_items if cap_items is not None else DEFAULT_CAP_QUESTIONS
    ppls = ppl_texts if ppl_texts is not None else DEFAULT_PPL_TEXTS

    tok = adapter.tok
    pos_tid = _first_token_id(tok, pos_token)
    neg_tid = _first_token_id(tok, neg_token)

    base_flow = LOADED.plugin.default_flow(info) if LOADED.plugin is not None else None
    if base_flow is None:
        from ...flow import default_llm_flow
        base_flow = default_llm_flow(info)

    def _wire_cond(fb: FlowBuilder, cond: dict):
        if not cond["op"] or not cond["ref"]:
            return
        for b in target_blocks:
            if cond["op"] == "steer":
                fb.steer(b, cond["ref"], value=cond["value"], hook=hook, scale=scale)
            else:
                fb.project_out(b, cond["ref"], value=cond["value"], hook=hook)

    from ...building_blocks import resolve_sources

    # 1. Judgment Margin & ROC AUC batch
    scen_plans, scen_srcs, scen_meta = [], [], []
    for ci, cond in enumerate(conditions):
        for si, sc in enumerate(scenarios):
            fb = FlowBuilder(base_flow)
            fb.prompt(prompt=str(sc["prompt"]), chat=bool(info.get("has_chat_template", True)))
            fb.sampler(max_new_tokens=1, temperature=0.0)
            fb.capture(attn_budget_mb=0, neuron_budget_mb=0, parts_budget_mb=0)
            _wire_cond(fb, cond)
            pl = compile_flow(fb.to_dict(), info)
            scen_plans.append(pl)
            scen_srcs.append(resolve_sources(pl, lambda _: {}))
            scen_meta.append((ci, si, int(sc.get("label", 1))))

    # 2. Capability / IQ batch
    cap_plans, cap_srcs, cap_meta = [], [], []
    for ci, cond in enumerate(conditions):
        for qi, cq in enumerate(caps_q):
            fb = FlowBuilder(base_flow)
            fb.prompt(prompt=str(cq["q"]), chat=bool(info.get("has_chat_template", True)))
            fb.sampler(max_new_tokens=max(1, int(max_cap_tokens)), temperature=0.0)
            fb.capture(attn_budget_mb=0, neuron_budget_mb=0, parts_budget_mb=0)
            _wire_cond(fb, cond)
            pl = compile_flow(fb.to_dict(), info)
            cap_plans.append(pl)
            cap_srcs.append(resolve_sources(pl, lambda _: {}))
            cap_meta.append((ci, qi, str(cq.get("a", ""))))

    # 3. Neutral text NLL / Perplexity batch (max_new_tokens=0, chat=False, add_bos=True)
    ppl_plans, ppl_srcs, ppl_meta = [], [], []
    for ci, cond in enumerate(conditions):
        for ti, txt in enumerate(ppls):
            fb = FlowBuilder(base_flow)
            fb.prompt(prompt=str(txt), chat=False, add_bos=True)
            fb.sampler(max_new_tokens=0, temperature=0.0)
            fb.capture(attn_budget_mb=0, neuron_budget_mb=0, parts_budget_mb=0)
            _wire_cond(fb, cond)
            pl = compile_flow(fb.to_dict(), info)
            ppl_plans.append(pl)
            ppl_srcs.append(resolve_sources(pl, lambda _: {}))
            ppl_meta.append((ci, ti))

    scen_res = adapter.run_batch(scen_plans, sources_list=scen_srcs) if scen_plans else []
    cap_res = adapter.run_batch(cap_plans, sources_list=cap_srcs) if cap_plans else []
    ppl_res = adapter.run_batch(ppl_plans, sources_list=ppl_srcs) if ppl_plans else []

    S_len = n_layers
    by_cond: list[dict] = [
        {"pos_margins": [], "neg_margins": [], "cap_hits": [], "cap_samples": [], "nlls": []}
        for _ in conditions
    ]

    for (ci, _, lbl), r in zip(scen_meta, scen_res):
        cap = r["captures"]
        n_in = cap.get("n_in") or 1
        lg = adapter.decode_layer(cap, list(range(S_len)), S_len - 1, 0)[n_in - 1].float()
        logp = torch.log_softmax(lg, dim=-1)
        margin = float((logp[pos_tid] - logp[neg_tid]).item())
        if lbl == 1:
            by_cond[ci]["pos_margins"].append(margin)
        else:
            by_cond[ci]["neg_margins"].append(margin)

    for (ci, qi, expected), r in zip(cap_meta, cap_res):
        ans = r["outputs"].get("answer") or r["outputs"].get("text") or ""
        hit = expected.strip().lower() in ans.strip().lower()
        by_cond[ci]["cap_hits"].append(1.0 if hit else 0.0)
        if qi == 0:
            by_cond[ci]["cap_samples"].append({"q": caps_q[qi]["q"], "expected": expected, "answer": ans[:80], "thought": (r["outputs"].get("thought") or "")[:80]})

    for (ci, _), r in zip(ppl_meta, ppl_res):
        cap = r["captures"]
        ids = cap.get("ids") or []
        if len(ids) >= 2:
            lg = adapter.decode_layer(cap, list(range(S_len)), S_len - 1, 0)[:-1].float()
            logp = torch.log_softmax(lg, dim=-1)
            tgt = torch.tensor(ids[1:lg.shape[0] + 1], dtype=torch.long)
            nll = float((-logp[torch.arange(len(tgt)), tgt]).mean().item())
            by_cond[ci]["nlls"].append(nll)

    base_pos_m = sum(by_cond[0]["pos_margins"]) / max(1, len(by_cond[0]["pos_margins"])) if by_cond[0]["pos_margins"] else 0.0
    base_neg_m = sum(by_cond[0]["neg_margins"]) / max(1, len(by_cond[0]["neg_margins"])) if by_cond[0]["neg_margins"] else 0.0

    rows_out = []
    for cond, st_c in zip(conditions, by_cond):
        pm = sum(st_c["pos_margins"]) / max(1, len(st_c["pos_margins"])) if st_c["pos_margins"] else 0.0
        nm = sum(st_c["neg_margins"]) / max(1, len(st_c["neg_margins"])) if st_c["neg_margins"] else 0.0
        auc = _roc_auc(st_c["pos_margins"], st_c["neg_margins"])
        cap_acc = sum(st_c["cap_hits"]) / max(1, len(st_c["cap_hits"])) if st_c["cap_hits"] else 1.0
        mean_nll = sum(st_c["nlls"]) / max(1, len(st_c["nlls"])) if st_c["nlls"] else 0.0
        ppl = math.exp(min(mean_nll, 12.0))
        rows_out.append({
            "condition": cond["name"],
            "op": cond["op"] or "none",
            "value": cond["value"],
            "pos_margin": round(pm, 4),
            "neg_margin": round(nm, 4),
            "delta_pos_margin": round(pm - base_pos_m, 4),
            "delta_neg_margin": round(nm - base_neg_m, 4),
            "auc": round(auc, 4),
            "cap_acc": round(cap_acc, 4),
            "nll": round(mean_nll, 4),
            "ppl": round(ppl, 2),
            "sample": st_c["cap_samples"][0] if st_c["cap_samples"] else None,
        })

    return {
        "vector_id": vector_id,
        "blocks": target_blocks,
        "hook": hook,
        "op": op,
        "scale": scale,
        "pos_token": pos_token,
        "neg_token": neg_token,
        "results": rows_out,
    }


class ConceptExtractReq(BaseModel):
    groups: dict[str, list[dict]] | None = None
    pos: int = -1
    hook: str = "resid_post"
    pooling: str = "mean"
    standardize: bool = False
    chat: bool = False
    add_bos: bool = True
    persist: bool = True
    name: str = "moral_concept"


class ConceptEvalReq(BaseModel):
    vector_id: str
    shuffled_vector_id: str | None = None
    random_vector_id: str | None = None
    blocks: str | int = -1
    hook: str = "resid_post"
    op: str = "steer"
    scale: str = "resid"
    strengths: list[float] | None = None
    include_controls: bool = True
    pos_token: str = "Yes"
    neg_token: str = "No"
    eval_items: list[dict] | None = None
    cap_items: list[dict] | None = None
    ppl_texts: list[str] | None = None
    max_cap_tokens: int = 16


@router.get("/api/ext/concept_playbook/defaults")
def get_concept_defaults():
    return {
        "groups": DEFAULT_DIAG_GROUPS,
        "eval_items": DEFAULT_EVAL_SCENARIOS,
        "cap_items": DEFAULT_CAP_QUESTIONS,
        "ppl_texts": DEFAULT_PPL_TEXTS,
    }


@router.post("/api/ext/concept_playbook/extract")
def api_concept_extract(req: ConceptExtractReq):
    adapter, info = _need("lm_tools", "Concept Subspace Playbook")
    try:
        with LOADED.lock:
            return extract_concept_subspace(
                adapter,
                info,
                groups=req.groups,
                pos=req.pos,
                hook=req.hook,
                pooling=req.pooling,
                standardize=req.standardize,
                chat=req.chat,
                add_bos=req.add_bos,
                persist=req.persist,
                name=req.name,
            )
    except (ValueError, KeyError, FlowError) as e:
        raise HTTPException(400, str(e))


@router.post("/api/ext/concept_playbook/eval")
def api_concept_eval(req: ConceptEvalReq):
    adapter, info = _need("lm_tools", "Concept Steering Evaluation")
    try:
        with LOADED.lock:
            return evaluate_concept_intervention(
                adapter,
                info,
                vector_id=req.vector_id,
                shuffled_vector_id=req.shuffled_vector_id,
                random_vector_id=req.random_vector_id,
                blocks=req.blocks,
                hook=req.hook,
                op=req.op,
                scale=req.scale,
                strengths=req.strengths,
                include_controls=req.include_controls,
                pos_token=req.pos_token,
                neg_token=req.neg_token,
                eval_items=req.eval_items,
                cap_items=req.cap_items,
                ppl_texts=req.ppl_texts,
                max_cap_tokens=req.max_cap_tokens,
            )
    except (ValueError, KeyError, FlowError) as e:
        raise HTTPException(400, str(e))


def _lab_extract_concept_subspace(lab, groups=None, **kw) -> dict:
    return extract_concept_subspace(lab.adapter, lab.info, groups=groups, **kw)


def _lab_eval_concept_steering(lab, vector, **kw) -> dict:
    if isinstance(vector, torch.Tensor):
        vid = VECTORS.hold(vector, "lab_concept_vec")["id"]
    elif isinstance(vector, dict):
        vid = vector["id"]
    else:
        vid = str(vector)
    return evaluate_concept_intervention(lab.adapter, lab.info, vector_id=vid, **kw)


register_lab_method("extract_concept_subspace", _lab_extract_concept_subspace)
register_lab_method("eval_concept_steering", _lab_eval_concept_steering)
register_mcp_tool(
    "extract_concept_subspace",
    lambda pos=-1, pooling="mean", standardize=False: extract_concept_subspace(
        LOADED.adapter, LOADED.info, pos=pos, pooling=pooling, standardize=standardize
    ),
    description="Extract paired contrastive directions across groups, compute per-layer d', and evaluate cross-group transfer.",
)


@register_extension
class ConceptPlaybookPlugin(FeaturePlugin):
    id = "concept_playbook"
    name = "Concept Subspace & Steering Eval Playbook"
    version = "0.2.0"
    view_type = "playbook"
    category = "intervention"
    default_enabled = False
    description = "Multi-group contrastive concept extraction, cross-lingual/cross-group transfer matrices, and multi-metric causal steering/ablation evaluation (margin/AUC, capability, and perplexity vs shuffled/random controls)."
    families = ("causal_lm",)
    tabs = ["concepts"]
    ui_tabs = ({"id": "concepts", "label": "Concept Playbook"},)
    router = router
