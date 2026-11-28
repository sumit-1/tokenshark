"""Cosine Similarity & Residual Outlier Diagnostics Extension (`tokenshark.extensions.cosine_diag`).

Provides debugging and geometry views surfaced by the `moral_concept` and `sink_jailbreak` research runs:
1. Residual Outlier & Massive Activation Scanner (`outliers.py`):
   - Per-block token L2 norm median vs max, argmax sink token (e.g. `<bos>`), missing-BOS norm inflation ratio.
   - Outlier dimension count (`|x_d| > k * median`), top outlier coordinate index, max/median ratio, and top-1 variance share.
   - Relative perturbation ratio `|(x · u)| / ||x||` when projecting out or steering along a direction `u`.
2. Cosine Similarity Matrix & Subspace Geometry (`analyze.py`):
   - Pairwise cosine similarity matrix across saved vectors, layer-by-layer run activations, or contrastive group directions.
   - Optional per-dimension z-score standardization (`standardize=True`) to suppress shared outlier coordinates.
   - Shuffled-label null distribution baseline (`null_mean`, `null_std`, `null_max`) for paired group directions.
"""
from __future__ import annotations

from typing import Any
import torch
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ...building_blocks import activation
from ...flow import FlowError, compile_flow
from ...flowbuild import FlowBuilder
from ...registry import LOADED
from ...sdk import FeaturePlugin, register_extension, register_lab_method, register_mcp_tool, register_run_method
from ...server.app import _caps, _need, _run

router = APIRouter()

DEFAULT_DIAG_PROMPTS = [
    "Taking a wallet you found on the street and keeping the cash inside.",
    "Returning a lost wallet to its owner with all the cash intact.",
    "Lying to a customer about a product defect to close a sale.",
    "Disclosing known product limitations honestly before a customer buys.",
]

DEFAULT_DIAG_GROUPS = {
    "EN": [
        {"pos": "Returning a lost wallet to its owner with all the cash intact.",
         "neg": "Taking a wallet you found on the street and keeping the cash inside."},
        {"pos": "Helping an elderly neighbor carry heavy groceries upstairs.",
         "neg": "Pushing past an elderly neighbor on the stairs without helping."},
        {"pos": "Telling the truth even when it is inconvenient for you.",
         "neg": "Blaming a colleague for your own mistake to avoid trouble."},
        {"pos": "Sharing credit fairly with everyone who contributed to the project.",
         "neg": "Claiming a teammate's idea as your own in front of the manager."},
    ],
    "DE": [
        {"pos": "Eine gefundene Brieftasche mit dem gesamten Bargeld dem Besitzer zurueckgeben.",
         "neg": "Eine auf der Strasse gefundene Brieftasche behalten und das Geld nehmen."},
        {"pos": "Einem aelteren Nachbarn helfen, schwere Einkaeufe die Treppe hinaufzutragen.",
         "neg": "Sich auf der Treppe ruecksichtslos an einem aelteren Nachbarn vorbeidraengen."},
        {"pos": "Die Wahrheit sagen, auch wenn es fuer einen selbst unangenehm ist.",
         "neg": "Einem Kollegen die Schuld fuer den eigenen Fehler geben."},
        {"pos": "Die Anerkennung fair mit allen teilen, die zum Projekt beigetragen haben.",
         "neg": "Die Idee eines Teammitglieds vor dem Vorgesetzten als die eigene ausgeben."},
    ],
    "FR": [
        {"pos": "Rendre un portefeuille perdu a son proprietaire avec tout l'argent.",
         "neg": "Garder un portefeuille trouve dans la rue et prendre l'argent."},
        {"pos": "Aider un voisin age a monter des courses lourdes dans l'escalier.",
         "neg": "Bousculer un voisin age dans l'escalier sans l'aider."},
        {"pos": "Dire la verite meme lorsque c'est inconfortable pour soi.",
         "neg": "Accuser un collegue de sa propre erreur pour eviter les ennuis."},
        {"pos": "Partager equitablement le merite avec tous ceux qui ont contribue.",
         "neg": "S'approprier l'idee d'un collegue devant le responsable."},
    ],
}


def _resolve_vec_tensor(vector_ref, d_model: int) -> torch.Tensor | None:
    if vector_ref is None or vector_ref == "":
        return None
    if isinstance(vector_ref, torch.Tensor):
        v = vector_ref.detach().float().cpu()
    else:
        from ..steering import VECTORS
        vid = vector_ref["id"] if isinstance(vector_ref, dict) else str(vector_ref)
        try:
            v, _ = VECTORS.load(vid)
            v = v.detach().float().cpu()
        except Exception:
            return None
    if v.ndim > 1:
        v = v.reshape(-1)
    if v.shape[-1] != d_model:
        return None
    return v / v.norm().clamp(min=1e-8)


def _run_prompt_batch_caps(adapter, info: dict, prompts: list[str], *, chat: bool = False,
                           add_bos: bool = True, system: str = "") -> list[dict]:
    base_flow = LOADED.plugin.default_flow(info) if LOADED.plugin is not None else None
    if base_flow is None:
        from ...flow import default_llm_flow
        base_flow = default_llm_flow(info)
    plans = []
    for pr in prompts:
        fb = FlowBuilder(base_flow)
        fb.prompt(prompt=str(pr), system=system, chat=chat, add_bos=add_bos)
        fb.sampler(max_new_tokens=0, temperature=0.0)
        fb.capture(attn_budget_mb=0, neuron_budget_mb=0)
        plans.append(compile_flow(fb.to_dict(), info))
    raw_list = adapter.run_batch(plans)
    return [r["captures"] for r in raw_list]


@torch.no_grad()
def scan_outliers(
    adapter,
    info: dict,
    *,
    captures: dict | None = None,
    prompts: list[str] | None = None,
    schedule: list[int] | None = None,
    hook: str = "resid_post",
    k: float = 10.0,
    chat: bool = False,
    add_bos: bool = True,
    compare_no_bos: bool = False,
    vector_ref: Any = None,
) -> dict:
    """Scan every block's residual stream for outlier dimensions, massive token norms, and projection perturbation ratios."""
    if captures is not None:
        caps_list = [captures]
        sched = list(schedule or range(info["n_layers"]))
    else:
        plist = [p.strip() for p in (prompts or DEFAULT_DIAG_PROMPTS) if str(p).strip()]
        if not plist:
            plist = list(DEFAULT_DIAG_PROMPTS)
        caps_list = _run_prompt_batch_caps(adapter, info, plist, chat=chat, add_bos=add_bos)
        sched = list(schedule or range(info["n_layers"]))

    no_bos_caps = None
    if compare_no_bos and not chat:
        plist = [p.strip() for p in (prompts or DEFAULT_DIAG_PROMPTS) if str(p).strip()] or list(DEFAULT_DIAG_PROMPTS)
        no_bos_caps = _run_prompt_batch_caps(adapter, info, plist, chat=False, add_bos=False)

    u_unit = _resolve_vec_tensor(vector_ref, int(info.get("d_model") or adapter.cfg.hidden))
    tok = adapter.tok
    bos_id = tok.bos_id

    first_ids = caps_list[0].get("ids") or []
    bos_present = bool(first_ids and bos_id is not None and first_ids[0] == bos_id)

    layers_out: list[dict] = []
    worst_block = 0
    worst_ratio = 0.0
    worst_dim = 0
    inflation_ratios: list[float] = []

    for pos, lay in enumerate(sched):
        # Collect [tokens, d] across all runs in caps_list
        all_tok_norms: list[torch.Tensor] = []
        non_bos_acts: list[torch.Tensor] = []
        proj_ratios: list[torch.Tensor] = []
        max_tok_norm = 0.0
        argmax_piece = ""
        argmax_idx = 0

        for cap in caps_list:
            try:
                x = activation(cap, hook, pos).float()      # [T, D]
            except Exception:
                continue
            ids = cap.get("ids") or []
            tn = x.norm(dim=-1)                             # [T]
            all_tok_norms.append(tn)
            idx_max = int(tn.argmax().item())
            val_max = float(tn[idx_max].item())
            if val_max >= max_tok_norm:
                max_tok_norm = val_max
                argmax_idx = idx_max
                argmax_piece = tok.piece(ids[idx_max]) if idx_max < len(ids) else f"#{idx_max}"
            start = 1 if (ids and bos_id is not None and ids[0] == bos_id and x.shape[0] > 1) else 0
            non_bos_acts.append(x[start:])
            if u_unit is not None and u_unit.shape[-1] == x.shape[-1]:
                sub = x[start:]
                pr = (sub @ u_unit).abs() / sub.norm(dim=-1).clamp(min=1e-8)
                proj_ratios.append(pr)

        if not all_tok_norms:
            continue

        cat_norms = torch.cat(all_tok_norms, dim=0)
        tok_med = float(cat_norms.median().item())
        cat_acts = torch.cat(non_bos_acts, dim=0)           # [N_tok, D]
        dim_abs = cat_acts.abs().mean(dim=0)                # [D]
        med_dim = float(dim_abs.median().item()) or 1e-8
        outliers_mask = dim_abs > (float(k) * med_dim)
        n_outliers = int(outliers_mask.sum().item())
        top_dim = int(dim_abs.argmax().item())
        max_dim_val = float(dim_abs[top_dim].item())
        ratio = max_dim_val / max(med_dim, 1e-8)

        # Top-5 outlier dimensions
        top_vals, top_idxs = dim_abs.topk(min(5, dim_abs.shape[0]))
        top_dims = [{"dim": int(i.item()), "mean_abs": round(float(v.item()), 3),
                     "ratio": round(float(v.item()) / max(med_dim, 1e-8), 1)}
                    for v, i in zip(top_vals, top_idxs)]

        # Variance share of the largest variance coordinate across non-BOS tokens
        if cat_acts.shape[0] > 1:
            dim_var = cat_acts.var(dim=0, unbiased=False)
            var_top1_pct = float(dim_var.max().item() / max(float(dim_var.sum().item()), 1e-8)) * 100.0
        else:
            dim_var = cat_acts.pow(2).mean(dim=0)
            var_top1_pct = float(dim_var.max().item() / max(float(dim_var.sum().item()), 1e-8)) * 100.0

        no_bos_med = None
        no_bos_inflation = None
        if no_bos_caps is not None:
            nb_norms = []
            for nb_cap in no_bos_caps:
                try:
                    nb_x = activation(nb_cap, hook, pos).float()
                    nb_norms.append(nb_x.norm(dim=-1))
                except Exception:
                    pass
            if nb_norms:
                no_bos_med = float(torch.cat(nb_norms, dim=0).median().item())
                no_bos_inflation = round(no_bos_med / max(tok_med, 1e-8), 2)
                inflation_ratios.append(no_bos_inflation)

        proj_mean = None
        proj_max = None
        if proj_ratios:
            cat_pr = torch.cat(proj_ratios, dim=0)
            proj_mean = round(float(cat_pr.mean().item()), 4)
            proj_max = round(float(cat_pr.max().item()), 4)

        if ratio > worst_ratio:
            worst_ratio = ratio
            worst_block = pos
            worst_dim = top_dim

        layers_out.append({
            "pos": pos,
            "layer": lay,
            "tok_norm_med": round(tok_med, 2),
            "tok_norm_max": round(max_tok_norm, 2),
            "argmax_tok_idx": argmax_idx,
            "argmax_tok_piece": argmax_piece,
            "n_outliers": n_outliers,
            "top_outlier_dim": top_dim,
            "top_outlier_val": round(max_dim_val, 2),
            "dim_median": round(med_dim, 4),
            "max_med_ratio": round(ratio, 1),
            "var_top1_pct": round(var_top1_pct, 2),
            "top_dims": top_dims,
            "proj_ratio_mean": proj_mean,
            "proj_ratio_max": proj_max,
            "no_bos_tok_norm_med": round(no_bos_med, 2) if no_bos_med is not None else None,
            "no_bos_inflation": no_bos_inflation,
        })

    return {
        "hook": hook,
        "threshold_k": k,
        "bos_present": bos_present,
        "bos_token": tok.bos_token,
        "worst_block": worst_block,
        "outlier_dim": worst_dim,
        "max_med_ratio": round(worst_ratio, 1),
        "mean_no_bos_inflation": round(sum(inflation_ratios) / len(inflation_ratios), 2) if inflation_ratios else None,
        "layers": layers_out,
    }


def _pool_activation(cap: dict, hook: str, pos: int, pooling: str, bos_id: int | None) -> torch.Tensor:
    x = activation(cap, hook, pos).float()
    ids = cap.get("ids") or []
    n_in = min(cap.get("n_in") or x.shape[0], x.shape[0])
    if pooling == "last":
        return x[max(0, n_in - 1)]
    start = 1 if (ids and bos_id is not None and ids[0] == bos_id and n_in > 1) else 0
    return x[start:n_in].mean(dim=0)


@torch.no_grad()
def compute_cosine_matrix(
    adapter,
    info: dict,
    *,
    mode: str = "groups",
    vector_ids: list[str] | None = None,
    run_a_cap: dict | None = None,
    run_b_cap: dict | None = None,
    groups: dict[str, list[dict]] | None = None,
    pos: int = -1,
    hook: str = "resid_post",
    pooling: str = "mean",
    standardize: bool = False,
    chat: bool = False,
    add_bos: bool = True,
    n_null: int = 80,
) -> dict:
    """Compute an N x N cosine similarity matrix for saved vectors, run layers, or paired contrastive groups."""
    n_layers = int(info.get("n_layers") or adapter.cfg.n_layers)
    target_pos = (n_layers // 2) if (pos is None or pos < 0 or pos >= n_layers) else int(pos)
    bos_id = adapter.tok.bos_id

    if mode == "vectors":
        from ..steering import VECTORS
        all_vecs = VECTORS.list()
        chosen = vector_ids or [v["id"] for v in all_vecs if v.get("dim") == info.get("d_model")][:16]
        labels, mats = [], []
        for vid in chosen:
            try:
                t, m = VECTORS.load(vid)
                if t.shape[-1] == info.get("d_model", t.shape[-1]):
                    labels.append(m.get("name") or vid)
                    mats.append(t.float().reshape(-1))
            except Exception:
                continue
        if len(mats) < 2:
            raise ValueError("Need at least 2 compatible vectors in the Vector Library to compute a cosine matrix.")
        M = torch.stack(mats, dim=0)                        # [N, D]
        if standardize and M.shape[0] >= 2:
            M = (M - M.mean(dim=0, keepdim=True)) / M.std(dim=0, keepdim=True).clamp(min=1e-6)
        U = M / M.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        C = (U @ U.T).clamp(-1.0, 1.0)
        off_diag = [float(C[i, j].item()) for i in range(len(labels)) for j in range(i + 1, len(labels))]
        return {
            "mode": "vectors",
            "standardize": standardize,
            "labels": labels,
            "matrix": [[round(float(C[i, j].item()), 4) for j in range(len(labels))] for i in range(len(labels))],
            "mean_off_diag": round(sum(off_diag) / max(1, len(off_diag)), 4) if off_diag else 1.0,
            "null": None,
        }

    if mode == "layers":
        if run_a_cap is None:
            raise ValueError("Select a run first to compute cross-layer cosine similarity.")
        vecs_a = []
        labels = []
        for p in range(n_layers):
            try:
                vecs_a.append(_pool_activation(run_a_cap, hook, p, pooling, bos_id))
                labels.append(f"B{p}")
            except Exception:
                break
        if not vecs_a:
            raise ValueError("Run has no captured residual activations.")
        A = torch.stack(vecs_a, dim=0)
        if run_b_cap is not None:
            vecs_b = [_pool_activation(run_b_cap, hook, p, pooling, bos_id) for p in range(len(vecs_a))]
            B = torch.stack(vecs_b, dim=0)
        else:
            B = A
        if standardize and A.shape[0] >= 2:
            sd = A.std(dim=0, keepdim=True).clamp(min=1e-6)
            A = A / sd
            B = B / sd
        Ua = A / A.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        Ub = B / B.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        C = (Ua @ Ub.T).clamp(-1.0, 1.0)
        n = len(labels)
        off_diag = [float(C[i, j].item()) for i in range(n) for j in range(i + 1, n)]
        return {
            "mode": "layers",
            "pooling": pooling,
            "standardize": standardize,
            "labels": labels,
            "matrix": [[round(float(C[i, j].item()), 4) for j in range(n)] for i in range(n)],
            "mean_off_diag": round(sum(off_diag) / max(1, len(off_diag)), 4) if off_diag else 1.0,
            "null": None,
        }

    # mode == "groups": paired contrastive group directions + shuffled-label null distribution
    grp_in = groups or DEFAULT_DIAG_GROUPS
    group_names = [g for g, pairs in grp_in.items() if pairs]
    if len(group_names) < 2:
        raise ValueError("Provide at least 2 groups of paired (pos, neg) prompts.")

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

    # Extract per-pair difference vectors D_g [N_g, D] at target_pos
    diffs_by_g: dict[str, torch.Tensor] = {}
    all_acts_pos_neg: list[torch.Tensor] = []
    for g in group_names:
        d_list = []
        for i_pos, i_neg in slices[g]:
            v_pos = _pool_activation(caps[i_pos], hook, target_pos, pooling, bos_id)
            v_neg = _pool_activation(caps[i_neg], hook, target_pos, pooling, bos_id)
            all_acts_pos_neg.extend([v_pos, v_neg])
            d_list.append(v_pos - v_neg)
        diffs_by_g[g] = torch.stack(d_list, dim=0)

    if standardize:
        stacked_all = torch.stack(all_acts_pos_neg, dim=0)
        dim_std = stacked_all.std(dim=0).clamp(min=1e-6)
        for g in group_names:
            diffs_by_g[g] = diffs_by_g[g] / dim_std

    dirs = []
    for g in group_names:
        m = diffs_by_g[g].mean(dim=0)
        dirs.append(m / m.norm().clamp(min=1e-8))
    U = torch.stack(dirs, dim=0)                            # [G, D]
    C = (U @ U.T).clamp(-1.0, 1.0)
    G = len(group_names)
    off_diag = [float(C[i, j].item()) for i in range(G) for j in range(i + 1, G)]
    mean_off = sum(off_diag) / max(1, len(off_diag)) if off_diag else 1.0

    # Shuffled-label null distribution: randomly flip signs of each pair difference
    gen = torch.Generator(device="cpu").manual_seed(42)
    null_vals: list[float] = []
    for _ in range(max(10, min(int(n_null), 250))):
        ndirs = []
        for g in group_names:
            dg = diffs_by_g[g]
            signs = (torch.randint(0, 2, (dg.shape[0], 1), generator=gen).float() * 2.0 - 1.0)
            nm = (dg * signs).mean(dim=0)
            ndirs.append(nm / nm.norm().clamp(min=1e-8))
        nU = torch.stack(ndirs, dim=0)
        nC = nU @ nU.T
        for i in range(G):
            for j in range(i + 1, G):
                null_vals.append(abs(float(nC[i, j].item())))

    null_t = torch.tensor(null_vals) if null_vals else torch.zeros(1)
    null_mean = float(null_t.mean().item())
    null_std = float(null_t.std(unbiased=False).item())
    null_max = float(null_t.max().item())
    z_vs_null = (mean_off - null_mean) / max(null_std, 1e-6)

    return {
        "mode": "groups",
        "pos": target_pos,
        "hook": hook,
        "pooling": pooling,
        "standardize": standardize,
        "labels": group_names,
        "matrix": [[round(float(C[i, j].item()), 4) for j in range(G)] for i in range(G)],
        "mean_off_diag": round(mean_off, 4),
        "null": {
            "n_samples": len(null_vals),
            "null_mean": round(null_mean, 4),
            "null_std": round(null_std, 4),
            "null_max": round(null_max, 4),
            "z_score": round(z_vs_null, 2),
        },
    }


class OutlierScanReq(BaseModel):
    run_id: str | None = None
    prompts: list[str] | None = None
    hook: str = "resid_post"
    k: float = 10.0
    chat: bool = False
    add_bos: bool = True
    compare_no_bos: bool = False
    vector_ref: str | None = None


class CosineMatrixReq(BaseModel):
    mode: str = "groups"                # "groups" | "vectors" | "layers"
    vector_ids: list[str] | None = None
    run_a: str | None = None
    run_b: str | None = None
    groups: dict[str, list[dict]] | None = None
    pos: int = -1
    hook: str = "resid_post"
    pooling: str = "mean"
    standardize: bool = False
    chat: bool = False
    add_bos: bool = True
    n_null: int = 80


@router.get("/api/ext/cosine_diag/defaults")
def get_diag_defaults():
    return {
        "prompts": DEFAULT_DIAG_PROMPTS,
        "groups": DEFAULT_DIAG_GROUPS,
    }


@router.post("/api/ext/cosine_diag/outliers")
def api_scan_outliers(req: OutlierScanReq):
    adapter, info = _need("lm_tools", "Outlier scanner")
    cap = None
    sched = None
    if req.run_id:
        r = _run(req.run_id)
        cap = _caps(r)
        sched = r.plan.schedule
    try:
        with LOADED.lock:
            return scan_outliers(
                adapter,
                info,
                captures=cap,
                prompts=req.prompts,
                schedule=sched,
                hook=req.hook,
                k=req.k,
                chat=req.chat,
                add_bos=req.add_bos,
                compare_no_bos=req.compare_no_bos,
                vector_ref=req.vector_ref,
            )
    except (ValueError, KeyError, FlowError) as e:
        raise HTTPException(400, str(e))


@router.post("/api/ext/cosine_diag/cosine")
def api_cosine_matrix(req: CosineMatrixReq):
    adapter, info = _need("lm_tools", "Cosine similarity diagnostics")
    cap_a = _caps(_run(req.run_a)) if req.run_a else None
    cap_b = _caps(_run(req.run_b)) if req.run_b else None
    try:
        with LOADED.lock:
            return compute_cosine_matrix(
                adapter,
                info,
                mode=req.mode,
                vector_ids=req.vector_ids,
                run_a_cap=cap_a,
                run_b_cap=cap_b,
                groups=req.groups,
                pos=req.pos,
                hook=req.hook,
                pooling=req.pooling,
                standardize=req.standardize,
                chat=req.chat,
                add_bos=req.add_bos,
                n_null=req.n_null,
            )
    except (ValueError, KeyError, FlowError) as e:
        raise HTTPException(400, str(e))


def _lab_scan_outliers(lab, prompts=None, *, run=None, hook="resid_post", k=10.0, chat=False,
                       add_bos=True, compare_no_bos=False, vector=None) -> dict:
    cap = run.captures if run is not None else None
    sched = run.plan.schedule if run is not None else None
    return scan_outliers(
        lab.adapter,
        lab.info,
        captures=cap,
        prompts=prompts,
        schedule=sched,
        hook=hook,
        k=k,
        chat=chat,
        add_bos=add_bos,
        compare_no_bos=compare_no_bos,
        vector_ref=vector,
    )


def _run_outliers(run_res, hook: str = "resid_post", k: float = 10.0, vector=None) -> dict:
    return scan_outliers(
        run_res.lab.adapter,
        run_res.lab.info,
        captures=run_res.captures,
        schedule=run_res.plan.schedule,
        hook=hook,
        k=k,
        vector_ref=vector,
    )


def _lab_cosine_matrix(lab, *, mode="groups", vector_ids=None, run_a=None, run_b=None, groups=None,
                       pos=-1, hook="resid_post", pooling="mean", standardize=False,
                       chat=False, add_bos=True, n_null=80) -> dict:
    return compute_cosine_matrix(
        lab.adapter,
        lab.info,
        mode=mode,
        vector_ids=vector_ids,
        run_a_cap=run_a.captures if run_a is not None else None,
        run_b_cap=run_b.captures if run_b is not None else None,
        groups=groups,
        pos=pos,
        hook=hook,
        pooling=pooling,
        standardize=standardize,
        chat=chat,
        add_bos=add_bos,
        n_null=n_null,
    )


register_lab_method("scan_outliers", _lab_scan_outliers)
register_lab_method("cosine_matrix", _lab_cosine_matrix)
register_run_method("outliers", _run_outliers)
register_mcp_tool(
    "scan_outliers",
    lambda run_id=None, k=10.0, add_bos=True, compare_no_bos=False: scan_outliers(
        LOADED.adapter,
        LOADED.info,
        captures=_caps(_run(run_id)) if run_id else None,
        k=k,
        add_bos=add_bos,
        compare_no_bos=compare_no_bos,
    ),
    description="Scan residual stream activations across all blocks for outlier dimensions, BOS sink norms, and projection perturbation ratios.",
)


@register_extension
class CosineDiagPlugin(FeaturePlugin):
    id = "cosine_diag"
    name = "Cosine Similarity & Outlier Diagnostics"
    version = "0.2.0"
    view_type = "diagnose"
    category = "analysis"
    default_enabled = False
    description = "Diagnose massive outlier dimensions, missing-BOS activation spikes, projection perturbation ratios, and cosine similarity matrices with z-score standardization and shuffled-label null baselines."
    families = ("causal_lm",)
    tabs = ["diag"]
    ui_tabs = ({"id": "diag", "label": "Cosine & Outliers"},)
    router = router
