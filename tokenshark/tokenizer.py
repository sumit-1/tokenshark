"""Tokenizers without transformers.

* LLMTokenizer — Hugging Face tokenizer.json (via the `tokenizers` package) or a Meta Llama-3
  tiktoken `tokenizer.model`; chat templates rendered with jinja2 from tokenizer_config.json.
* CLIPTokenizer — pure-Python byte-pair encoder for Stable Diffusion's CLIP text encoder.
"""
from __future__ import annotations

import gzip
import html
import json
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import regex as re


def _tok_str(v):
    if isinstance(v, dict):
        return v.get("content")
    return v


class LLMTokenizer:
    def __init__(self, root: Path):
        root = Path(root)
        self.cfg = json.loads((root / "tokenizer_config.json").read_text(encoding="utf-8")) \
            if (root / "tokenizer_config.json").exists() else {}
        self.kind = None
        if (root / "tokenizer.json").exists():
            from tokenizers import Tokenizer
            self.tk = Tokenizer.from_file(str(root / "tokenizer.json"))
            self.kind = "hf"
            self.vocab_size = self.tk.get_vocab_size(with_added_tokens=True)
        elif (root / "tokenizer.model").exists() and (root / "params.json").exists():
            self._init_tiktoken(root / "tokenizer.model")
        else:
            raise FileNotFoundError(f"No tokenizer.json (or Meta tokenizer.model) in {root}")
        tmpl = self.cfg.get("chat_template")
        if isinstance(tmpl, list):  # [{name, template}]
            tmpl = next((t["template"] for t in tmpl if t.get("name") == "default"), tmpl[0]["template"])
        if tmpl is None and (root / "chat_template.jinja").exists():
            tmpl = (root / "chat_template.jinja").read_text(encoding="utf-8")
        if tmpl is None and self.kind == "tiktoken":
            tmpl = LLAMA3_TEMPLATE
        self.chat_template = tmpl
        self.bos_token = _tok_str(self.cfg.get("bos_token")) or ("<|begin_of_text|>" if self.kind == "tiktoken" else None)
        self.eos_token = _tok_str(self.cfg.get("eos_token")) or ("<|eot_id|>" if self.kind == "tiktoken" else None)
        self.bos_id = self.token_to_id(self.bos_token) if self.bos_token else None
        self.eos_id = self.token_to_id(self.eos_token) if self.eos_token else None

    def _init_tiktoken(self, path: Path):
        import tiktoken
        from tiktoken.load import load_tiktoken_bpe
        ranks = load_tiktoken_bpe(str(path))
        special = ["<|begin_of_text|>", "<|end_of_text|>", "<|reserved_special_token_0|>",
                   "<|reserved_special_token_1|>", "<|finetune_right_pad_id|>", "<|step_id|>",
                   "<|start_header_id|>", "<|end_header_id|>", "<|eom_id|>", "<|eot_id|>", "<|python_tag|>"]
        special += [f"<|reserved_special_token_{i}|>" for i in range(2, 256 - len(special) + 2)]
        pat = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*"
               r"|\s*[\r\n]+|\s+(?!\S)|\s+")
        self.special = {t: len(ranks) + i for i, t in enumerate(special)}
        self.tk = tiktoken.Encoding(name=path.name, pat_str=pat, mergeable_ranks=ranks, special_tokens=self.special)
        self.kind = "tiktoken"
        self.vocab_size = len(ranks) + len(special)

    # ---------------------------------------------------------------- basic ops
    def encode(self, text: str, add_special: bool = False) -> list[int]:
        if not text:
            return [self.bos_id] if add_special and self.bos_id is not None else []
        if self.kind == "hf":
            return self.tk.encode(text, add_special_tokens=add_special).ids
        ids = self.tk.encode(text, allowed_special="all")
        if add_special and self.bos_id is not None and (not ids or ids[0] != self.bos_id):
            ids = [self.bos_id] + ids
        return ids

    def decode(self, ids: list[int], skip_special: bool = False) -> str:
        if self.kind == "hf":
            return self.tk.decode(list(ids), skip_special_tokens=skip_special)
        if skip_special:
            ids = [i for i in ids if i not in self.special.values()]
        return self.tk.decode(list(ids))

    def token_to_id(self, tok: str | None):
        if tok is None:
            return None
        if self.kind == "hf":
            return self.tk.token_to_id(tok)
        return self.special.get(tok)

    @lru_cache(maxsize=200_000)
    def piece(self, i: int) -> str:
        """Readable text of a single token (keeps leading spaces, shows newlines)."""
        if self.kind == "hf":
            s = self.tk.id_to_token(int(i))
            if s is None:
                return "?"
            d = self.tk.decode([int(i)], skip_special_tokens=False)
            s = d if d and "\ufffd" not in d else s.replace("Ġ", " ").replace("▁", " ").replace("Ċ", "\n")
        else:
            s = self.tk.decode_single_token_bytes(int(i)).decode("utf-8", "replace")
        return s.replace("\n", "\\n").replace("\t", "\\t")

    def apply_chat(self, messages: list[dict], add_generation_prompt: bool = True, extra: dict | None = None) -> str:
        if not self.chat_template:
            raise ValueError("This tokenizer has no chat template.")
        from jinja2.sandbox import ImmutableSandboxedEnvironment

        def raise_exception(msg):
            raise ValueError(msg)

        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[_GenerationTag])
        env.globals["raise_exception"] = raise_exception
        env.globals["strftime_now"] = lambda fmt: datetime.now().strftime(fmt)
        env.filters["tojson"] = lambda x, indent=None, ensure_ascii=False, **kw: json.dumps(x, indent=indent, ensure_ascii=ensure_ascii)
        return env.from_string(self.chat_template).render(
            messages=messages, add_generation_prompt=add_generation_prompt,
            bos_token=self.bos_token or "", eos_token=self.eos_token or "", **(extra or {}))


from jinja2.ext import Extension as _Ext   # noqa: E402
from jinja2 import nodes as _nodes        # noqa: E402


class _GenerationTag(_Ext):
    """transformers' {% generation %} … {% endgeneration %} marks the assistant's own text (for training masks).
    For rendering it is a no-op: output the body unchanged."""
    tags = {"generation"}

    def parse(self, parser):
        lineno = next(parser.stream).lineno
        body = parser.parse_statements(("name:endgeneration",), drop_needle=True)
        return _nodes.Scope(body, lineno=lineno)


LLAMA3_TEMPLATE = (
    "{{ bos_token }}{% for message in messages %}<|start_header_id|>{{ message['role'] }}<|end_header_id|>\n\n"
    "{{ message['content'] | trim }}<|eot_id|>{% endfor %}"
    "{% if add_generation_prompt %}<|start_header_id|>assistant<|end_header_id|>\n\n{% endif %}")


# ======================================================================================
# CLIP BPE (port of OpenAI CLIP's simple_tokenizer, reading HF vocab.json + merges.txt)
# ======================================================================================

@lru_cache()
def _bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


def _get_pairs(word):
    return {(a, b) for a, b in zip(word, word[1:])}


class CLIPTokenizer:
    def __init__(self, root: Path):
        root = Path(root)
        self.encoder = json.loads((root / "vocab.json").read_text(encoding="utf-8"))
        self.decoder = {v: k for k, v in self.encoder.items()}
        merges = (root / "merges.txt").read_text(encoding="utf-8").split("\n")
        merges = [tuple(m.split()) for m in merges[1:] if m and not m.startswith("#version") and len(m.split()) == 2]
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        cfg = json.loads((root / "tokenizer_config.json").read_text()) if (root / "tokenizer_config.json").exists() else {}
        self.model_max_length = int(cfg.get("model_max_length", 77)) if int(cfg.get("model_max_length", 77)) < 10_000 else 77
        self.bos, self.eos = "<|startoftext|>", "<|endoftext|>"
        self.bos_id, self.eos_id = self.encoder[self.bos], self.encoder[self.eos]
        pad = _tok_str(cfg.get("pad_token")) or self.eos
        self.pad_id = self.encoder.get(pad, self.eos_id)
        self.byte_encoder = _bytes_to_unicode()
        self.cache = {self.bos: self.bos, self.eos: self.eos}
        self.pat = re.compile(r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""",
                              re.IGNORECASE)

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        pairs = _get_pairs(word)
        if not pairs:
            return token + "</w>"
        while True:
            bigram = min(pairs, key=lambda p: self.bpe_ranks.get(p, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new, i = [], 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new.extend(word[i:j])
                    i = j
                except ValueError:
                    new.extend(word[i:])
                    break
                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new.append(first + second)
                    i += 2
                else:
                    new.append(word[i])
                    i += 1
            word = tuple(new)
            if len(word) == 1:
                break
            pairs = _get_pairs(word)
        out = " ".join(word)
        self.cache[token] = out
        return out

    def encode(self, text: str) -> list[int]:
        text = html.unescape(html.unescape(text))
        text = re.sub(r"\s+", " ", text).strip().lower()
        ids = []
        for tok in re.findall(self.pat, text):
            tok = "".join(self.byte_encoder[b] for b in tok.encode("utf-8"))
            ids.extend(self.encoder[b] for b in self.bpe(tok).split(" ") if b in self.encoder)
        return ids

    def __call__(self, text: str) -> tuple[list[int], list[str]]:
        """Padded ids for the text encoder + readable tokens up to and including end-of-text."""
        n = self.model_max_length
        ids = [self.bos_id] + self.encode(text)[: n - 2] + [self.eos_id]
        shown = [self.decoder[i].replace("</w>", "") for i in ids]
        shown[0], shown[-1] = "<start>", "<end>"
        return ids + [self.pad_id] * (n - len(ids)), shown
