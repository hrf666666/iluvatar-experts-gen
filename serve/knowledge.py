"""serve/knowledge.py — knowledge bases that can be plugged in, swapped and pulled out while the server runs.

A knowledge base (KB) is a set of text chunks with an embedding for each, plus an index that is just the
row-normalized embedding matrix: retrieval is a dot product, so a load is reading one .npy, not building a tree.
The registry hands out *snapshots* — tuples of finished KB objects — and every hot operation (load, unload,
reload, swap) builds the replacement first and swaps it inside one short lock. A request that already holds a
snapshot keeps reading the KB it started with, so nothing half-built is ever searched and a swap never breaks an
in-flight answer. That is the whole "hot-pluggable" promise; everything else is bookkeeping.

Two embedders, same interface: `LexicalEmbedder` (the default: hashed word + character n-grams, deterministic,
nothing to download — retrieval that works on a fresh checkout, and the one the tests run on) and
`LlamaEmbedder` (a llama-server started with --embedding, e.g. the same Iluvatar build serving a small GGUF
embedding model on another card; better recall, one HTTP hop). A KB is searched with the embedder that built it,
so a cache from either one loads back exactly.

Sources are plain files: a directory is walked for .txt/.md/.jsonl/.csv/.log, a single file is read as one of
those. .jsonl lines become one chunk each when they are objects with a "text"/"content" field (their other fields
ride along as metadata), and plain text is split on paragraphs into chunks of about `chunk_size` characters with
`overlap` carried over. Built KBs are cached under --kb-cache (chunks.jsonl + embeddings.npy + a manifest with
the source's fingerprint), so a restart — or the JEV gateway on the same machine — reattaches in the time of an
mmap instead of re-embedding.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import urllib.request
from pathlib import Path
from threading import RLock

import numpy as np

TEXT_SUFFIXES = {".txt", ".md", ".jsonl", ".csv", ".log"}


# --------------------------------------------------------------------------- embedders

class LexicalEmbedder:
    """Bag of hashed n-grams: ASCII word unigrams (lowercased) + CJK character 2- and 3-grams, term counts,
    L2-normalized. Deterministic across runs and machines (blake2b, not Python's hash), which is what makes a
    cache entry reusable. Recall is below a real semantic embedder, but it needs no model, no GPU and no
    download — the retrieval tests and a first look run on it."""

    name = "lexical"
    dim = 2048

    def features(self, text: str) -> dict[str, int]:
        tf: dict[str, int] = {}
        for w in re.findall(r"[0-9a-zA-Z_]+", text):
            k = "w:" + w.lower()
            tf[k] = tf.get(k, 0) + 1
        for run in re.findall(r"[\u4e00-\u9fff]+", text):
            for n in (2, 3):
                for i in range(len(run) - n + 1):
                    k = f"c{n}:" + run[i:i + n]
                    tf[k] = tf.get(k, 0) + 1
        return tf

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for r, text in enumerate(texts):
            for f, count in self.features(text).items():
                h = int.from_bytes(hashlib.blake2b(f.encode(), digest_size=8).digest(), "big")
                out[r, h % self.dim] += count
            n = float(np.linalg.norm(out[r]))
            if n > 0:
                out[r] /= n
        return out


class LlamaEmbedder:
    """Embeddings from a llama-server started with --embedding (the same Iluvatar binary can serve a small GGUF
    embedder on another card). Rows are L2-normalized here so retrieval stays a plain dot product."""

    def __init__(self, base_url: str, model: str | None = None, batch: int = 32, timeout: float = 120.0):
        self.base, self.model, self.batch, self.timeout = base_url.rstrip("/"), model, batch, timeout
        self.name = f"llama:{self.base}"
        self.dim: int | None = None

    def _call(self, texts: list[str]) -> list[list[float]]:
        body = {"input": texts}
        if self.model:
            body["model"] = self.model
        req = urllib.request.Request(f"{self.base}/v1/embeddings", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            data = json.load(r)["data"]
        return [d["embedding"] for d in sorted(data, key=lambda d: d["index"])]

    def embed(self, texts: list[str]) -> np.ndarray:
        rows: list[list[float]] = []
        for i in range(0, len(texts), self.batch):
            rows += self._call(texts[i:i + self.batch])
        m = np.asarray(rows, dtype=np.float32)
        self.dim = m.shape[1]
        n = np.linalg.norm(m, axis=1, keepdims=True)
        n[n == 0] = 1.0
        return m / n


def make_embedder(spec: str):
    """'lexical' (the default) or 'llama:http://host:port' — the --kb-embedder / "kb_embedder" spelling."""
    if spec == "lexical" or not spec:
        return LexicalEmbedder()
    if spec.startswith("llama:"):
        url = spec[len("llama:"):]
        model = None
        if "#" in url:                                  # llama:http://h:p/model-name
            url, model = url.split("#", 1)
        return LlamaEmbedder(url, model)
    raise ValueError(f'kb_embedder: "lexical" or "llama:http://host:port", not {spec!r}')


# --------------------------------------------------------------------------- ingest

def chunk_text(text: str, size: int = 900, overlap: int = 140) -> list[str]:
    """Paragraphs first, then sentences, then a hard cut: chunks near `size` characters that never split a
    paragraph when one fits whole, and always overlap by `overlap` so a fact crossing a boundary is found from
    either side."""
    size, overlap = max(120, int(size)), max(0, min(int(overlap), 120))
    chunks: list[str] = []
    for para in (p.strip() for p in re.split(r"\n\s*\n", text)):
        if not para:
            continue
        if len(para) <= size:
            chunks.append(para)
            continue
        pieces, cur = [], ""
        for sent in re.split(r"(?<=[。！？；!?;])\s+|\n+", para):
            if len(cur) + len(sent) + 1 <= size or not cur:
                cur = (cur + "\n" + sent).strip() if cur else sent
            else:
                pieces.append(cur)
                cur = (cur[-overlap:] + "\n" + sent).strip() if overlap else sent
            while len(cur) > size:                      # one sentence longer than the whole chunk
                pieces.append(cur[:size])
                cur = (cur[size - overlap:] if overlap else cur[size:]).strip()
        if cur:
            pieces.append(cur)
        chunks += pieces
    return chunks


def ingest(source: Path) -> list[dict]:
    """-> chunks [{"text", "source"}]; a directory is walked (TEXT_SUFFIXES, sorted for a stable order), a single
    file is read as one. .jsonl lines that parse to an object with a "text"/"content" field are one chunk each
    with the object's other fields kept as meta; anything else is plain text."""
    files = sorted(p for p in (source.rglob("*") if source.is_dir() else [source])
                   if p.is_file() and p.suffix.lower() in TEXT_SUFFIXES)
    if not files and source.is_dir():
        raise ValueError(f"no .txt/.md/.jsonl/.csv/.log files under {source}")
    chunks: list[dict] = []
    for p in files:
        if p.suffix.lower() == ".jsonl":
            for i, line in enumerate(p.read_text(encoding="utf-8", errors="replace").splitlines()):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except ValueError:
                    obj = None
                if isinstance(obj, dict) and isinstance(obj.get("text") or obj.get("content"), str):
                    meta = {k: v for k, v in obj.items() if k not in ("text", "content")}
                    chunks.append({"text": obj.get("text") or obj.get("content"),
                                   "source": f"{p.name}:{i + 1}", **meta})
                else:
                    chunks.append({"text": line, "source": p.name})
        else:
            text = p.read_text(encoding="utf-8", errors="replace")
            chunks += [{"text": c, "source": p.name} for c in chunk_text(text)]
    return chunks


def fingerprint(source: Path) -> str:
    """The source's identity for the cache: path + size + mtime of every file, hashed. A touch rebuilds; the
    same tree reattaches."""
    files = sorted(p for p in (source.rglob("*") if source.is_dir() else [source])
                   if p.is_file() and p.suffix.lower() in TEXT_SUFFIXES)
    h = hashlib.sha256()
    for p in files:
        st = p.stat()
        h.update(f"{p}|{st.st_size}|{st.st_mtime_ns}\n".encode())
    return h.hexdigest()[:16]


# --------------------------------------------------------------------------- the KB

class KnowledgeBase:
    """Immutable once built: chunks, their normalized embeddings and the embedder that made them. Loading or
    reloading builds a *new* one; nothing ever mutates a KB that requests can already see."""

    def __init__(self, name: str, chunks: list[dict], matrix: np.ndarray, embedder, source: str, fp: str,
                 built_at: float):
        self.name, self.chunks, self.matrix, self.embedder = name, chunks, matrix, embedder
        self.source, self.fingerprint, self.built_at = source, fp, built_at

    @classmethod
    def build(cls, name: str, source: Path, embedder, chunk_size: int = 900, overlap: int = 140,
              cache: Path | None = None) -> "KnowledgeBase":
        """From files (or the cache, when the fingerprint still matches). The cache write is atomic
        (tmp + rename), so a killed build leaves either the old cache or no cache, never a half one."""
        fp = fingerprint(source)
        if cache is not None:
            hit = cls._from_cache(name, cache, fp, embedder)
            if hit is not None:
                return hit
        chunks = ingest(source)
        if not chunks:
            raise ValueError(f"{source}: no text found")
        matrix = embedder.embed([c["text"] for c in chunks])
        kb = cls(name, chunks, matrix, embedder, str(source), fp, time.time())
        if cache is not None:
            kb._to_cache(cache)
        return kb

    @classmethod
    def _from_cache(cls, name: str, cache: Path, fp: str, embedder) -> "KnowledgeBase | None":
        d = cache / name
        try:
            manifest = json.loads((d / "manifest.json").read_text())
            if manifest.get("fingerprint") != fp or manifest.get("embedder") != embedder.name:
                return None
            matrix = np.load(d / "embeddings.npy", mmap_mode="r")
            chunks = [json.loads(l) for l in (d / "chunks.jsonl").read_text().splitlines() if l]
            if len(chunks) != matrix.shape[0]:
                return None
            if getattr(embedder, "dim", None) is None:  # a llama embedder learns its dim from the first call
                embedder.dim = int(matrix.shape[1])
            if matrix.shape[1] != getattr(embedder, "dim", matrix.shape[1]):
                return None
            return cls(name, chunks, matrix, embedder, manifest["source"], fp,
                       manifest.get("built_at", 0.0))
        except (OSError, ValueError, KeyError):
            return None

    def _to_cache(self, cache: Path) -> None:
        d = cache / self.name
        tmp = cache / f".{self.name}.tmp"
        try:
            tmp.mkdir(parents=True, exist_ok=True)
            np.save(tmp / "embeddings.npy", self.matrix)
            (tmp / "chunks.jsonl").write_text(
                "".join(json.dumps(c, ensure_ascii=False) + "\n" for c in self.chunks), encoding="utf-8")
            (tmp / "manifest.json").write_text(json.dumps(
                {"name": self.name, "source": self.source, "fingerprint": self.fingerprint,
                 "embedder": self.embedder.name, "dim": int(self.matrix.shape[1]), "chunks": len(self.chunks),
                 "built_at": self.built_at}), encoding="utf-8")
            if d.exists():
                import shutil
                shutil.rmtree(d)
            tmp.rename(d)
        except OSError:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)       # a read-only cache dir: build anyway, skip the cache

    def search(self, query: str, embedder2vec: dict, top_k: int = 4) -> list[dict]:
        emb = self.embedder
        q = embedder2vec.setdefault(id(emb), emb.embed([query])[0])
        scores = np.asarray(self.matrix) @ q            # both sides L2-normalized: the dot product is cosine
        order = np.argsort(-scores)[:max(1, int(top_k))]
        return [{"kb": self.name, "score": round(float(scores[i]), 4),
                 "text": self.chunks[int(i)]["text"], "source": self.chunks[int(i)].get("source", "")}
                for i in order]

    def stats(self) -> dict:
        return {"name": self.name, "source": self.source, "chunks": len(self.chunks),
                "dim": int(self.matrix.shape[1]), "embedder": self.embedder.name,
                "fingerprint": self.fingerprint, "built_at": self.built_at}


# --------------------------------------------------------------------------- the registry

class KBRegistry:
    """The hot-plug point. `snapshot()`/`search()` never block on a build: they take the lock only to read two
    references, and every mutator prepares the new state first (`_kb.build` runs outside the lock, so a big load
    does not stall answers) and publishes it in one assignment."""

    def __init__(self, embedder=None, cache_dir: str | Path | None = None, root: str | Path | None = None):
        self.embedder = embedder or LexicalEmbedder()
        self.cache = Path(cache_dir) if cache_dir else None
        self.root = Path(root).resolve() if root else None
        self.lock = RLock()
        self._kbs: dict[str, KnowledgeBase] = {}
        self._active: tuple[str, ...] = ()

    # -- the hot operations ----------------------------------------------------

    def load(self, name: str, source: str | Path, chunk_size: int = 900, overlap: int = 140) -> dict:
        """Build (or reattach from cache) and plug in. An existing name is replaced atomically: requests see
        the old KB until the new one is fully built, then the new one — never a mix."""
        if not name or not re.fullmatch(r"[\w.-]{1,64}", name):
            raise ValueError("kb name: 1-64 of letters, digits, ., _, -")
        src = self._under_root(source)
        kb = KnowledgeBase.build(name, src, self.embedder, chunk_size, overlap, self.cache)  # outside the lock
        with self.lock:
            self._kbs[name] = kb
            return kb.stats()

    def reload(self, name: str) -> dict:
        with self.lock:
            src = self._kbs[name].source                 # KeyError -> "unknown", mapped by the caller
        return self.load(name, src)

    def unload(self, name: str) -> dict:
        with self.lock:
            if name not in self._kbs:
                raise KeyError(name)
            del self._kbs[name]
            self._active = tuple(n for n in self._active if n != name)
            return {"name": name, "loaded": False, "active": list(self._active)}

    def swap(self, active: str | list[str]) -> dict:
        """Change which KBs answer requests by default, in one step (load the new one first, then swap — the
        gap between two loads is served by the old set). An empty list unplugs them all."""
        names = (active,) if isinstance(active, str) else tuple(active or ())
        if isinstance(active, str) and names and names[0] == "":
            names = ()
        with self.lock:
            missing = [n for n in names if n not in self._kbs]
            if missing:
                raise KeyError(", ".join(missing))
            self._active = names
            return self.list()

    # -- reads (what requests use) ----------------------------------------------

    def snapshot(self, names=None) -> tuple[KnowledgeBase, ...]:
        """The KB objects a request pins for its whole life. `names=None` is the active set; unknown names are a
        KeyError (the server answers 400 with the name, not a silent empty retrieval)."""
        with self.lock:
            want = self._active if names is None else tuple(names)
            try:
                return tuple(self._kbs[n] for n in want)
            except KeyError as e:
                raise KeyError(e.args[0]) from None

    def search(self, query: str, names=None, top_k: int = 4) -> list[dict]:
        kbs = self.snapshot(names)
        if not kbs or not query.strip():
            return []
        per_embedder: dict[int, np.ndarray] = {}         # one query embedding per distinct embedder
        hits = (h for kb in kbs for h in kb.search(query, per_embedder, top_k))
        return sorted(hits, key=lambda h: -h["score"])[:max(1, int(top_k))]

    def list(self) -> dict:
        with self.lock:
            return {"active": list(self._active),
                    "kbs": [kb.stats() for kb in self._kbs.values()]}

    def stats(self, name: str) -> dict:
        with self.lock:
            return self._kbs[name].stats()

    # -- wiring into requests ----------------------------------------------------

    def resolve(self, kb_field) -> tuple[str, ...] | None:
        """A request's "kb" field -> the names to search, or None when the request says nothing (then the
        server's active set decides). `false`/`null`/`""`/`[]` is an explicit off; "auto" is the active set."""
        if kb_field is None:
            return None
        if kb_field is False:
            return ()
        if kb_field == "auto":
            return self.snapshot_names()
        if isinstance(kb_field, str):
            return () if kb_field == "" else (kb_field,)
        if isinstance(kb_field, list) and all(isinstance(n, str) for n in kb_field):
            return tuple(kb_field)
        raise ValueError('"kb": a name, a list of names, "auto", or false to turn retrieval off')

    def snapshot_names(self) -> tuple[str, ...]:
        with self.lock:
            return self._active

    def inject(self, req: dict, messages: list[dict]) -> list[dict]:
        """The chat hook: retrieve with the last user message and hand the model the hits as context at the top
        of that same message (every chat template renders a user turn, while a system turn mid-conversation is
        quietly dropped by some). Returns the messages unchanged — the same list object — when the request has
        no user text, no hits or retrieval off, so the default path stays byte-identical."""
        names = self.resolve(req.get("kb"))
        if names is None:
            names = self.snapshot_names()
        if not names:
            return messages
        top_k = req.get("kb_top_k", 4)
        if not isinstance(top_k, int) or isinstance(top_k, bool) or not 1 <= top_k <= 32:
            raise ValueError("kb_top_k: a whole number from 1 to 32")
        last = next((m for m in reversed(messages) if m.get("role") == "user"), None)
        if last is None:
            return messages
        content = last.get("content")
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            text = "".join(p.get("text", "") for p in content
                           if isinstance(p, dict) and p.get("type") == "text")
        else:
            text = ""
        if not text.strip():
            return messages
        try:
            hits = self.search(text, names, top_k)
        except KeyError as e:
            raise ValueError(f"unknown knowledge base: {e.args[0]}") from None
        if not hits:
            return messages
        block = "\n".join(f"[knowledge base {h['kb']} #{i + 1} | source: {h['source']}]\n{h['text']}"
                          for i, h in enumerate(hits))
        context = f"[knowledge base context]\n{block}\n[end of knowledge base context]\n\n{text}"
        if isinstance(content, list):                    # a turn with images: a text part in front, the images keep their place
            last["content"] = [{"type": "text", "text": context}] + list(content)
        else:
            last["content"] = context
        return messages

    # -- helpers -------------------------------------------------------------------

    def _under_root(self, source: str | Path) -> Path:
        p = Path(source)
        if not p.is_absolute() and self.root is not None:
            p = self.root / p
        p = p.resolve()
        if self.root is not None and self.root not in p.parents and p != self.root:
            raise ValueError(f"{source}: outside the kb root {self.root}")
        if not p.exists():
            raise ValueError(f"{source}: not found")
        return p
