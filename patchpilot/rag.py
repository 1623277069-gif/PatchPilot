"""Commit-scoped code retrieval with AST chunks and optional local FAISS search."""

import ast
import hashlib
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import faiss
import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from sqlalchemy import select

from .db import CodeChunk, Snapshot
from .workspace import Tree, WorkspaceError

CHUNK_VERSION = "2"
CHUNK_LIMIT = 10000
SOURCE_SUFFIXES = {
    ".py",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".go",
    ".rs",
    ".java",
    ".c",
    ".h",
    ".cpp",
    ".cs",
    ".rb",
    ".php",
    ".sh",
    ".sql",
    ".md",
    ".toml",
    ".yaml",
    ".yml",
    ".json",
}
WORDS = re.compile(r"[a-zA-Z][a-zA-Z0-9_]{1,60}")
CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|_+")


def tokens(text):
    for word in WORDS.findall(text):
        yield word.lower()
        yield from (part.lower() for part in CAMEL.split(word) if part != word and len(part) > 1)


@dataclass(frozen=True)
class Slice:
    symbol: str
    start_line: int
    end_line: int
    content: str


def slices(path: str, source: str):
    lines = source.splitlines()
    definitions = []
    if path.endswith(".py"):
        try:
            tree = ast.parse(source)
            definitions = [
                (
                    min([node.lineno, *(item.lineno for item in node.decorator_list)]),
                    node.end_lineno,
                    node.name,
                )
                for node in ast.walk(tree)
                if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
            ]
        except (SyntaxError, RecursionError):
            pass
    covered = bytearray(len(lines))
    for start, end, symbol in sorted(definitions):
        covered[start - 1 : end] = b"\1" * (end - start + 1)
        yield from windows(lines, start, end, symbol[:512])
    start = 0
    while start < len(lines):
        if covered[start]:
            start += 1
            continue
        end = start + 1
        while end < len(lines) and not covered[end]:
            end += 1
        yield from windows(lines, start + 1, end, "")
        start = end


def windows(lines, start, end, symbol):
    for first in range(start, end + 1, 30):
        last = min(first + 29, end)
        text = "\n".join(line[:300] for line in lines[first - 1 : last])
        if text.strip():
            yield Slice(symbol, first, last, text[:5000])


class LocalEmbeddings(Embeddings):
    def __init__(self, model: str, cache: Path):
        from fastembed import TextEmbedding

        self.model = TextEmbedding(
            model_name=model, cache_dir=str(cache), local_files_only=True, threads=2
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [vector.tolist() for vector in self.model.embed(texts, batch_size=16)]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self.model.query_embed(text))).tolist()


def normalized(vectors):
    array = np.ascontiguousarray(vectors, dtype=np.float32)
    faiss.normalize_L2(array)
    return array


class CodeIndex:
    def __init__(
        self, session_factory, directory: Path, identity="", embedder: Embeddings | None = None
    ):
        self.sessions = session_factory
        self.directory = directory
        self.embedder = embedder
        self.identity = hashlib.sha256(f"{CHUNK_VERSION}:{identity}".encode()).hexdigest()

    def vector_path(self, snapshot_id):
        return self.directory / f"{snapshot_id}.faiss"

    def read_vectors(self, snapshot_id, rows):
        if not self.embedder or not rows:
            return None
        path = self.vector_path(snapshot_id)
        if not path.is_file():
            return None
        try:
            index = faiss.read_index(str(path))
        except RuntimeError:
            return None
        if index.ntotal != len(rows) or index.metric_type != faiss.METRIC_INNER_PRODUCT:
            return None
        return index

    def build(self, repository_id: str, tree: Tree):
        with self.sessions.begin() as session:
            versions = select(Snapshot).where(
                Snapshot.repository_id == repository_id,
                Snapshot.embedding == self.identity,
            )
            existing = session.scalar(versions.where(Snapshot.commit == tree.commit))
            if existing:
                rows = self.rows(session, existing.id)
                if not self.embedder or not rows or self.read_vectors(existing.id, rows) is not None:
                    return existing.id
                session.delete(existing)
                session.flush()
            previous = session.scalar(
                versions.order_by(Snapshot.created_at.desc(), Snapshot.id).limit(1)
            )
            old_rows = self.rows(session, previous.id) if previous else []
            old_index = self.read_vectors(previous.id, old_rows) if previous else None
            old_files = defaultdict(list)
            for row in old_rows:
                old_files[(row.path, row.blob)].append(row)
            snapshot = Snapshot(
                id=str(uuid4()), repository_id=repository_id, commit=tree.commit, embedding=self.identity
            )
            session.add(snapshot)
            session.flush()
            new_rows = []
            vectors = []
            missing = []
            with tree.workspace() as workspace:
                for path, file in sorted(tree.files.items()):
                    if Path(path).suffix.lower() not in SOURCE_SUFFIXES:
                        continue
                    prior = old_files[path, file.oid]
                    if prior:
                        pieces = [
                            Slice(row.symbol, row.start_line, row.end_line, row.content) for row in prior
                        ]
                    else:
                        try:
                            pieces = slices(path, workspace.text(path, limit=128 * 1024))
                        except WorkspaceError:
                            continue
                    for offset, piece in enumerate(pieces):
                        if len(new_rows) >= CHUNK_LIMIT:
                            break
                        vector_id = len(new_rows) if self.embedder else None
                        row = CodeChunk(
                            snapshot_id=snapshot.id,
                            path=path,
                            blob=file.oid,
                            symbol=piece.symbol,
                            start_line=piece.start_line,
                            end_line=piece.end_line,
                            content=piece.content,
                            vector_id=vector_id,
                        )
                        new_rows.append(row)
                        if not self.embedder:
                            continue
                        if old_index is not None and prior:
                            vectors.append(old_index.reconstruct(prior[offset].vector_id))
                        else:
                            vectors.append(None)
                            missing.append(row)
            if vectors:
                self.write_vectors(snapshot.id, vectors, missing)
            session.add_all(new_rows)
            return snapshot.id

    def write_vectors(self, snapshot_id, vectors, missing):
        for first in range(0, len(missing), 64):
            batch = missing[first : first + 64]
            values = self.embedder.embed_documents(
                [f"{row.path} {row.symbol}\n{row.content}" for row in batch]
            )
            for row, value in zip(batch, values, strict=True):
                vectors[row.vector_id] = value
        data = normalized(vectors)
        index = faiss.IndexFlatIP(data.shape[1])
        index.add(data)
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        path = self.vector_path(snapshot_id)
        temporary = path.with_suffix(".tmp")
        try:
            faiss.write_index(index, str(temporary))
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def rows(session, snapshot_id):
        return list(
            session.scalars(
                select(CodeChunk).where(CodeChunk.snapshot_id == snapshot_id).order_by(CodeChunk.id)
            )
        )

    def search(self, snapshot_id: str, query: str, limit=6) -> list[Document]:
        with self.sessions() as session:
            snapshot = session.get(Snapshot, snapshot_id)
            if snapshot is None or snapshot.embedding != self.identity:
                raise ValueError("Code index version unavailable")
            rows = self.rows(session, snapshot_id)
            commit = snapshot.commit
        terms = set(tokens(query))
        lexical = []
        for offset, row in enumerate(rows):
            code = Counter(tokens(row.content))
            labels = set(tokens(f"{row.path} {row.symbol}"))
            score = sum(min(3, code[term]) + 4 * (term in labels) for term in terms)
            if score:
                lexical.append((score, offset))
        lexical.sort(key=lambda pair: (-pair[0], pair[1]))
        ranks = [[offset for _, offset in lexical[: max(30, limit)]]]
        vector_index = self.read_vectors(snapshot_id, rows)
        if vector_index is not None:
            _, ids = vector_index.search(normalized([self.embedder.embed_query(query)]), max(30, limit))
            positions = {row.vector_id: offset for offset, row in enumerate(rows)}
            ranks.append([positions[int(vector)] for vector in ids[0] if vector >= 0])
        elif self.embedder and rows:
            raise ValueError("FAISS data unavailable for the current snapshot")
        fused = defaultdict(float)
        for ranking in ranks:
            for position, offset in enumerate(ranking, start=1):
                fused[offset] += 1 / (30 + position)
        hits = sorted(fused, key=lambda offset: (-fused[offset], offset))[:limit]
        return [
            Document(
                page_content=row.content,
                metadata={
                    "path": row.path,
                    "start": row.start_line,
                    "end": row.end_line,
                    "symbol": row.symbol,
                    "commit": commit,
                },
            )
            for row in (rows[offset] for offset in hits)
        ]
