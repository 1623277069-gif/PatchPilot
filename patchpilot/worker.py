"""Independent worker with one lock owner per MySQL database and durable replay."""

import hashlib
import logging
import time
from contextlib import closing
from uuid import uuid4

import pymysql
from langgraph.checkpoint.mysql.pymysql import PyMySQLSaver
from langgraph.types import Command
from sqlalchemy import select
from sqlalchemy.engine import make_url
from sqlalchemy.exc import OperationalError

from .config import Settings
from .db import Job, Repository, event, make_engine, sessions
from .graph import repair_graph
from .model import GemmaModel, MockModel, ModelError
from .rag import CodeIndex, LocalEmbeddings
from .sandbox import DockerSandbox, SandboxError
from .workspace import ImportError as FetchError
from .workspace import WorkspaceError, fetch, patch_sha

MAX_ATTEMPTS = 3
LOG = logging.getLogger("patchpilot.worker")


class LockLost(RuntimeError):
    pass


class Worker:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.engine = make_engine(settings)
        self.sessions = sessions(self.engine)
        self.root = settings.state_dir.resolve()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.index = None
        self.sandbox = DockerSandbox(
            settings.sandbox_image, settings.test_timeout, settings.sandbox_runtime
        )
        self.url = make_url(settings.database_url.get_secret_value())
        self.lock = "patchpilot:" + hashlib.sha256(self.url.database.encode()).hexdigest()[:40]

    def connection(self):
        return pymysql.connect(
            host=self.url.host or "localhost",
            port=self.url.port or 3306,
            user=self.url.username,
            password=self.url.password,
            database=self.url.database,
            unix_socket=self.url.query.get("unix_socket"),
            autocommit=True,
            connect_timeout=5,
            read_timeout=15,
            write_timeout=15,
            charset="utf8mb4",
        )

    def acquire(self, connection):
        with connection.cursor() as cursor:
            cursor.execute("SELECT GET_LOCK(%s, 0)", (self.lock,))
            return cursor.fetchone()[0] == 1

    def own_lock(self, connection):
        # The checkpointer also uses this physical connection and never reconnects it.
        with connection.cursor() as cursor:
            cursor.execute("SELECT IS_USED_LOCK(%s) = CONNECTION_ID()", (self.lock,))
            if cursor.fetchone()[0] != 1:
                raise LockLost("Worker lost ownership")

    def next_job(self):
        with self.sessions.begin() as session:
            job = session.scalar(
                select(Job)
                .where(Job.status.in_(["queued", "running"]))
                .order_by(Job.created_at, Job.id)
                .limit(1)
                .with_for_update(skip_locked=True)
            )
            if job is None:
                return None
            if job.attempts >= MAX_ATTEMPTS:
                job.status = "failed"
                job.error = "retry_limit"
                event(session, job.id, "failed", reason=job.error)
                return None
            resumed = job.status == "running"
            job.status = "running"
            job.attempts += 1
            job.claim_token = str(uuid4())
            job.error = None
            event(session, job.id, "running", attempt=job.attempts, recovered=resumed)
            return job

    @staticmethod
    def owned_row(session, claimed):
        row = session.get(Job, claimed.id, with_for_update=True)
        if row.status != "running" or row.claim_token != claimed.claim_token:
            raise LockLost("Task claim was replaced")
        return row

    def record_node(self, job, node, updates):
        with self.sessions.begin() as session:
            self.owned_row(session, job)
            details = {}
            if "step" in updates:
                details["step"] = updates["step"]
                details["tool"] = updates.get("action", {}).get("name")
            if node == "baseline":
                details.update(updates["baseline"])
            if "verification" in updates and updates["verification"]:
                details["test_exit"] = updates["verification"]["code"]
            event(session, job.id, node, **details)

    def process(self, job, connection):
        with self.sessions() as session:
            url = session.get(Repository, job.repository_id).url
        token = self.settings.github_token
        tree = fetch(self.root, job.repository_id, url, job.commit, token and token.get_secret_value())
        if self.index is None:
            name = self.settings.embedding_model
            embedder = LocalEmbeddings(name, self.settings.embedding_cache) if name else None
            self.index = CodeIndex(self.sessions, self.root / "faiss", name, embedder)
        model = MockModel() if job.model_mode == "mock" else GemmaModel(self.settings, job.model_mode)
        try:
            self.run_graph(job, tree, model, connection)
        finally:
            if isinstance(model, GemmaModel):
                model.close()

    def run_graph(self, job, tree, model, connection):
        self.own_lock(connection)
        saver = PyMySQLSaver(connection)
        graph = repair_graph(tree, model, self.index, self.sandbox, saver, self.settings.max_seconds)
        config = {"configurable": {"thread_id": job.id}, "recursion_limit": job.max_steps * 2 + 12}
        snapshot = graph.get_state(config)
        initial = {
            "job_id": job.id,
            "repository_id": job.repository_id,
            "commit": job.commit,
            "issue": job.issue,
            "profile": job.profile,
            "model": model.identity,
            "max_steps": job.max_steps,
        }
        if snapshot.values and any(snapshot.values.get(key) != value for key, value in initial.items()):
            raise WorkspaceError("Resume inputs differ from the pinned task")
        if snapshot.interrupts:
            pending = snapshot.values
            if job.approval is None:
                self.finish(job, pending, awaiting=True)
                return
            if job.review_sha != pending.get("review_sha"):
                raise WorkspaceError("Approval does not match saved review")
            graph_input = Command(resume={"approve": job.approval, "patch_sha": job.review_sha})
        elif not snapshot.values:
            graph_input = initial
        elif snapshot.next:
            graph_input = None
        else:
            self.finish(job, snapshot.values)
            return
        for frame in graph.stream(graph_input, config, stream_mode="updates", durability="sync"):
            # LangGraph may write pending results on a background thread during streaming.
            with saver.lock:
                self.own_lock(connection)
            for node, updates in frame.items():
                if not node.startswith("__") and updates:
                    self.record_node(job, node, updates)
        final = graph.get_state(config)
        self.own_lock(connection)
        self.finish(job, final.values, awaiting=bool(final.interrupts))

    def finish(self, claimed, state, awaiting=False):
        patch = state.get("patch") or ""
        review_sha = state.get("review_sha")
        verification = state.get("verification")
        if review_sha:
            if (
                review_sha != patch_sha(state["commit"], patch)
                or (verification or {}).get("sha") != review_sha
            ):
                raise WorkspaceError("Verified diff identity changed")
        status = "awaiting_approval" if awaiting else state["outcome"]
        with self.sessions.begin() as session:
            job = self.owned_row(session, claimed)
            job.claim_token = None
            job.status = status
            job.patch = patch or None
            job.review_sha = review_sha
            job.result = {
                "model_identity": state["model"],
                "baseline_exit": state.get("baseline", {}).get("code"),
                "test_exit": verification and verification["code"],
                "test_timed_out": verification and verification["timed_out"],
                "test_patch_sha": verification and verification["sha"],
                "steps": state.get("step", 0),
                "detail": state.get("detail"),
            }
            event(session, job.id, status, patch_sha=review_sha, **job.result)

    def run_once(self, connection):
        self.own_lock(connection)
        job = self.next_job()
        if job is None:
            return False
        try:
            self.process(job, connection)
        except (pymysql.MySQLError, OperationalError, LockLost):
            raise
        except Exception as exc:
            self.own_lock(connection)
            reason = {
                FetchError: "git_fetch_failed",
                ModelError: "model_service_failed",
                SandboxError: "sandbox_unavailable",
                WorkspaceError: "workspace_inconsistent",
            }.get(type(exc), "processing_error")
            retryable = isinstance(exc, (FetchError, ModelError, SandboxError))
            LOG.warning("Job %s failed step (%s, %s)", job.id, reason, type(exc).__name__)
            with self.sessions.begin() as session:
                row = self.owned_row(session, job)
                row.claim_token = None
                row.error = reason
                row.status = "queued" if retryable and row.attempts < MAX_ATTEMPTS else "failed"
                event(
                    session,
                    row.id,
                    "retry_scheduled" if row.status == "queued" else "failed",
                    reason=reason,
                    attempt=row.attempts,
                )
            if retryable:
                time.sleep(job.attempts)
        return True


def main():
    logging.basicConfig(level=logging.INFO)
    worker = Worker(Settings())
    while True:
        try:
            with closing(worker.connection()) as connection:
                if worker.acquire(connection):
                    PyMySQLSaver(connection).setup()
                    while True:
                        if not worker.run_once(connection):
                            time.sleep(1)
        except (pymysql.MySQLError, OperationalError, LockLost) as exc:
            LOG.warning("Worker reconnecting (%s)", type(exc).__name__)
        time.sleep(2)


if __name__ == "__main__":
    main()
