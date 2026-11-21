import os
from contextlib import closing
from uuid import uuid4

import pytest
from conftest import TOKEN, local_commit
from fastapi.testclient import TestClient
from langchain_core.embeddings import Embeddings
from langgraph.checkpoint.mysql.pymysql import PyMySQLSaver
from sqlalchemy import select
from test_graph import SandboxStub

from patchpilot.api import create_app
from patchpilot.db import Job, Repository
from patchpilot.model import DEMO_COMMIT, DEMO_ISSUE, DEMO_REPO
from patchpilot.rag import CodeIndex
from patchpilot.sandbox import DockerSandbox
from patchpilot.worker import Worker
from patchpilot.workspace import Tree

pytestmark = pytest.mark.mysql
AUTH = {"Authorization": f"Bearer {TOKEN}"}


def test_persisted_review_api_sse_and_worker_recovery(mysql_settings, fixture_tree, monkeypatch):
    # Exercise the real database and checkpoints with a deterministic sandbox double.
    monkeypatch.setattr("patchpilot.worker.fetch", lambda *args: PinnedFixture(fixture_tree))
    drive_review(mysql_settings, sandbox=SandboxStub())


class PinnedFixture:
    # Preserve the API's demo pin while using a local synthetic source tree for this test.
    commit = DEMO_COMMIT

    def __init__(self, tree):
        self.files = tree.files
        self.workspace = tree.workspace


def drive_review(settings, sandbox):
    worker = Worker(settings)
    worker.sandbox = sandbox
    with TestClient(create_app(settings)) as client:
        imported = client.post("/repositories", json={"url": DEMO_REPO}, headers=AUTH)
        assert imported.status_code == 201, imported.text
        repository = imported.json()
        repeated = client.post("/repositories", json={"url": DEMO_REPO}, headers=AUTH)
        assert repeated.json()["id"] == repository["id"]
        submitted = client.post(
            "/jobs",
            json={
                "repository_id": repository["id"],
                "commit": DEMO_COMMIT,
                "issue": DEMO_ISSUE,
            },
            headers=AUTH,
        )
        assert submitted.status_code == 202, submitted.text
        job_id = submitted.json()["id"]
        with closing(worker.connection()) as connection:
            assert worker.acquire(connection)
            with closing(worker.connection()) as competitor:
                assert not worker.acquire(competitor)
            PyMySQLSaver(connection).setup()
            # Crash after checkpoint persistence but before publishing the review row.
            finish = worker.finish

            def unpublish(*args, **kwargs):
                worker.finish = finish
                raise SystemExit("simulate worker termination")

            worker.finish = unpublish
            with pytest.raises(SystemExit):
                worker.run_once(connection)
        assert client.get(f"/jobs/{job_id}", headers=AUTH).json()["status"] == "running"
        recovered_worker = Worker(settings)
        recovered_worker.sandbox = sandbox
        with closing(recovered_worker.connection()) as connection:
            assert recovered_worker.acquire(connection)
            assert recovered_worker.run_once(connection)
        waiting = client.get(f"/jobs/{job_id}", headers=AUTH).json()
        assert waiting["status"] == "awaiting_approval", waiting
        assert waiting["result"]["baseline_exit"] == 1
        assert waiting["result"]["test_exit"] == 0
        patch = client.get(f"/jobs/{job_id}/patch", headers=AUTH)
        assert patch.status_code == 200 and "number=0" in patch.text
        digest = patch.headers["X-Patch-SHA256"]
        assert digest == waiting["review_sha"]
        assert (
            client.post(
                f"/jobs/{job_id}/approval", headers=AUTH, json={"approve": True, "patch_sha": "0" * 64}
            ).status_code
            == 409
        )
        assert (
            client.post(
                f"/jobs/{job_id}/approval", headers=AUTH, json={"approve": True, "patch_sha": digest}
            ).status_code
            == 202
        )
        resumed = Worker(settings)
        resumed.sandbox = sandbox
        with closing(resumed.connection()) as connection:
            assert resumed.acquire(connection)
            assert resumed.run_once(connection)
        result = client.get(f"/jobs/{job_id}", headers=AUTH).json()
        assert result["status"] == "succeeded", result
        assert result["result"]["model_identity"].startswith("mock:")
        assert client.get(f"/jobs/{job_id}/patch", headers=AUTH).text == patch.text
        stream = client.get(f"/jobs/{job_id}/events", headers=AUTH)
        assert "event: awaiting_approval" in stream.text
        assert "event: succeeded" in stream.text
        cursor = next(line[4:] for line in stream.text.splitlines() if line.startswith("id: "))
        replay = client.get(f"/jobs/{job_id}/events", headers={**AUTH, "Last-Event-ID": cursor})
        assert "event: queued\n" not in replay.text
        assert (
            client.get("/jobs", headers=AUTH, params={"status": "succeeded"}).json()[0]["id"] == job_id
        )


@pytest.mark.sandbox
def test_public_repo_full_docker_demo(mysql_settings):
    if os.environ.get("PATCHPILOT_TEST_DOCKER") != "1":
        pytest.skip("Set PATCHPILOT_TEST_DOCKER=1 after building the sandbox image")
    drive_review(mysql_settings, DockerSandbox(mysql_settings.sandbox_image, 35))


class CountingEmbeddings(Embeddings):
    def __init__(self):
        self.encoded = []

    def embed_documents(self, documents):
        self.encoded.extend(documents)
        return [[1.0, 0.0] if "unchanged_token" in doc else [0.0, 1.0] for doc in documents]

    def embed_query(self, query):
        return [1.0, 0.0]


def test_incremental_commit_scoped_hybrid_retrieval(mysql_settings, tmp_path):
    root = tmp_path / "source"
    first_sha = local_commit(
        root,
        {
            "keep.py": "def unchanged_token():\n    return True\n",
            "changed.py": "def legacy_handler():\n    return 1\n",
            "removed.py": "def deleted_token():\n    return False\n",
        },
    )
    second_sha = local_commit(
        root,
        {
            "changed.py": "def new_handler():\n    return 2\n",
            "removed.py": None,
        },
    )
    worker = Worker(mysql_settings)
    repository_id = str(uuid4())
    with worker.sessions.begin() as session:
        session.add(Repository(id=repository_id, url=f"https://github.com/pypa/{repository_id}.git"))
    embeddings = CountingEmbeddings()
    index = CodeIndex(worker.sessions, tmp_path / "faiss", "test", embeddings)
    first = index.build(repository_id, Tree(root / ".git", first_sha))
    assert len(embeddings.encoded) == 3
    second = index.build(repository_id, Tree(root / ".git", second_sha))
    assert len(embeddings.encoded) == 4
    assert "new_handler" in embeddings.encoded[-1]
    hits = index.search(second, "new_handler")
    assert {hit.metadata["path"] for hit in hits} == {"keep.py", "changed.py"}
    assert all(hit.metadata["commit"] == second_sha for hit in hits)
    assert all("legacy_handler" not in hit.page_content for hit in hits)
    assert any(hit.metadata["path"] == "removed.py" for hit in index.search(first, "deleted_token"))
    assert index.build(repository_id, Tree(root / ".git", second_sha)) == second
    assert len(embeddings.encoded) == 4


def test_failed_jobs_can_be_retried(mysql_settings, monkeypatch):
    from patchpilot.sandbox import SandboxError

    class Unavailable:
        def run(self, *args):
            raise SandboxError("synthetic unavailability")

    monkeypatch.setattr("patchpilot.worker.time.sleep", lambda _: None)
    worker = Worker(mysql_settings)
    worker.sandbox = Unavailable()
    with TestClient(create_app(mysql_settings)) as client:
        repository = client.post("/repositories", json={"url": DEMO_REPO}, headers=AUTH).json()
        job = client.post(
            "/jobs",
            json={
                "repository_id": repository["id"],
                "commit": DEMO_COMMIT,
                "issue": DEMO_ISSUE,
            },
            headers=AUTH,
        ).json()
        with closing(worker.connection()) as connection:
            assert worker.acquire(connection)
            PyMySQLSaver(connection).setup()
            for _ in range(3):
                assert worker.run_once(connection)
        assert client.get(f"/jobs/{job['id']}", headers=AUTH).json()["status"] == "failed"
        assert client.post(f"/jobs/{job['id']}/retry", headers=AUTH).json()["status"] == "queued"
        with worker.sessions.begin() as session:
            row = session.scalar(select(Job).where(Job.id == job["id"]))
            row.status = "failed"


def test_real_local_embeddings_in_faiss(mysql_settings, fixture_tree, tmp_path):
    cache = os.environ.get("PATCHPILOT_TEST_EMBEDDING_CACHE")
    if not cache:
        pytest.skip("Set PATCHPILOT_TEST_EMBEDDING_CACHE to a prepared FastEmbed cache")

    from pathlib import Path

    from patchpilot.rag import LocalEmbeddings

    name = "BAAI/bge-small-en-v1.5"
    worker = Worker(mysql_settings)
    repository_id = str(uuid4())
    with worker.sessions.begin() as session:
        session.add(Repository(id=repository_id, url=f"https://github.com/pypa/{repository_id}.git"))
    code = CodeIndex(worker.sessions, tmp_path / "vectors", name, LocalEmbeddings(name, Path(cache)))
    snapshot_id = code.build(repository_id, fixture_tree)
    assert code.vector_path(snapshot_id).stat().st_size > 0
    hits = code.search(snapshot_id, "increment the input with add_one")
    assert hits[0].metadata["path"] == "src/sample/simple.py"
    assert all(hit.metadata["commit"] == fixture_tree.commit for hit in hits)
