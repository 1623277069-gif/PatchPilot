import pytest
from langchain_core.documents import Document
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from patchpilot.graph import repair_graph
from patchpilot.model import DEMO_ISSUE, MockModel
from patchpilot.sandbox import TestResult as SandboxResult
from patchpilot.workspace import WorkspaceError, patch_sha


class IndexStub:
    def build(self, repository_id, tree):
        return tree.commit

    def search(self, snapshot_id, query):
        return [
            Document(
                page_content="def add_one(number)",
                metadata={
                    "path": "src/sample/simple.py",
                    "start": 1,
                    "end": 2,
                },
            )
        ]


class SandboxStub:
    calls = 0

    def run(self, job_id, workspace, profile):
        self.calls += 1
        source = workspace.text("src/sample/simple.py")
        return SandboxResult(0 if "number=0" in source else 1, "synthetic test feedback")


def initial(tree, max_steps=6):
    return {
        "job_id": "job",
        "repository_id": "repository",
        "commit": tree.commit,
        "issue": DEMO_ISSUE,
        "profile": "demo",
        "model": MockModel.identity,
        "max_steps": max_steps,
    }


def build(tree, sandbox, saver):
    return repair_graph(tree, MockModel(), IndexStub(), sandbox, saver, 120)


def test_worktree_is_replayable_and_test_files_are_protected(fixture_tree):
    tree = fixture_tree
    with tree.workspace() as workspace:
        with pytest.raises(WorkspaceError):
            workspace.text("../test_old.py")
        with pytest.raises(WorkspaceError):
            workspace.edit("tests/test_old.py", "assert 1", "assert 2")
        patch = workspace.edit("src/sample/simple.py", "number):", "number=0):")
    with tree.workspace(patch) as workspace:
        assert "number=0" in workspace.text("src/sample/simple.py")
    with tree.workspace() as workspace:
        assert "number=0" not in workspace.text("src/sample/simple.py")
    with pytest.raises(WorkspaceError):
        with tree.workspace(patch.replace("return number + 1", "return number + 9")):
            pass


@pytest.mark.parametrize(("approve", "outcome"), [(True, "succeeded"), (False, "rejected")])
def test_review_replays_only_the_saved_diff(fixture_tree, approve, outcome):
    saver = InMemorySaver()
    sandbox = SandboxStub()
    config = {"configurable": {"thread_id": "test"}}
    first = build(fixture_tree, sandbox, saver)
    interrupted = first.invoke(initial(fixture_tree), config, durability="sync")
    assert "__interrupt__" in interrupted
    sha = patch_sha(fixture_tree.commit, interrupted["patch"])
    assert sha == interrupted["review_sha"] == interrupted["verification"]["sha"]
    second = build(fixture_tree, sandbox, saver)
    result = second.invoke(
        Command(resume={"approve": approve, "patch_sha": sha}), config, durability="sync"
    )
    assert result["outcome"] == outcome
    assert result["patch"] == interrupted["patch"]
    assert sandbox.calls == 2


def test_budget_exhaustion_is_not_a_verified_patch(fixture_tree):
    graph = build(fixture_tree, SandboxStub(), InMemorySaver())
    state = graph.invoke(initial(fixture_tree, max_steps=2), {"configurable": {"thread_id": "short"}})
    assert state["outcome"] == "unresolved"
    assert state["step"] == 2
    assert not state["verification"]


def test_wrong_review_hash_is_rejected(fixture_tree):
    graph = build(fixture_tree, SandboxStub(), InMemorySaver())
    config = {"configurable": {"thread_id": "wrong"}}
    graph.invoke(initial(fixture_tree), config)
    with pytest.raises(WorkspaceError, match="does not match"):
        graph.invoke(Command(resume={"approve": True, "patch_sha": "0" * 64}), config)
