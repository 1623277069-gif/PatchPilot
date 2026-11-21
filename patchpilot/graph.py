"""One StateGraph agent. Only the canonical Git diff is persisted between tools."""

import time
from dataclasses import asdict
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from .model import Decision, RepairModel
from .workspace import Tree, WorkspaceError, patch_sha


class State(TypedDict, total=False):
    job_id: str
    repository_id: str
    commit: str
    issue: str
    profile: str
    model: str
    max_steps: int
    deadline: float
    snapshot_id: str
    context: str
    feedback: list[str]
    step: int
    action: dict
    patch: str
    baseline: dict
    verification: dict | None
    outcome: str
    detail: str
    review_sha: str


def format_context(documents):
    return "\n\n".join(
        f"{doc.metadata['path']}:{doc.metadata['start']}-{doc.metadata['end']}\n{doc.page_content}"
        for doc in documents
    )[:9000]


def repair_graph(tree: Tree, model: RepairModel, index, sandbox, saver, max_seconds: int):
    def initialize(state: State):
        if state["model"] != model.identity or state["commit"] != tree.commit:
            raise WorkspaceError("Task identity changed")
        return {
            "patch": "",
            "step": 0,
            "feedback": [],
            "verification": None,
            "deadline": time.time() + max_seconds,
        }

    def retrieve(state: State):
        snapshot_id = index.build(state["repository_id"], tree)
        documents = index.search(snapshot_id, state["issue"])
        return {"snapshot_id": snapshot_id, "context": format_context(documents)}

    def baseline(state: State):
        with tree.workspace() as workspace:
            result = sandbox.run(state["job_id"], workspace, state["profile"])
        values = {"baseline": {"code": result.code, "timed_out": result.timed_out}}
        if result.code != 1 or result.timed_out:
            return dict(values, outcome="unresolved", detail="baseline_did_not_reproduce")
        return dict(values, feedback=[f"Baseline failed with exit 1:\n{result.output}"])

    def decide(state: State):
        if state["step"] >= state["max_steps"] or time.time() >= state["deadline"]:
            return {"outcome": "unresolved", "detail": "budget_exhausted"}
        decision = model.decide(state["issue"], state["context"], state["feedback"], state["step"])
        return {"action": decision.model_dump(), "step": state["step"] + 1}

    def report(state: State, message: str, **values):
        return {"feedback": [*state["feedback"][-4:], message[:8500]], **values}

    def execute(state: State):
        if time.time() >= state["deadline"]:
            return {"outcome": "unresolved", "detail": "budget_exhausted"}
        decision = Decision.model_validate(state["action"])
        try:
            args = decision.validated()
        except ValueError:
            return report(state, "Invalid action. Use one listed tool with valid arguments.")
        name = decision.name
        if name == "search_code":
            documents = index.search(state["snapshot_id"], args.query)
            return report(state, format_context(documents) or "No matching base chunks.")
        if name == "get_status":
            return report(
                state,
                f"Actions used: {state['step']}/{state['max_steps']}. "
                f"Current test: {state['verification'] and state['verification']['code']}.",
            )
        if name == "submit_patch":
            verified = state["verification"]
            identity = patch_sha(state["commit"], state["patch"])
            if state["patch"] and verified and verified["code"] == 0 and verified["sha"] == identity:
                return {"outcome": "review", "review_sha": identity}
            return report(state, "Submission requires a nonempty diff and a fresh passing test.")
        try:
            with tree.workspace(state["patch"]) as workspace:
                if name == "read_file":
                    lines = workspace.text(args.path).splitlines()[
                        args.start - 1 : args.start - 1 + args.lines
                    ]
                    return report(
                        state,
                        "\n".join(f"{i}: {line}" for i, line in enumerate(lines, start=args.start)),
                    )
                if name == "edit_file":
                    patch = workspace.edit(args.path, args.old, args.new)
                    return report(
                        state,
                        f"Updated {args.path}. Run the acceptance profile.",
                        patch=patch,
                        verification=None,
                    )
        except WorkspaceError as exc:
            return report(state, str(exc))
        return {}

    def test(state: State):
        if time.time() >= state["deadline"]:
            return {"outcome": "unresolved", "detail": "budget_exhausted"}
        with tree.workspace(state["patch"]) as workspace:
            result = sandbox.run(state["job_id"], workspace, state["profile"])
        verification = dict(asdict(result), sha=patch_sha(state["commit"], state["patch"]))
        return report(state, f"Sandbox exit {result.code}:\n{result.output}", verification=verification)

    def review(state: State):
        identity = patch_sha(state["commit"], state["patch"])
        if identity != state["review_sha"]:
            raise WorkspaceError("Review hash mismatch")
        # Replaying the proposed patch against the pinned tree precedes every approval.
        with tree.workspace(state["patch"]):
            pass
        answer = interrupt({"commit": state["commit"], "patch_sha": identity})
        if answer["patch_sha"] != identity or type(answer["approve"]) is not bool:
            raise WorkspaceError("Approval does not match the candidate")
        return {"outcome": "succeeded" if answer["approve"] else "rejected", "detail": "reviewed"}

    def route(state: State):
        if state.get("outcome") == "review":
            return "review"
        return END if state.get("outcome") else "decide"

    graph = StateGraph(State)
    for node in (initialize, retrieve, baseline, decide, execute, test, review):
        graph.add_node(node.__name__, node)
    graph.add_edge(START, "initialize")
    graph.add_edge("initialize", "retrieve")
    graph.add_edge("retrieve", "baseline")
    graph.add_conditional_edges("baseline", route, {"decide": "decide", END: END, "review": "review"})
    graph.add_conditional_edges(
        "decide",
        lambda state: (
            END
            if state.get("outcome")
            else (
                "test"
                if state["action"]["name"] == "run_command"
                and state["action"]["arguments"] in ({}, {"command": "test"})
                else "execute"
            )
        ),
        ["test", "execute", END],
    )
    graph.add_conditional_edges("execute", route, {"decide": "decide", END: END, "review": "review"})
    graph.add_conditional_edges("test", route, {"decide": "decide", END: END, "review": "review"})
    graph.add_edge("review", END)
    return graph.compile(checkpointer=saver)
