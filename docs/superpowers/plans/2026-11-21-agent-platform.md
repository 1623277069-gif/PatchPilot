# PatchPilot Agent Platform Implementation Plan

**Goal:** Run an issue through an isolated, resumable single-agent repair and manual review.

**Architecture:** The Worker owns a MySQL advisory lock and persists each LangGraph step. The workspace is reconstructed from the pinned Git commit and the checkpointed diff. Tests get a disposable repository archive with no host mounts or network.

**Tech Stack:** Python 3.12, LangGraph, LangChain Core, FastAPI, MySQL 8.4, Alembic, FAISS and Docker.

## Implementation

- [ ] Preserve the frozen `workflows/rs_sft/` tree. Map Base and LoRA selection to the IDs on the local OpenAI-compatible service. Exercise native function calls with an HTTP stub.
- [ ] Implement `workspace.py`, `sandbox.py` and `graph.py`. Verify relative paths, replay a patch from a clean commit, bound model and test calls, and rehydrate the proposed diff across an approval interrupt.
- [ ] Implement `rag.py` and versioned code metadata. Reuse AST chunks and their vectors for matching blobs. Check that deletions and changed blobs do not appear in searches of a new snapshot.
- [ ] Implement migrations, `api.py`, `worker.py` and Compose services. Exercise queue claims, retry, checkpoint recovery, persisted SSE cursors and approval of the reviewed hash.
- [ ] Run the legacy tests and the public-repository Mock demonstration. Record dependency limits and GPU verification separately. Audit the diff and publish a feature branch to the existing private remote.
