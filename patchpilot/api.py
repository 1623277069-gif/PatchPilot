"""Authenticated local API for repository imports, repair jobs and review."""

import asyncio
import hmac
import json
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Annotated
from uuid import UUID, uuid4

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import PlainTextResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy import select, text
from sqlalchemy.dialects.mysql import insert

from .config import Mode, Profile, Settings, repo_url
from .db import TERMINAL, Event, Job, Repository, event, make_engine, now, sessions
from .model import DEMO_COMMIT, DEMO_ISSUE, DEMO_REPO


class Input(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class ImportRepository(Input):
    url: str = Field(max_length=200)

    @field_validator("url")
    @classmethod
    def canonical(cls, value):
        return repo_url(value)


class Issue(Input):
    repository_id: UUID
    commit: str = Field(pattern=r"^[a-f0-9]{40}$")
    issue: str = Field(min_length=3, max_length=12000)
    model_mode: Mode = "mock"
    profile: Profile = "demo"
    max_steps: int = Field(default=16, ge=1, le=28)


class Approval(Input):
    approve: bool = Field(strict=True)
    patch_sha: str = Field(pattern=r"^[a-f0-9]{64}$")


class RepoView(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    url: str
    created_at: datetime


class JobView(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    repository_id: str
    commit: str
    issue: str
    model_mode: Mode
    profile: Profile
    max_steps: int
    status: str
    attempts: int
    review_sha: str | None
    approval: bool | None
    result: dict | None
    error: str | None
    created_at: datetime
    updated_at: datetime


def create_app(settings: Settings | None = None):
    settings = settings or Settings()
    engine = make_engine(settings)
    factory = sessions(engine)
    bearer = HTTPBearer(auto_error=False)

    @asynccontextmanager
    async def lifespan(app):
        yield
        engine.dispose()

    def operator(credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)]):
        secret = settings.api_token.get_secret_value().encode()
        if not credentials or not hmac.compare_digest(credentials.credentials.encode(), secret):
            raise HTTPException(401, "Bearer token required", headers={"WWW-Authenticate": "Bearer"})

    app = FastAPI(title="PatchPilot", version="0.1.0", lifespan=lifespan)
    app.state.sessions = factory
    protected = [Depends(operator)]

    def get_job(session, job_id, lock=False):
        job = session.get(Job, str(job_id), with_for_update=lock)
        if job is None:
            raise HTTPException(404, "Repair job not found")
        return job

    @app.get("/health", include_in_schema=False)
    def health():
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        return {"status": "ok"}

    @app.post("/repositories", response_model=RepoView, status_code=201, dependencies=protected)
    def import_repository(body: ImportRepository):
        with factory.begin() as session:
            statement = insert(Repository).values(id=str(uuid4()), url=body.url, created_at=now())
            session.execute(statement.on_duplicate_key_update(url=statement.inserted.url))
            return session.scalar(select(Repository).where(Repository.url == body.url))

    @app.get("/repositories", response_model=list[RepoView], dependencies=protected)
    def repositories(limit: Annotated[int, Query(ge=1, le=100)] = 50):
        with factory() as session:
            return list(
                session.scalars(select(Repository).order_by(Repository.created_at.desc()).limit(limit))
            )

    @app.post("/jobs", response_model=JobView, status_code=202, dependencies=protected)
    def submit(body: Issue):
        with factory.begin() as session:
            repository = session.get(Repository, str(body.repository_id))
            if repository is None:
                raise HTTPException(404, "Repository not found")
            demo = (
                repository.url == DEMO_REPO and body.commit == DEMO_COMMIT and body.issue == DEMO_ISSUE
            )
            if body.model_mode == "mock" and not demo:
                raise HTTPException(422, "Mock is available for the scripted public demo only")
            if body.profile == "demo" and not demo:
                raise HTTPException(422, "Demo profile requires the pinned example issue")
            if body.model_mode == "lora" and not settings.lora_model:
                raise HTTPException(422, "LoRA served ID is not configured")
            job = Job(
                id=str(uuid4()),
                **body.model_dump(exclude={"repository_id"}),
                repository_id=repository.id,
            )
            session.add(job)
            session.flush()
            event(session, job.id, "queued", model_mode=job.model_mode)
            return job

    @app.get("/jobs", response_model=list[JobView], dependencies=protected)
    def history(
        repository_id: UUID | None = None,
        status: str | None = None,
        limit: Annotated[int, Query(ge=1, le=100)] = 50,
        offset: Annotated[int, Query(ge=0, le=100000)] = 0,
    ):
        query = select(Job)
        if repository_id:
            query = query.where(Job.repository_id == str(repository_id))
        if status:
            query = query.where(Job.status == status)
        with factory() as session:
            return list(
                session.scalars(
                    query.order_by(Job.created_at.desc(), Job.id.desc()).offset(offset).limit(limit)
                )
            )

    @app.get("/jobs/{job_id}", response_model=JobView, dependencies=protected)
    def status(job_id: UUID):
        with factory() as session:
            return get_job(session, job_id)

    @app.get("/jobs/{job_id}/patch", response_class=PlainTextResponse, dependencies=protected)
    def patch(job_id: UUID):
        with factory() as session:
            job = get_job(session, job_id)
            if job.patch is None:
                raise HTTPException(404, "Patch unavailable")
            headers = {
                "X-Base-Commit": job.commit,
                "Cache-Control": "no-store",
                "X-Content-Type-Options": "nosniff",
            }
            if job.review_sha:
                headers["X-Patch-SHA256"] = job.review_sha
            return PlainTextResponse(job.patch, headers=headers)

    @app.post("/jobs/{job_id}/approval", status_code=202, response_model=JobView, dependencies=protected)
    def approve(job_id: UUID, body: Approval):
        with factory.begin() as session:
            job = get_job(session, job_id, lock=True)
            if (
                job.status != "awaiting_approval"
                or job.review_sha != body.patch_sha
                or job.approval is not None
            ):
                raise HTTPException(409, "No matching pending review")
            job.approval = body.approve
            job.attempts = 0
            job.status = "queued"
            event(session, job.id, "approval_recorded", approved=body.approve, patch_sha=body.patch_sha)
            return job

    @app.post("/jobs/{job_id}/retry", status_code=202, response_model=JobView, dependencies=protected)
    def retry(job_id: UUID):
        with factory.begin() as session:
            job = get_job(session, job_id, lock=True)
            if job.status != "failed":
                raise HTTPException(409, "Only failed jobs can be retried")
            job.status = "queued"
            job.error = None
            job.attempts = 0
            event(session, job.id, "retry_requested")
            return job

    @app.get("/jobs/{job_id}/events", dependencies=protected)
    async def stream(
        request: Request,
        job_id: UUID,
        after: Annotated[int, Query(ge=0, le=2**63 - 1)] = 0,
        last_event_id: Annotated[int | None, Header(ge=0, le=2**63 - 1)] = None,
    ):
        def read(cursor):
            with factory() as session:
                current_status = get_job(session, job_id).status
                events = list(
                    session.scalars(
                        select(Event)
                        .where(
                            Event.job_id == str(job_id),
                            Event.id > cursor,
                        )
                        .order_by(Event.id)
                        .limit(100)
                    )
                )
                return current_status, events

        cursor = last_event_id if last_event_id is not None else after
        first = await run_in_threadpool(read, cursor)

        async def generate():
            position = cursor
            current_status, events = first
            while not await request.is_disconnected():
                for entry in events:
                    position = entry.id
                    payload = json.dumps(
                        {"created_at": entry.created_at.isoformat() + "Z", **entry.details}
                    )
                    yield f"id: {entry.id}\nevent: {entry.kind}\ndata: {payload}\n\n"
                if len(events) < 100:
                    if current_status in TERMINAL:
                        return
                    if not events:
                        yield ": keepalive\n\n"
                    await asyncio.sleep(1)
                current_status, events = await run_in_threadpool(read, position)

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app
