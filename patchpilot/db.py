"""MySQL business tables and transaction helpers."""

from datetime import UTC, datetime

from sqlalchemy import (
    JSON,
    BigInteger,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
    create_engine,
)
from sqlalchemy.dialects.mysql import MEDIUMTEXT
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from .config import Settings

TERMINAL = frozenset({"succeeded", "unresolved", "rejected", "failed"})


def now():
    return datetime.now(UTC).replace(tzinfo=None)


class Base(DeclarativeBase):
    pass


class Repository(Base):
    __tablename__ = "repositories"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    url: Mapped[str] = mapped_column(String(200), unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now)


class Job(Base):
    __tablename__ = "jobs"
    __table_args__ = (Index("ix_jobs_status_created", "status", "created_at", "id"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    repository_id: Mapped[str] = mapped_column(ForeignKey("repositories.id"))
    commit: Mapped[str] = mapped_column(String(40))
    issue: Mapped[str] = mapped_column(MEDIUMTEXT)
    model_mode: Mapped[str] = mapped_column(String(12))
    profile: Mapped[str] = mapped_column(String(16))
    max_steps: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(24), default="queued")
    claim_token: Mapped[str | None] = mapped_column(String(36))
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    patch: Mapped[str | None] = mapped_column(MEDIUMTEXT)
    review_sha: Mapped[str | None] = mapped_column(String(64))
    approval: Mapped[bool | None]
    result: Mapped[dict | None] = mapped_column(JSON)
    error: Mapped[str | None] = mapped_column(String(160))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=now, onupdate=now)


class Event(Base):
    __tablename__ = "job_events"
    __table_args__ = (Index("ix_job_events_cursor", "job_id", "id"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id"))
    kind: Mapped[str] = mapped_column(String(40))
    details: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now)


class Snapshot(Base):
    __tablename__ = "index_snapshots"
    __table_args__ = (UniqueConstraint("repository_id", "commit", "embedding"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    repository_id: Mapped[str] = mapped_column(ForeignKey("repositories.id"))
    commit: Mapped[str] = mapped_column(String(40))
    embedding: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now)


class CodeChunk(Base):
    __tablename__ = "code_chunks"
    __table_args__ = (Index("ix_chunks_snapshot_blob", "snapshot_id", "blob"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    snapshot_id: Mapped[str] = mapped_column(ForeignKey("index_snapshots.id", ondelete="CASCADE"))
    path: Mapped[str] = mapped_column(String(512))
    blob: Mapped[str] = mapped_column(String(40))
    symbol: Mapped[str] = mapped_column(String(512))
    start_line: Mapped[int] = mapped_column(Integer)
    end_line: Mapped[int] = mapped_column(Integer)
    content: Mapped[str] = mapped_column(MEDIUMTEXT)
    vector_id: Mapped[int | None] = mapped_column(Integer)


def make_engine(settings: Settings):
    return create_engine(
        settings.database_url.get_secret_value(),
        pool_pre_ping=True,
        hide_parameters=True,
        isolation_level="READ COMMITTED",
        pool_recycle=1800,
        connect_args={"connect_timeout": 5, "read_timeout": 15, "write_timeout": 15},
    )


def sessions(engine):
    return sessionmaker(bind=engine, expire_on_commit=False)


def event(session, job_id: str, kind: str, **details):
    session.add(Event(job_id=job_id, kind=kind, details=details))
