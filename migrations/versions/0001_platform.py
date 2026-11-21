"""Create repair jobs, events and versioned code indexes."""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.mysql import MEDIUMTEXT

revision = "0001_platform"
down_revision = None


def upgrade():
    op.create_table(
        "repositories",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("url", sa.String(200), nullable=False, unique=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        mysql_charset="utf8mb4",
    )
    op.create_table(
        "jobs",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("repository_id", sa.String(36), sa.ForeignKey("repositories.id"), nullable=False),
        sa.Column("commit", sa.String(40), nullable=False),
        sa.Column("issue", MEDIUMTEXT(), nullable=False),
        sa.Column("model_mode", sa.String(12), nullable=False),
        sa.Column("profile", sa.String(16), nullable=False),
        sa.Column("max_steps", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(24), nullable=False),
        sa.Column("claim_token", sa.String(36)),
        sa.Column("attempts", sa.Integer(), nullable=False),
        sa.Column("patch", MEDIUMTEXT()),
        sa.Column("review_sha", sa.String(64)),
        sa.Column("approval", sa.Boolean()),
        sa.Column("result", sa.JSON()),
        sa.Column("error", sa.String(160)),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        mysql_charset="utf8mb4",
    )
    op.create_index("ix_jobs_status_created", "jobs", ["status", "created_at", "id"])
    op.create_table(
        "job_events",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("job_id", sa.String(36), sa.ForeignKey("jobs.id"), nullable=False),
        sa.Column("kind", sa.String(40), nullable=False),
        sa.Column("details", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        mysql_charset="utf8mb4",
    )
    op.create_index("ix_job_events_cursor", "job_events", ["job_id", "id"])
    op.create_table(
        "index_snapshots",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("repository_id", sa.String(36), sa.ForeignKey("repositories.id"), nullable=False),
        sa.Column("commit", sa.String(40), nullable=False),
        sa.Column("embedding", sa.String(64), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.UniqueConstraint("repository_id", "commit", "embedding"),
        mysql_charset="utf8mb4",
    )
    op.create_table(
        "code_chunks",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column(
            "snapshot_id",
            sa.String(36),
            sa.ForeignKey("index_snapshots.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("path", sa.String(512), nullable=False),
        sa.Column("blob", sa.String(40), nullable=False),
        sa.Column("symbol", sa.String(512), nullable=False),
        sa.Column("start_line", sa.Integer(), nullable=False),
        sa.Column("end_line", sa.Integer(), nullable=False),
        sa.Column("content", MEDIUMTEXT(), nullable=False),
        sa.Column("vector_id", sa.Integer()),
        mysql_charset="utf8mb4",
    )
    op.create_index("ix_chunks_snapshot_blob", "code_chunks", ["snapshot_id", "blob"])


def downgrade():
    for table in ("code_chunks", "index_snapshots", "job_events", "jobs", "repositories"):
        op.drop_table(table)
