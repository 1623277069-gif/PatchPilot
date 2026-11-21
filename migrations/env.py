from alembic import context

from patchpilot.config import Settings
from patchpilot.db import Base, make_engine

settings = Settings()
if context.is_offline_mode():
    context.configure(dialect_name="mysql", target_metadata=Base.metadata, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()
else:
    engine = make_engine(settings)
    with engine.connect() as connection:
        context.configure(connection=connection, target_metadata=Base.metadata)
        with context.begin_transaction():
            context.run_migrations()
    engine.dispose()
