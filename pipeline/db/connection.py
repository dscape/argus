"""Database connection management using psycopg3."""

import glob
import logging
import os
from contextlib import contextmanager

import psycopg
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

load_dotenv()

DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://argus:argus_dev@localhost:5432/argus"
)


def get_connection(**kwargs):
    """Get a new database connection."""
    return psycopg.connect(DATABASE_URL, **kwargs)


@contextmanager
def get_cursor(*, autocommit=False):
    """Context manager that yields a cursor and handles commit/rollback."""
    conn = get_connection(autocommit=autocommit)
    try:
        with conn:
            with conn.cursor() as cur:
                yield cur
    finally:
        conn.close()


@contextmanager
def get_conn():
    """Context manager that yields a connection."""
    conn = get_connection()
    try:
        yield conn
    finally:
        conn.close()


def init_schema():
    """Drop all tables and apply schema.sql from scratch."""
    schema_path = os.path.join(os.path.dirname(__file__), "schema.sql")
    with open(schema_path) as f:
        sql = f.read()
    with get_connection(autocommit=True) as conn:
        conn.execute("DROP SCHEMA public CASCADE; CREATE SCHEMA public;")
        conn.execute(sql)
    print("Schema applied successfully.")


def migrate():
    """Run pending SQL migrations from pipeline/db/migrations/.

    Tracks applied migrations in a `schema_migrations` table.
    Safe to call on every startup — already-applied migrations are skipped.
    """
    migrations_dir = os.path.join(os.path.dirname(__file__), "migrations")
    if not os.path.isdir(migrations_dir):
        return

    files = sorted(glob.glob(os.path.join(migrations_dir, "*.sql")))
    if not files:
        return

    with get_connection(autocommit=True) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                filename TEXT PRIMARY KEY,
                applied_at TIMESTAMPTZ DEFAULT now()
            )
            """
        )
        cur = conn.execute("SELECT filename FROM schema_migrations")
        applied = {row[0] for row in cur.fetchall()}

        for path in files:
            name = os.path.basename(path)
            if name in applied:
                continue
            logger.info("Applying migration: %s", name)
            with open(path) as f:
                sql = f.read()
            conn.execute(sql)
            conn.execute(
                "INSERT INTO schema_migrations (filename) VALUES (%s)", (name,)
            )
            logger.info("Migration applied: %s", name)
