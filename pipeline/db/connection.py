"""Database connection management using psycopg3."""

import os
from contextlib import contextmanager

import psycopg
from dotenv import load_dotenv

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
