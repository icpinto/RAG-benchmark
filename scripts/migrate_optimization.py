"""Add optimizer application/checkpoint tables without altering historical runs.

Run once with API/worker database environment: python -m scripts.migrate_optimization
"""

from ragbench_core.engine.database import engine, init_db


def main():
    from langgraph.checkpoint.postgres import PostgresSaver

    init_db()
    uri = engine.url.set(drivername="postgresql").render_as_string(hide_password=False)
    with PostgresSaver.from_conn_string(uri) as saver:
        saver.setup()


if __name__ == "__main__":
    main()
