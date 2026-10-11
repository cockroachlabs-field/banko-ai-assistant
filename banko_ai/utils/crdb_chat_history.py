"""CockroachDB-backed chat message history using langchain-cockroachdb.

Provides persistent conversation storage per session/thread so chat
history survives server restarts and browser refreshes.
"""


from langchain_cockroachdb import CockroachDBChatMessageHistory

from .crdb_engine import normalize_crdb_url
from .db_retry import get_database_url


def get_chat_history(
    session_id: str,
    database_url: str | None = None,
    table_name: str = "chat_message_store",
) -> CockroachDBChatMessageHistory:
    """Return a CockroachDBChatMessageHistory for the given session.

    The table is created automatically on first use.

    Args:
        session_id: Unique session / thread identifier.
        database_url: CockroachDB connection string (falls back to env).
        table_name: Table name for messages.
    """
    url = normalize_crdb_url(get_database_url(database_url))

    history = CockroachDBChatMessageHistory(
        session_id=session_id,
        connection_string=url,
        table_name=table_name,
    )
    history.create_table_if_not_exists()
    return history
