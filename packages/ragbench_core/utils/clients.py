from __future__ import annotations


def close_client(client) -> None:
    close = getattr(client, "close", None)
    if callable(close):
        close()
        return
    transport = getattr(client, "transport", None)
    transport_close = getattr(transport, "close", None)
    if callable(transport_close):
        transport_close()
