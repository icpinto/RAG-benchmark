from ragbench_core.utils.clients import close_client


class CloseClient:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class Transport:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class TransportClient:
    def __init__(self):
        self.transport = Transport()


def test_close_client_prefers_client_close():
    client = CloseClient()

    close_client(client)

    assert client.closed is True


def test_close_client_falls_back_to_transport_close():
    client = TransportClient()

    close_client(client)

    assert client.transport.closed is True


def test_token_usage_accumulates_and_resets():
    from ragbench_core.utils.clients import consume_token_usage, record_token_usage

    consume_token_usage()
    record_token_usage(embedding_tokens=3, prompt_tokens="4", completion_tokens=None)
    record_token_usage(embedding_tokens=2)

    assert consume_token_usage()["embedding_tokens"] == 5
    assert consume_token_usage()["embedding_tokens"] == 0
