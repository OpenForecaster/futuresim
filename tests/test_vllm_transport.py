import requests

from inference.vllm import VLLMChatTimeoutError, VLLMInference, VLLMTransportError


class FakeSession:
    def __init__(self, exc):
        self.exc = exc
        self.calls = 0

    def post(self, *args, **kwargs):
        self.calls += 1
        raise self.exc


def _make_inference(fake_session):
    inference = VLLMInference(model_path="/tmp/fake-model", timeout=1)
    inference._ensure_server = lambda: None
    inference._get_session = lambda: fake_session
    inference._port = 8001
    return inference


def test_chat_timeout_raises_transport_error():
    session = FakeSession(requests.exceptions.Timeout())
    inference = _make_inference(session)

    try:
        inference.chat([{"role": "user", "content": "ping"}], {"max_tokens": 1})
        assert False, "Expected timeout error"
    except VLLMChatTimeoutError as exc:
        assert "timeout after 1s" in str(exc)

    assert session.calls == 3


def test_chat_connection_error_raises_transport_error():
    session = FakeSession(requests.exceptions.ConnectionError())
    inference = _make_inference(session)

    try:
        inference.chat([{"role": "user", "content": "ping"}], {"max_tokens": 1})
        assert False, "Expected connection error"
    except VLLMTransportError as exc:
        assert "connection error" in str(exc)

    assert session.calls == 3
