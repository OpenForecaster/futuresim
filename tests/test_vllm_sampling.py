from inference.vllm import VLLMInference


class _FakeResponse:
    def __init__(self, payload):
        self.status_code = 200
        self._payload = payload

    def json(self):
        return self._payload


class _FakeSession:
    def __init__(self, payload):
        self.payload = payload
        self.last_json = None

    def post(self, url, json, timeout):
        del url, timeout
        self.last_json = json
        return _FakeResponse(self.payload)


def test_vllm_chat_forwards_top_p_and_top_k():
    session = _FakeSession(
        {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
    )
    inference = VLLMInference(model_path="dummy-model")
    inference._ensure_server = lambda: None
    inference._get_session = lambda: session
    inference._port = 8001

    response, usage = inference.chat(
        [{"role": "user", "content": "hello"}],
        {"temperature": 0.7, "max_tokens": 32, "top_p": 0.95, "top_k": 20},
    )

    assert response == "ok"
    assert usage["prompt_tokens"] == 1
    assert session.last_json["top_p"] == 0.95
    assert session.last_json["top_k"] == 20


def test_vllm_chat_json_forwards_top_p_and_top_k():
    session = _FakeSession({"choices": [], "usage": {}})
    inference = VLLMInference(model_path="dummy-model")
    inference._ensure_server = lambda: None
    inference._get_session = lambda: session
    inference._port = 8001

    payload = inference.chat_json(
        [{"role": "user", "content": "hello"}],
        {"temperature": 0.7, "max_tokens": 32, "top_p": 0.95, "top_k": 20},
    )

    assert payload == {"choices": [], "usage": {}}
    assert session.last_json["top_p"] == 0.95
    assert session.last_json["top_k"] == 20
