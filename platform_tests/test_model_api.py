import json

import httpx
import pytest
from fastapi.testclient import TestClient

from patchpilot.api import create_app
from patchpilot.config import Settings, repo_url
from patchpilot.model import GemmaModel, ModelError
from patchpilot.rag import slices


def settings(**overrides):
    values = {
        "database_url": "mysql+pymysql://user:secret@127.0.0.1/db",
        "api_token": "test-token" * 3,
        "_env_file": None,
    }
    return Settings(**(values | overrides))


@pytest.mark.parametrize("mode,expected", [("base", "base-id"), ("lora", "adapter-id")])
def test_gemma_native_tool_call_and_exact_variant(mode, expected):
    def reply(request):
        if request.method == "GET":
            assert request.url.path == "/v1/models"
            return httpx.Response(200, json={"data": [{"id": "base-id"}, {"id": "adapter-id"}]})
        assert request.url.path == "/v1/chat/completions"
        body = json.loads(request.content)
        assert body["model"] == expected
        assert any(tool["function"]["name"] == "read_file" for tool in body["tools"])
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": "read_file",
                                        "arguments": '{"path":"module.py"}',
                                    },
                                }
                            ]
                        }
                    }
                ]
            },
        )

    model = GemmaModel(
        settings(base_model="base-id", lora_model="adapter-id"),
        mode,
        transport=httpx.MockTransport(reply),
    )
    try:
        assert model.decide("issue", "context", [], 0).validated().path == "module.py"
    finally:
        model.close()


def test_remote_model_and_redirect_are_rejected():
    with pytest.raises(ModelError):
        GemmaModel(settings(model_url="https://example.com/v1"), "base")
    model = GemmaModel(
        settings(),
        "base",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(302, headers={"Location": "https://example.com/v1/models"})
        ),
    )
    try:
        with pytest.raises(ModelError, match="request failed"):
            model.decide("issue", "", [], 0)
    finally:
        model.close()


@pytest.mark.parametrize(
    "url",
    [
        "https://user@github.com/a/b",
        "https://github.com/a/b/extra",
        "file:///tmp/source",
        "https://github.com/a/..",
        "https://github.com.evil.test/a/b",
    ],
)
def test_import_url_is_constrained(url):
    with pytest.raises(ValueError):
        repo_url(url)


def test_api_docs_auth_and_request_validation():
    with TestClient(create_app(settings())) as client:
        assert client.get("/docs").status_code == 200
        assert client.get("/jobs").status_code == 401
        token = settings().api_token.get_secret_value()
        response = client.post(
            "/repositories",
            headers={"Authorization": f"Bearer {token}"},
            json={"url": "http://localhost/a/b"},
        )
        assert response.status_code == 422


def test_ast_slices_keep_source_coordinates():
    source = "import math\n\n@cached\ndef solve(x):\n    return x + 1\n\nclass Tool:\n    pass\n"
    chunks = list(slices("repair.py", source))
    by_symbol = {part.symbol: part for part in chunks}
    assert (by_symbol["solve"].start_line, by_symbol["solve"].end_line) == (3, 5)
    assert by_symbol["Tool"].content.startswith("class Tool")
    assert any("import math" in part.content for part in chunks)
