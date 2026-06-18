import httpx
import pytest

from find_api.ml.remote_client import RemoteMLClient


@pytest.mark.asyncio
async def test_remote_ml_client_sends_bearer_token_for_health():
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"status": "ok"})

    transport = httpx.MockTransport(handler)

    client = RemoteMLClient(
        base_url="http://remote-ml.test",
        api_key="secret-token",
        transport=transport,
    )

    response = await client.health()

    assert response == {"status": "ok"}
    assert requests[0].headers["Authorization"] == "Bearer secret-token"
    assert str(requests[0].url) == "http://remote-ml.test/api/ml/health"


@pytest.mark.asyncio
async def test_remote_ml_client_posts_image_for_analyze():
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"caption": "test image", "objects": []})

    transport = httpx.MockTransport(handler)

    client = RemoteMLClient(
        base_url="http://remote-ml.test",
        api_key="secret-token",
        transport=transport,
    )

    response = await client.analyze(b"fake-image-bytes")

    assert response == {"caption": "test image", "objects": []}
    assert requests[0].headers["Authorization"] == "Bearer secret-token"
    assert str(requests[0].url) == "http://remote-ml.test/api/ml/analyze"


@pytest.mark.asyncio
async def test_remote_ml_client_returns_embedding():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"embedding": [0.1, 0.2, 0.3]})

    transport = httpx.MockTransport(handler)

    client = RemoteMLClient(
        base_url="http://remote-ml.test",
        api_key="secret-token",
        transport=transport,
    )

    embedding = await client.embed(b"fake-image-bytes")

    assert embedding == [0.1, 0.2, 0.3]


@pytest.mark.asyncio
async def test_remote_ml_client_posts_embeddings_for_cluster():
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"clusters": [[0, 1]]})

    transport = httpx.MockTransport(handler)

    client = RemoteMLClient(
        base_url="http://remote-ml.test",
        api_key="secret-token",
        transport=transport,
    )

    response = await client.cluster([[0.1, 0.2], [0.2, 0.3]])

    assert response == {"clusters": [[0, 1]]}
    assert requests[0].headers["Authorization"] == "Bearer secret-token"
    assert str(requests[0].url) == "http://remote-ml.test/api/ml/cluster"


def test_remote_ml_client_requires_base_url():
    with pytest.raises(ValueError, match="Remote ML base URL is required"):
        RemoteMLClient(base_url="", api_key="secret-token")


def test_remote_ml_client_requires_api_key():
    with pytest.raises(ValueError, match="Remote ML API key is required"):
        RemoteMLClient(base_url="http://remote-ml.test", api_key="")
