"""Remote ML client for optional self-hosted acceleration."""

from __future__ import annotations

from typing import Any

import httpx

from find_api.core.config import settings


class RemoteMLClient:
    """HTTP client for calling a self-hosted remote ML service."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 60.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.base_url = (base_url or settings.REMOTE_ML_URL or "").rstrip("/")
        self.api_key = api_key or settings.REMOTE_ML_API_KEY
        self.timeout = timeout
        self.transport = transport

        if not self.base_url:
            raise ValueError("Remote ML base URL is required")

        if not self.api_key:
            raise ValueError("Remote ML API key is required")

    @property
    def headers(self) -> dict[str, str]:
        """Return bearer auth headers for remote inference requests."""
        return {"Authorization": f"Bearer {self.api_key}"}

    def _client(self) -> httpx.AsyncClient:
        """Create an async HTTP client."""
        return httpx.AsyncClient(timeout=self.timeout, transport=self.transport)

    async def health(self) -> dict[str, Any]:
        """Check remote ML service health."""
        async with self._client() as client:
            response = await client.get(
                f"{self.base_url}/api/ml/health",
                headers=self.headers,
            )
            response.raise_for_status()
            return response.json()

    async def analyze(self, image_bytes: bytes) -> dict[str, Any]:
        """Request remote image analysis."""
        async with self._client() as client:
            response = await client.post(
                f"{self.base_url}/api/ml/analyze",
                headers=self.headers,
                files={"image": ("image", image_bytes, "application/octet-stream")},
            )
            response.raise_for_status()
            return response.json()

    async def embed(self, image_bytes: bytes) -> list[float]:
        """Request remote image embedding."""
        async with self._client() as client:
            response = await client.post(
                f"{self.base_url}/api/ml/embed",
                headers=self.headers,
                files={"image": ("image", image_bytes, "application/octet-stream")},
            )
            response.raise_for_status()
            data = response.json()
            return data["embedding"]

    async def cluster(self, embeddings: list[list[float]]) -> dict[str, Any]:
        """Request remote clustering from embeddings."""
        async with self._client() as client:
            response = await client.post(
                f"{self.base_url}/api/ml/cluster",
                headers=self.headers,
                json={"embeddings": embeddings},
            )
            response.raise_for_status()
            return response.json()


def get_remote_ml_client() -> RemoteMLClient:
    """Create a remote ML client using application settings."""
    return RemoteMLClient()
