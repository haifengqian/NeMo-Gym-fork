# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio

import pytest

from environment_servers.single_agent_turn.tests.test_app import _Client, _environment_server, _request, _Response


@pytest.mark.parametrize("verify_error", [False, True])
async def test_finish_verify_close_order_and_final_cookies(monkeypatch, verify_error: bool) -> None:
    environment, client = _environment_server()
    environment.config.finish_agent_before_verification = True
    original = _Client.post
    service_alive = False

    async def post(self, server_name, url_path, **kwargs):
        nonlocal service_alive
        if url_path == "/v1/agent_sessions":
            service_alive = True
        elif url_path == "/v1/agent_sessions/finish":
            assert service_alive
            response = await original(self, server_name, "/v1/agent_sessions/close", **kwargs)
            self.calls[-1] = (server_name, url_path, kwargs)
            return response
        elif url_path == "/verify":
            assert service_alive
            assert kwargs["cookies"] == {"session": "updated-cookie"}
            if verify_error:
                self.calls.append((server_name, url_path, kwargs))
                self.responses.pop(0)
                raise RuntimeError("verifier failed")
        elif url_path == "/v1/agent_sessions/close":
            service_alive = False
            self.calls.append((server_name, url_path, kwargs))
            return _Response({"agent_session_id": kwargs["json"]["agent_session_id"]})
        elif url_path == "/close_session":
            assert not service_alive
        return await original(self, server_name, url_path, **kwargs)

    monkeypatch.setattr(_Client, "post", post)
    result = await environment.run_request(_request())
    assert [path for _, path, _ in client.calls][-4:] == [
        "/v1/agent_sessions/finish",
        "/verify",
        "/v1/agent_sessions/close",
        "/close_session",
    ]
    assert not service_alive
    if verify_error:
        assert result.failure.stage == "verification"
    else:
        assert result.result.reward == 1
    assert not client.responses


async def test_failed_final_agent_close_retries_before_resource_destruction(monkeypatch) -> None:
    environment, client = _environment_server()
    environment.config.finish_agent_before_verification = True
    original = _Client.post
    close_attempts = 0

    async def post(self, server_name, url_path, **kwargs):
        nonlocal close_attempts
        if url_path == "/v1/agent_sessions/finish":
            response = await original(self, server_name, "/v1/agent_sessions/close", **kwargs)
            self.calls[-1] = (server_name, url_path, kwargs)
            return response
        if url_path == "/v1/agent_sessions/close":
            close_attempts += 1
            self.calls.append((server_name, url_path, kwargs))
            if close_attempts == 1:
                raise RuntimeError("transient descendant cleanup failure")
            return _Response({"agent_session_id": kwargs["json"]["agent_session_id"]})
        if url_path == "/close_session":
            assert close_attempts == 2
        return await original(self, server_name, url_path, **kwargs)

    monkeypatch.setattr(_Client, "post", post)
    result = await environment.run_request(_request())
    assert result.result.reward == 1
    assert [path for _, path, _ in client.calls][-3:] == [
        "/v1/agent_sessions/close",
        "/v1/agent_sessions/close",
        "/close_session",
    ]
    assert not client.responses


@pytest.mark.parametrize("cancel", [False, True], ids=["episode-timeout", "caller-cancellation"])
async def test_interrupted_verification_closes_finished_agent_before_resources(monkeypatch, cancel: bool) -> None:
    environment, client = _environment_server()
    environment.config.finish_agent_before_verification = True
    environment.config.default_episode_timeout_seconds = 0.1
    original = _Client.post
    verifying = asyncio.Event()
    closed = False

    async def post(self, server_name, url_path, **kwargs):
        nonlocal closed
        if url_path == "/v1/agent_sessions/finish":
            response = await original(self, server_name, "/v1/agent_sessions/close", **kwargs)
            self.calls[-1] = (server_name, url_path, kwargs)
            return response
        if url_path == "/verify":
            self.calls.append((server_name, url_path, kwargs))
            self.responses.pop(0)
            verifying.set()
            await asyncio.Event().wait()
        if url_path == "/v1/agent_sessions/close":
            closed = True
            self.calls.append((server_name, url_path, kwargs))
            return _Response({"agent_session_id": kwargs["json"]["agent_session_id"]})
        if url_path == "/close_session":
            assert closed
        return await original(self, server_name, url_path, **kwargs)

    monkeypatch.setattr(_Client, "post", post)
    task = asyncio.create_task(environment.run_request(_request()))
    await asyncio.wait_for(verifying.wait(), timeout=1)
    if cancel:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        response = await asyncio.wait_for(task, timeout=1)
        assert response.failure.failure_reason == "Episode timed out"
    assert [path for _, path, _ in client.calls][-2:] == ["/v1/agent_sessions/close", "/close_session"]
    assert not client.responses


async def test_stalled_finish_is_bounded_and_closes_before_resources(monkeypatch: pytest.MonkeyPatch) -> None:
    environment, client = _environment_server()
    environment.config.finish_agent_before_verification = True
    environment.config.cleanup_timeout_seconds = 0.01
    # Final cleanup consumes the existing close receipt; verification never runs.
    client.responses.pop(4)
    original = _Client.post

    async def post(self, server_name: str, url_path: str, **kwargs) -> _Response:
        if url_path == "/v1/agent_sessions/finish":
            self.calls.append((server_name, url_path, kwargs))
            await asyncio.Event().wait()
        return await original(self, server_name, url_path, **kwargs)

    monkeypatch.setattr(_Client, "post", post)
    result = await asyncio.wait_for(environment.run_request(_request()), timeout=0.5)
    assert result.failure.stage == "cleanup"
    assert result.failure.terminal is False
    assert [path for _, path, _ in client.calls][-3:] == [
        "/v1/agent_sessions/finish",
        "/v1/agent_sessions/close",
        "/close_session",
    ]
    assert not client.responses
