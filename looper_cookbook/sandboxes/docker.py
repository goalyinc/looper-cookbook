"""A sandbox in a container of this machine's Docker daemon.

``DockerSandbox`` starts one container, runs each command in it with
``docker exec``, and removes it when it closes. It drives the ``docker``
program, so it needs Docker installed and nothing else.

What keeps a model's code in its box:

- the container has no network unless ``network`` is set;
- ``memory_mb`` and ``cpus`` bound what one sandbox takes from the machine;
- a command's deadline is enforced inside the container, by ``timeout``,
  which kills the command and what it started;
- the container removes itself ``lifetime_s`` after it starts, so one whose
  owner crashed does not stay.

The image needs ``sh``, ``timeout``, ``cat`` and ``mkdir``, which Debian's
and Alpine's images have.
"""

from __future__ import annotations

import dataclasses
import logging
import math
import time
import uuid
from collections.abc import Sequence

from looper_cookbook.sandboxes.base import CommandResult, Sandbox, SandboxError
from looper_cookbook.sandboxes.process import run_process

__all__ = ["DockerSandbox"]

logger = logging.getLogger(__name__)

# Starting a container may pull its image first.
_START_TIMEOUT_S = 600.0
_REMOVE_TIMEOUT_S = 30.0
# What `docker exec` may take beyond its command's own deadline before it counts as hung.
_CLIENT_GRACE_S = 10.0
# How Docker's own failures start on stderr, as against the command's.
_DOCKER_ERRORS = (
    "Error response from daemon:",
    "Cannot connect to the Docker daemon",
    "OCI runtime exec failed:",
)
# Statuses `timeout -s KILL` exits with at the deadline: GNU's, and a shell's for a killed child.
_KILLED = (124, 137)


class DockerSandbox(Sandbox):
    """Commands in one container, removed when the sandbox closes.

    Attributes:
        name: The container's name, as ``docker ps`` lists it.
    """

    @dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
    class Config:
        """The container a sandbox runs in.

        Attributes:
            image: The image to run, pulled if this machine lacks it.
            network: Whether commands reach the network. Off, a model's code
                can fetch nothing and send nothing.
            memory_mb: The memory the container may use, in MiB; a command
                that passes it is killed. ``None`` sets no limit.
            cpus: The CPUs the container may use, as Docker counts them:
                1.5 is one and a half. ``None`` sets no limit.
            lifetime_s: Seconds after which the container removes itself,
                closed or not. It must outlast the episode that holds it.
        """

        image: str
        network: bool = False
        memory_mb: int | None = None
        cpus: float | None = None
        lifetime_s: int = 3600

        def __post_init__(self) -> None:
            if not self.image.strip():
                raise ValueError("image must name a Docker image, such as python:3.12-slim")
            if self.memory_mb is not None and self.memory_mb < 1:
                raise ValueError(f"memory_mb must be at least 1 MiB, not {self.memory_mb}")
            if self.cpus is not None and not (math.isfinite(self.cpus) and self.cpus > 0):
                raise ValueError(f"cpus must be a positive number, not {self.cpus}")
            if self.lifetime_s < 1:
                raise ValueError(f"lifetime_s must be at least 1 second, not {self.lifetime_s}")

        async def build(self) -> DockerSandbox:
            name = f"looper-sandbox-{uuid.uuid4().hex[:12]}"
            argv = ["docker", "run", "--detach", "--rm", "--init", "--name", name]
            argv += ["--workdir", "/workspace"]
            if not self.network:
                argv += ["--network", "none"]
            if self.memory_mb is not None:
                argv += ["--memory", f"{self.memory_mb}m"]
            if self.cpus is not None:
                argv += ["--cpus", f"{self.cpus:g}"]
            argv += [self.image, "sleep", str(self.lifetime_s)]
            try:
                result = await _docker(argv, timeout_s=_START_TIMEOUT_S)
            except BaseException:
                # The daemon may have made the container before the client was interrupted.
                await _remove(name)
                raise
            if result.exit_code != 0:
                await _remove(name)
                why = result.stderr.strip() or f"no answer in {_START_TIMEOUT_S:g} s"
                raise SandboxError(
                    f"could not start a container of {self.image}: {why}. Check that Docker "
                    "is running and that `docker pull` gets the image",
                    retryable=False,
                )
            return DockerSandbox(name)

    def __init__(self, name: str) -> None:
        self.name = name
        self._closed = False

    async def run(
        self, command: str, *, timeout_s: float, stdin: str | None = None
    ) -> CommandResult:
        if self._closed:
            raise SandboxError("the sandbox is closed, so it runs no command", retryable=False)
        argv = ["docker", "exec"]
        if stdin is not None:
            argv.append("--interactive")
        # The deadline is kept inside the container: killing `docker exec` would leave the command.
        deadline = f"{timeout_s:.3f}".rstrip("0").rstrip(".")
        argv += [self.name, "timeout", "-s", "KILL", deadline, "sh", "-c", command]
        started = time.monotonic()
        result = await _docker(argv, timeout_s=timeout_s + _CLIENT_GRACE_S, stdin=stdin)
        if result.exit_code is None:
            await self.close()
            raise SandboxError(
                f"`docker exec` in {self.name} did not return {_CLIENT_GRACE_S:g} s after its "
                "command's deadline, so the container was removed",
                retryable=True,
            )
        if result.exit_code != 0 and result.stderr.startswith(_DOCKER_ERRORS):
            gone = "is not running" in result.stderr or "No such container" in result.stderr
            hint = " The container is gone: it was killed, or its lifetime_s ended" if gone else ""
            raise SandboxError(
                f"Docker could not run a command in {self.name}: {result.stderr.strip()}.{hint}",
                retryable=gone,
            )
        # A command killed for its memory exits the same way, but before its deadline.
        if result.exit_code in _KILLED and time.monotonic() - started >= timeout_s:
            return dataclasses.replace(result, exit_code=None)
        return result

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        await _remove(self.name)


async def _docker(
    argv: Sequence[str], *, timeout_s: float, stdin: str | None = None
) -> CommandResult:
    try:
        return await run_process(argv, timeout_s=timeout_s, stdin=stdin)
    except FileNotFoundError:
        raise SandboxError(
            "the docker program is not on PATH: install Docker, or use a LocalSandbox for "
            "code you trust",
            retryable=False,
        ) from None


async def _remove(name: str) -> None:
    """Removes the container if it exists. A failure is logged, never raised:
    it must not hide the error that led here."""
    try:
        result = await _docker(("docker", "rm", "--force", name), timeout_s=_REMOVE_TIMEOUT_S)
    except SandboxError:
        return
    if result.exit_code != 0 and "No such container" not in result.stderr:
        logger.warning(
            "could not remove the container %s: %s. It removes itself when its lifetime_s "
            "ends; to remove it now, run `docker rm --force %s`",
            name,
            result.stderr.strip() or "docker rm did not answer",
            name,
        )
