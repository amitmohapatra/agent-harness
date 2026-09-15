"""Deterministic pipeline ordering (§19/§20)."""

from __future__ import annotations

from universal_agent_harness import AgentResult, BaseInterceptor, Order
from universal_agent_harness.contracts.errors import AgentError
from universal_agent_harness.interceptors.base import InterceptorChain


class Recorder(BaseInterceptor):
    def __init__(self, name: str, order: int, log: list[str]) -> None:
        self.name = name
        self.order = order
        self.log = log

    async def before(self, request, runtime):
        self.log.append(f"before:{self.name}")
        return request

    async def after(self, result, runtime):
        self.log.append(f"after:{self.name}")
        return result

    async def on_error(self, error, runtime):
        self.log.append(f"error:{self.name}")


async def test_before_ascends_and_after_descends():
    log: list[str] = []
    chain = InterceptorChain(
        [Recorder("c", 30, log), Recorder("a", 10, log), Recorder("b", 20, log)]
    )
    assert chain.names == ["a", "b", "c"]
    await chain.before(None, None)
    await chain.after(AgentResult.ok(), None)
    assert log == [
        "before:a", "before:b", "before:c",
        "after:c", "after:b", "after:a",
    ]


async def test_ordering_is_stable_for_equal_orders():
    log: list[str] = []
    chain = InterceptorChain([Recorder("z", 10, log), Recorder("a", 10, log)])
    assert chain.names == ["a", "z"]


async def test_error_path_lets_one_interceptor_recover():
    class Recovering(BaseInterceptor):
        name, order = "recover", Order.USER

        async def on_error(self, error, runtime):
            return AgentResult.ok("fallback")

    log: list[str] = []
    chain = InterceptorChain([Recovering(), Recorder("t", 10, log)])
    result = await chain.on_error(AgentError(code="X"), None)
    assert result is not None and result.data == "fallback"
    assert log == ["error:t"]


async def test_chain_extension_keeps_order():
    log: list[str] = []
    chain = InterceptorChain([Recorder("core", 10, log)])
    extended = chain.with_extra([Recorder("extra", 5, log)])
    assert extended.names == ["extra", "core"]
    assert chain.names == ["core"]  # the original is untouched
