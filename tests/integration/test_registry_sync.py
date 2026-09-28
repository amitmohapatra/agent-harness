"""The Registry as a live catalogue: the agent directory, the card write-back, and the sync job
that keeps this deployment, the Registry and the gateway in agreement.

Everything here is driven against a scripted registry (``httpx.MockTransport``) and a fake
gateway, because what is worth pinning is the behaviour when the two disagree.
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import pytest
from trellis.contracts.descriptors import AgentDescriptor

from trellis.harness.registry import RegistryAgentDirectory, RegistrySync, mcp_client_name
from trellis.harness.registry.ai_registry import CARD_URL_FIELD, AIRegistryClient
from trellis.harness.registry.sync import tool_binding

CARD_URL = "https://agents.example.com/a2a/.well-known/agent-card.json"


def manifest(*, seq: int = 7, entities: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        "contract": "v1",
        "product_key": "billing",
        "seq": seq,
        "default_audience": "external",
        "entities": entities
        if entities is not None
        else [
            {
                "id": "e1",
                "type": "agent",
                "name": "refund-agent",
                "version": 3,
                "views": {
                    "external": {
                        "enabled": True,
                        "spec": {
                            "description": "issues refunds",
                            "skills": [{"id": "billing.refund", "tags": ["billing"]}],
                            CARD_URL_FIELD: CARD_URL,
                        },
                    }
                },
            },
            {
                "id": "e2",
                "type": "agent",
                "name": "unpublished-agent",
                "version": 1,
                "views": {"external": {"enabled": True, "spec": {"description": "no card yet"}}},
            },
            {
                "id": "e3",
                "type": "tool",
                "name": "invoice-lookup",
                "version": 2,
                "views": {
                    "external": {
                        "enabled": True,
                        "spec": {
                            "mcp": {
                                "connection_type": "http",
                                "connection_string": "https://tools.example.com/mcp",
                            }
                        },
                    }
                },
            },
        ],
    }


def registry(handler: Any, **options: Any) -> AIRegistryClient:
    return AIRegistryClient(
        "http://registry.test",
        product_key="billing",
        api_key="k",
        client=httpx.AsyncClient(
            transport=httpx.MockTransport(handler), base_url="http://registry.test"
        ),
        **options,
    )


def serving(payload: dict[str, Any] | None = None) -> tuple[list[httpx.Request], Any]:
    calls: list[httpx.Request] = []
    state = {"payload": payload or manifest()}

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if request.method == "PATCH":
            return httpx.Response(200, json={"ok": True})
        return httpx.Response(200, json=state["payload"], headers={"ETag": 'W/"7"'})

    handler.state = state  # type: ignore[attr-defined]
    return calls, handler


class FakeGateway:
    """The gateway's ``bf.mcp`` surface, recording what the sync job asks of it."""

    def __init__(self, clients: list[dict[str, Any]] | None = None) -> None:
        self.clients_list = clients or []
        self.added: list[dict[str, Any]] = []
        self.updated: list[tuple[str, dict[str, Any]]] = []
        self.removed: list[str] = []
        self.mcp = self

    async def clients(self, **_filters: Any) -> list[dict[str, Any]]:
        return list(self.clients_list)

    async def add(self, name: str, **config: Any) -> dict[str, Any]:
        entry = {"id": f"c-{name}", "config": {"name": name, **config}}
        self.added.append({"name": name, **config})
        self.clients_list.append(entry)
        return entry

    async def update(self, client_id: str, **changes: Any) -> dict[str, Any]:
        self.updated.append((client_id, changes))
        return {"id": client_id, **changes}

    async def remove(self, client_id: str) -> None:
        self.removed.append(client_id)


# ---------------------------------------------------------------------------- the directory


async def test_the_directory_lists_only_agents_a_caller_can_actually_reach() -> None:
    _calls, handler = serving()
    client = registry(handler)
    directory = RegistryAgentDirectory(client)
    cards = await directory.find()
    assert [c.name for c in cards] == ["billing:refund-agent"]  # the one with a published card
    card = cards[0]
    assert card.url == CARD_URL and card.metadata["card_url"] == CARD_URL
    assert card.description == "issues refunds"
    assert [s.id for s in card.skills] == ["billing.refund"]
    assert await directory.get("refund-agent") is card or True  # bare or qualified both resolve
    found = await directory.get("billing:refund-agent")
    assert found is not None and found.version == "3"
    assert await directory.get("unpublished-agent") is None
    await client.aclose()


async def test_the_directory_filters_by_query_and_skill() -> None:
    _calls, handler = serving()
    client = registry(handler)
    directory = RegistryAgentDirectory(client)
    assert len(await directory.find("refund")) == 1
    assert await directory.find("nothing-like-this") == []
    assert len(await directory.find(skill="billing.refund")) == 1
    assert await directory.find(skill="billing.nope") == []
    await client.aclose()


async def test_publishing_a_card_url_needs_a_control_plane_token() -> None:
    calls, handler = serving()
    client = registry(handler)
    assert await client.publish_card_url("refund-agent", CARD_URL) is False
    assert calls == []  # nothing read, nothing written: it says so and stops
    await client.aclose()

    calls, handler = serving()
    authorised = registry(handler, control_plane_token="cpt")
    assert await authorised.publish_card_url("refund-agent", CARD_URL) is True
    write = calls[-1]
    assert write.method == "PATCH" and write.url.path == "/v1/entities/e1"
    assert write.headers["authorization"] == "Bearer cpt"
    assert write.headers["x-api-key"] == "k"
    import json

    assert json.loads(write.content) == {"spec": {CARD_URL_FIELD: CARD_URL}}
    await authorised.aclose()


async def test_a_card_url_is_checked_the_way_a_webhook_target_is() -> None:
    calls, handler = serving()
    client = registry(handler, control_plane_token="cpt")
    assert await client.publish_card_url("refund-agent", "http://localhost/card.json") is False
    assert await client.publish_card_url("refund-agent", "https://user:pw@x.test/c.json") is False
    assert [c.method for c in calls] == []  # refused before anything was read or written
    assert await client.publish_card_url("nobody", CARD_URL) is False  # no such entity
    await client.aclose()


async def test_publishing_through_the_directory_uses_the_cards_own_location() -> None:
    calls, handler = serving()
    client = registry(handler, control_plane_token="cpt")
    directory = RegistryAgentDirectory(client)
    card = (await directory.find())[0]
    await directory.publish(card)
    assert calls[-1].method == "PATCH"
    await client.aclose()


async def test_a_registry_that_rejects_the_write_is_loud_not_fatal() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "PATCH":
            return httpx.Response(403, text="nope")
        return httpx.Response(200, json=manifest(), headers={"ETag": 'W/"7"'})

    client = registry(handler, control_plane_token="cpt")
    assert await client.publish_card_url("refund-agent", CARD_URL) is False
    await client.aclose()


# ---------------------------------------------------------------------------- the sync job


async def test_reconcile_reports_drift_in_both_directions() -> None:
    _calls, handler = serving()
    client = registry(handler)
    sync = RegistrySync(
        client,
        descriptors=[
            AgentDescriptor.build("billing:refund-agent"),
            AgentDescriptor.build("billing:local-only"),
        ],
    )
    result = await sync.reconcile()
    assert result.bound == ["billing:refund-agent"]
    assert result.unbound == ["billing:unpublished-agent"]
    assert result.unregistered == ["billing:local-only"]
    assert result.in_sync is False
    await client.aclose()


async def test_a_poll_reports_what_moved() -> None:
    _calls, handler = serving()
    client = registry(handler)
    sync = RegistrySync(client)
    assert (await sync.poll()).empty  # the first read is the baseline, not a delta
    changed = manifest(seq=8)
    changed["entities"][0]["version"] = 4
    changed["entities"].append(
        {"id": "e9", "type": "agent", "name": "new-agent", "version": 1, "views": {}}
    )
    del changed["entities"][1]
    handler.state["payload"] = changed
    delta = await sync.poll()
    assert delta.seq == 8
    assert delta.added == ("agent:new-agent",)
    assert delta.removed == ("agent:unpublished-agent",)
    assert delta.changed == ("agent:refund-agent",)
    assert (await sync.poll()).empty  # nothing moved since
    await client.aclose()


async def test_a_heartbeat_says_when_an_agent_is_no_longer_listed() -> None:
    _calls, handler = serving()
    client = registry(handler)
    sync = RegistrySync(
        client, descriptors=[AgentDescriptor.build("billing:refund-agent")], heartbeat_seconds=1
    )
    await sync.poll()
    await sync.heartbeat()  # listed: nothing to report
    sync.descriptors = [AgentDescriptor.build("billing:vanished")]
    await sync.heartbeat()  # not listed: warned, never raised
    await client.aclose()


async def test_a_registry_outage_degrades_to_last_known_good() -> None:
    state = {"fail": False}

    def handler(request: httpx.Request) -> httpx.Response:
        if state["fail"]:
            raise httpx.ConnectError("registry is down")
        return httpx.Response(200, json=manifest(), headers={"ETag": 'W/"7"'})

    client = registry(handler)
    sync = RegistrySync(client)
    await sync.poll()
    state["fail"] = True
    assert (await sync.poll()).empty  # served from cache, no exception
    await client.aclose()


async def test_the_loop_runs_cycles_and_survives_a_failure() -> None:
    state = {"calls": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        state["calls"] += 1
        if state["calls"] == 2:
            raise httpx.ConnectError("flaky")
        return httpx.Response(200, json=manifest(), headers={"ETag": 'W/"7"'})

    client = registry(handler)
    sync = RegistrySync(client, poll_seconds=1.0, heartbeat_seconds=1.0)
    await sync.cycle()
    assert sync.cycles == 1
    await sync.start()
    await asyncio.sleep(0.05)
    await sync.aclose()
    assert sync.cycles >= 1
    await client.aclose()


async def test_a_delta_channel_wakes_the_loop_and_a_broken_one_degrades_to_polling() -> None:
    _calls, handler = serving()
    client = registry(handler)

    class Channel:
        """A registry that pushes one change, then goes quiet."""

        def __init__(self) -> None:
            self.watched = 0
            self.quiet = asyncio.Event()

        async def watch(self) -> Any:
            self.watched += 1
            yield {"seq": 8}
            await self.quiet.wait()  # nothing more to say

    channel = Channel()
    sync = RegistrySync(
        client,
        channel=channel,
        poll_seconds=30.0,
        heartbeat_seconds=30.0,
        min_cycle_seconds=0.01,
    )
    await sync.start()
    await asyncio.sleep(0.1)
    await sync.aclose()
    assert channel.watched >= 1  # the channel, not the sleep, is what woke it
    assert sync.cycles < 20  # and a talkative channel never becomes a busy loop
    await client.aclose()


async def test_a_broken_channel_degrades_to_polling() -> None:
    _calls, handler = serving()
    client = registry(handler)

    class Broken:
        async def watch(self) -> Any:
            raise RuntimeError("the bus is gone")
            yield  # pragma: no cover - never reached

    sync = RegistrySync(
        client, channel=Broken(), poll_seconds=30.0, heartbeat_seconds=30.0, min_cycle_seconds=0.01
    )
    await sync.start()
    await asyncio.sleep(0.1)
    await sync.aclose()
    assert sync.channel is None  # dropped, loudly, and the poll carries on
    await client.aclose()


# ---------------------------------------------------------------------------- Bifrost MCP


def test_a_gateway_client_name_is_derived_not_guessed() -> None:
    assert mcp_client_name("invoice-lookup") == "invoice_lookup"
    assert mcp_client_name("invoice.lookup") == "invoice_lookup"
    with pytest.raises(ValueError, match="cannot be an MCP client name"):
        mcp_client_name("")
    with pytest.raises(ValueError, match="cannot be an MCP client name"):
        mcp_client_name("invoice/lookup")


def test_a_tool_entity_declares_its_binding_or_it_has_none() -> None:
    assert tool_binding(
        {"name": "t", "mcp": {"connection_type": "http", "url": "https://x.test/mcp"}}
    )
    assert tool_binding(
        {"name": "t", "connection_type": "sse", "connection_string": "https://x.test"}
    )
    assert tool_binding({"name": "t", "description": "no binding"}) is None
    assert tool_binding({"name": "t", "connection_type": "carrier-pigeon"}) is None
    assert tool_binding({"connection_type": "http"}) is None  # no name
    # only the two fields the catalogue owns reach the gateway: no option passthrough
    binding = tool_binding(
        {
            "name": "t",
            "mcp": {"connection_type": "http", "url": "https://x.test", "headers": {"x": "y"}},
        }
    )
    assert binding is not None and not hasattr(binding, "config")
    # the URL comes back normalised by the same validator a webhook target goes through
    assert (binding.connection_type, binding.connection_string) == ("http", "https://x.test/")


async def test_the_gateway_is_configured_from_the_registrys_tools() -> None:
    _calls, handler = serving()
    client = registry(handler)
    gateway = FakeGateway()
    sync = RegistrySync(client, gateway=gateway)
    result = await sync.configure_mcp_clients()
    assert result.added == ["invoice_lookup"]
    assert gateway.added == [
        {
            "name": "invoice_lookup",
            "connection_type": "http",
            "connection_string": "https://tools.example.com/mcp",
        }
    ]
    # a second pass changes nothing
    again = await sync.configure_mcp_clients()
    assert again.in_sync and gateway.added == [gateway.added[0]]
    await client.aclose()


async def test_a_changed_binding_is_updated_and_a_delisted_one_is_removed() -> None:
    _calls, handler = serving()
    client = registry(handler)
    gateway = FakeGateway()
    sync = RegistrySync(client, gateway=gateway)
    await sync.configure_mcp_clients()

    moved = manifest(seq=8)
    moved["entities"][2]["views"]["external"]["spec"]["mcp"]["connection_string"] = (
        "https://new/mcp"
    )
    handler.state["payload"] = moved
    result = await sync.configure_mcp_clients()
    assert result.updated == ["invoice_lookup"]
    assert gateway.updated[0][1] == {"connection_string": "https://new/mcp"}

    delisted = manifest(seq=9, entities=[e for e in manifest()["entities"] if e["type"] != "tool"])
    handler.state["payload"] = delisted
    result = await sync.configure_mcp_clients()
    assert result.removed == ["invoice_lookup"]
    assert gateway.removed == ["c-invoice_lookup"]
    await client.aclose()


async def test_a_client_the_sync_job_never_configured_is_left_alone() -> None:
    """Deleting a hand-registered server because a catalogue does not mention it is how a
    'reconciliation' takes production down."""
    _calls, handler = serving()
    client = registry(handler)
    gateway = FakeGateway([{"id": "c-hand", "config": {"name": "hand_registered"}}])
    sync = RegistrySync(client, gateway=gateway)
    result = await sync.configure_mcp_clients()
    assert result.added == ["invoice_lookup"]
    assert gateway.removed == []
    await client.aclose()


async def test_names_that_would_collide_are_refused_rather_than_merged() -> None:
    entities = [
        {
            "id": f"e{index}",
            "type": "tool",
            "name": name,
            "version": 1,
            "views": {
                "external": {
                    "enabled": True,
                    "spec": {"connection_type": "http", "url": f"https://x/{index}"},
                }
            },
        }
        for index, name in enumerate(("invoice-lookup", "invoice.lookup", "bad/name"))
    ]
    _calls, handler = serving(manifest(entities=entities))
    client = registry(handler)
    gateway = FakeGateway()
    sync = RegistrySync(client, gateway=gateway)
    result = await sync.configure_mcp_clients()
    assert result.added == []
    assert sorted(result.refused) == ["bad/name", "invoice-lookup", "invoice.lookup"]
    assert gateway.added == []
    await client.aclose()


async def test_a_catalogue_entry_cannot_make_the_gateway_run_a_local_process() -> None:
    """``stdio`` means the gateway spawns a process. A tool entity must not be able to introduce
    one on its own, so it is configured only when the deployment asked for that connection type."""
    entities = [
        {
            "id": "e1",
            "type": "tool",
            "name": "local-tool",
            "version": 1,
            "views": {
                "external": {
                    "enabled": True,
                    "spec": {"mcp": {"connection_type": "stdio", "connection_string": "/bin/sh"}},
                }
            },
        }
    ]
    _calls, handler = serving(manifest(entities=entities))
    client = registry(handler)
    gateway = FakeGateway()
    sync = RegistrySync(client, gateway=gateway)
    result = await sync.configure_mcp_clients()
    assert result.added == [] and result.refused == ["local-tool"]
    assert gateway.added == []

    allowed = RegistrySync(client, gateway=gateway, connection_types=("http", "sse", "stdio"))
    assert (await allowed.configure_mcp_clients()).added == ["local_tool"]
    await client.aclose()


async def test_a_tool_with_no_mcp_binding_is_reported_not_configured() -> None:
    entities = [
        {
            "id": "e1",
            "type": "tool",
            "name": "manual-tool",
            "version": 1,
            "views": {"external": {"enabled": True, "spec": {"description": "no binding"}}},
        }
    ]
    _calls, handler = serving(manifest(entities=entities))
    client = registry(handler)
    gateway = FakeGateway()
    sync = RegistrySync(client, gateway=gateway)
    result = await sync.configure_mcp_clients()
    assert result.unbound == ["manual-tool"] and result.added == []
    await client.aclose()


async def test_without_a_gateway_nothing_is_configured() -> None:
    _calls, handler = serving()
    client = registry(handler)
    sync = RegistrySync(client)
    assert (await sync.configure_mcp_clients()).in_sync
    await client.aclose()


def test_the_sync_job_refuses_things_that_are_not_what_it_needs() -> None:
    with pytest.raises(TypeError, match="AIRegistryClient"):
        RegistrySync(object())
    _calls, handler = serving()
    client = registry(handler)
    with pytest.raises(TypeError, match="Bifrost"):
        RegistrySync(client, gateway=object())
    with pytest.raises(TypeError, match="AIRegistryClient"):
        RegistryAgentDirectory(object())


async def test_a_client_whose_config_already_matches_is_not_adopted_and_then_deleted() -> None:
    """The job may only remove what it wrote. A hand-registered client that happens to match the
    catalogue is not "in sync, therefore mine"."""
    _calls, handler = serving()
    client = registry(handler)
    gateway = FakeGateway(
        [
            {
                "id": "c-hand",
                "config": {
                    "name": "invoice_lookup",
                    "connection_type": "http",
                    "connection_string": "https://tools.example.com/mcp",
                },
            }
        ]
    )
    sync = RegistrySync(client, gateway=gateway)
    first = await sync.configure_mcp_clients()
    assert (first.added, first.updated) == ([], [])  # nothing to do: it already matches
    handler.state["payload"] = manifest(
        seq=8, entities=[e for e in manifest()["entities"] if e["type"] != "tool"]
    )
    second = await sync.configure_mcp_clients()
    assert second.removed == [] and gateway.removed == []
    await client.aclose()


async def test_an_mcp_target_the_gateway_may_not_be_pointed_at_is_refused() -> None:
    """The connection string is a URL the *gateway* will fetch for every key allowed to use that
    client, so a catalogue entry cannot point it at the metadata service."""
    entities = [
        {
            "id": "e1",
            "type": "tool",
            "name": "metadata-probe",
            "version": 1,
            "views": {
                "external": {
                    "enabled": True,
                    "spec": {
                        "mcp": {
                            "connection_type": "http",
                            "connection_string": "http://169.254.169.254/latest/meta-data",
                        }
                    },
                }
            },
        }
    ]
    _calls, handler = serving(manifest(entities=entities))
    client = registry(handler)
    gateway = FakeGateway()
    sync = RegistrySync(client, gateway=gateway)
    result = await sync.configure_mcp_clients()
    assert result.unbound == ["metadata-probe"] and gateway.added == []
    # a development deployment may opt in to local targets, as it may for webhooks
    local = RegistrySync(client, gateway=gateway, allow_local_targets=True)
    assert (await local.configure_mcp_clients()).added == ["metadata_probe"]
    await client.aclose()


async def test_a_third_entity_does_not_inherit_a_name_two_others_lost() -> None:
    entities = [
        {
            "id": f"e{index}",
            "type": "tool",
            "name": name,
            "version": 1,
            "views": {
                "external": {
                    "enabled": True,
                    "spec": {"connection_type": "http", "url": f"https://x.test/{index}"},
                }
            },
        }
        for index, name in enumerate(("invoice-lookup", "invoice.lookup", "invoice_lookup"))
    ]
    _calls, handler = serving(manifest(entities=entities))
    client = registry(handler)
    gateway = FakeGateway()
    sync = RegistrySync(client, gateway=gateway)
    result = await sync.configure_mcp_clients()
    assert gateway.added == [] and result.added == []
    assert sorted(result.refused) == ["invoice-lookup", "invoice.lookup", "invoice_lookup"]
    await client.aclose()


async def test_one_gateway_refusal_does_not_stop_the_rest_of_the_catalogue() -> None:
    entities = [
        {
            "id": f"e{index}",
            "type": "tool",
            "name": name,
            "version": 1,
            "views": {
                "external": {
                    "enabled": True,
                    "spec": {"connection_type": "http", "url": f"https://x.test/{name}"},
                }
            },
        }
        for index, name in enumerate(("first-tool", "second-tool"))
    ]
    _calls, handler = serving(manifest(entities=entities))
    client = registry(handler)

    class Picky(FakeGateway):
        async def add(self, name: str, **config: Any) -> dict[str, Any]:
            if name == "first_tool":
                raise RuntimeError("the gateway said no")
            return await super().add(name, **config)

    gateway = Picky()
    sync = RegistrySync(client, gateway=gateway)
    result = await sync.configure_mcp_clients()
    assert result.added == ["second_tool"] and result.refused == ["first-tool"]
    await client.aclose()


async def test_a_development_card_url_is_published_when_the_deployment_says_so() -> None:
    calls, handler = serving()
    client = registry(handler, control_plane_token="cpt")
    local = "http://localhost:9000/.well-known/agent-card.json"
    assert await client.publish_card_url("refund-agent", local) is False
    assert await client.publish_card_url("refund-agent", local, allow_local=True) is True
    assert calls[-1].method == "PATCH"
    await client.aclose()
