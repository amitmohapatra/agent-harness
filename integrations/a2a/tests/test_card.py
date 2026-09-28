"""The Agent Card: generated from the descriptor and the Registry entity, translated to the
protocol's own v1.0 shape, and carrying no secrets."""

from __future__ import annotations

import pytest
from a2a.server.request_handlers.response_helpers import agent_card_to_dict
from a2a.types import TaskState
from a2a.utils.constants import AGENT_CARD_WELL_KNOWN_PATH, PROTOCOL_VERSION_CURRENT
from trellis.contracts.descriptors import AgentDescriptor
from trellis.contracts.runs import RunOutcome, RunStatus

from trellis.harness.registry.directory import AGENT_CARD_PATH
from trellis.harness_a2a import agent_card, card_location, from_sdk_card, to_sdk_card
from trellis.harness_a2a.card import REGISTRY_VERSION_METADATA, TEAM_KEY_SCHEME
from trellis.harness_a2a.tasks import TASK_STATES
from trellis.harness_a2a.translate import OUTCOME_STATES

DESCRIPTOR = AgentDescriptor.build(
    "billing:refund-agent",
    skills=["billing.refund"],
    version="1.2.3",
    description="what the code says",
)
ENTITY = {
    "id": "e1",
    "name": "refund-agent",
    "agent_id": "billing:refund-agent",
    "version": 3,
    "description": "issues refunds",
    "skills": [{"id": "billing.refund", "description": "from the catalogue", "tags": ["billing"]}],
}
URL = "https://agents.example.com/a2a"


def test_the_core_and_this_package_agree_on_the_well_known_path() -> None:
    """The core names the path without importing the SDK; this is what keeps that honest."""
    assert AGENT_CARD_PATH == AGENT_CARD_WELL_KNOWN_PATH


def test_the_card_is_the_descriptor_with_the_catalogue_overlaid() -> None:
    card = agent_card(DESCRIPTOR, url=URL, entity=ENTITY, push_notifications=True)
    assert card.name == "billing:refund-agent"
    assert card.description == "issues refunds"  # the catalogue is what was approved
    assert card.version == "1.2.3"  # the code's version, not the entity's
    assert card.metadata[REGISTRY_VERSION_METADATA] == 3
    skill = card.skill("billing.refund")
    assert skill is not None and skill.description == "from the catalogue"
    assert skill.tags == ["billing"]
    assert card.capabilities.streaming is True
    assert card.capabilities.push_notifications is True
    assert [e["uri"] for e in card.capabilities.extensions] == [
        "https://trellis.dev/a2a/extensions/trusted-identity/v1"
    ]
    assert card.metadata["card_url"] == card_location(URL)
    assert card_location(URL) == f"{URL}{AGENT_CARD_WELL_KNOWN_PATH}"


def test_a_card_without_a_registry_entity_still_describes_the_agent() -> None:
    card = agent_card(DESCRIPTOR, url=URL)
    assert card.description == "what the code says"
    assert [s.id for s in card.skills] == ["billing.refund"]
    assert card.capabilities.push_notifications is False  # not configured, so not advertised


def test_the_team_key_is_the_default_scheme_and_no_secret_is_published() -> None:
    card = agent_card(DESCRIPTOR, url=URL, entity=ENTITY)
    assert card.security_schemes[TEAM_KEY_SCHEME] == {
        "type": "apiKey",
        "in": "header",
        "name": "Authorization",
    }
    assert card.security == [{TEAM_KEY_SCHEME: []}]
    served = agent_card_to_dict(to_sdk_card(card))
    assert "secret" not in str(served).lower()
    scheme = served["securitySchemes"][TEAM_KEY_SCHEME]["apiKeySecurityScheme"]
    assert set(scheme) == {"location", "name"}  # a name and a place, never a value


def test_the_protocol_card_moves_the_endpoint_into_an_interface() -> None:
    sdk = to_sdk_card(agent_card(DESCRIPTOR, url=URL, entity=ENTITY, push_notifications=True))
    served = agent_card_to_dict(sdk)
    assert served["supportedInterfaces"] == [
        {"url": URL, "protocolBinding": "JSONRPC", "protocolVersion": PROTOCOL_VERSION_CURRENT}
    ]
    assert "url" not in served  # v1.0 has no card-level url
    assert served["capabilities"]["pushNotifications"] is True
    assert served["securityRequirements"] == [{"schemes": {TEAM_KEY_SCHEME: {}}}]


def test_a_card_read_back_says_the_same_thing() -> None:
    card = agent_card(DESCRIPTOR, url=URL, entity=ENTITY, push_notifications=True)
    back = from_sdk_card(to_sdk_card(card))
    assert (back.name, back.url, back.version) == (card.name, card.url, card.version)
    assert [s.id for s in back.skills] == [s.id for s in card.skills]
    assert back.capabilities.streaming and back.capabilities.push_notifications
    assert back.security == card.security
    assert back.security_schemes[TEAM_KEY_SCHEME]["type"] == "apiKey"


def test_an_untranslatable_security_scheme_is_a_startup_error_not_a_silent_drop() -> None:
    """The schemes come from this deployment's configuration, so a shape we cannot publish is a
    mistake to fix, never a scheme callers are quietly not told about."""
    card = agent_card(
        DESCRIPTOR,
        url=URL,
        security_schemes={"oauth": {"type": "oauth2", "flows": {"clientCredentials": {}}}},
    )
    with pytest.raises(ValueError, match="OAuth2 flows"):
        to_sdk_card(card)
    with pytest.raises(ValueError, match="unsupported type"):
        to_sdk_card(agent_card(DESCRIPTOR, url=URL, security_schemes={"x": {"type": "magic"}}))
    with pytest.raises(ValueError, match="apiKey in a header"):
        to_sdk_card(
            agent_card(
                DESCRIPTOR, url=URL, security_schemes={"q": {"type": "apiKey", "in": "query"}}
            )
        )


def test_other_schemes_translate() -> None:
    card = agent_card(
        DESCRIPTOR,
        url=URL,
        security_schemes={
            "bearer": {"type": "http", "scheme": "bearer", "bearerFormat": "JWT"},
            "oidc": {
                "type": "openIdConnect",
                "openIdConnectUrl": "https://issuer.test/.well-known",
            },
            "mtls": {"type": "mutualTLS"},
            "oauth": {"type": "oauth2", "oauth2MetadataUrl": "https://issuer.test/meta"},
        },
    )
    served = agent_card_to_dict(to_sdk_card(card))["securitySchemes"]
    assert served["bearer"]["httpAuthSecurityScheme"]["scheme"] == "bearer"
    assert served["oidc"]["openIdConnectSecurityScheme"]["openIdConnectUrl"].startswith("https://")
    assert "mtlsSecurityScheme" in served["mtls"]
    assert (
        served["oauth"]["oauth2SecurityScheme"]["oauth2MetadataUrl"] == "https://issuer.test/meta"
    )
    back = from_sdk_card(to_sdk_card(card))
    assert {name: s["type"] for name, s in back.security_schemes.items()} == {
        "bearer": "http",
        "oidc": "openIdConnect",
        "mtls": "mutualTLS",
        "oauth": "oauth2",
    }


def test_a_card_with_no_callable_interface_is_refused() -> None:
    sdk = to_sdk_card(agent_card(DESCRIPTOR, url=URL))
    del sdk.supported_interfaces[:]
    with pytest.raises(ValueError, match="no interface to call"):
        from_sdk_card(sdk)


def test_every_run_status_and_outcome_has_a_task_state() -> None:
    """A run must always be reportable: a missing mapping would strand a task in an unknown state
    (and the rebuild from the run store would answer ``TASK_STATE_UNSPECIFIED``)."""
    assert set(TASK_STATES) == set(RunStatus)
    assert TASK_STATES[RunStatus.PAUSED] is TaskState.TASK_STATE_INPUT_REQUIRED
    assert set(OUTCOME_STATES) == set(RunOutcome)
    assert OUTCOME_STATES[RunOutcome.REJECTED] is TaskState.TASK_STATE_REJECTED
    assert OUTCOME_STATES[RunOutcome.INTERRUPT] is TaskState.TASK_STATE_INPUT_REQUIRED
