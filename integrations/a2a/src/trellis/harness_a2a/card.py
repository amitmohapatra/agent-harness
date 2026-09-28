"""The Agent Card: generated from the Registry entity and the ``AgentDescriptor`` (design §9).

Three types meet here and it is worth naming them once:

* ``AgentDescriptor`` — what the code says this agent is (id, version, skills).
* the **Registry entity** — what was approved: the catalogue's description, its skills, its
  version, and (once published) where the card lives.
* ``trellis.contracts.a2a.AgentCard`` — the platform's card model, spelled as A2A 0.3.
* the **SDK's protobuf ``AgentCard``** — what A2A v1.0 actually puts on the wire.

``agent_card()`` builds the third from the first two; ``to_sdk_card()`` translates it into the
fourth, because v1.0 moved the endpoint into ``supported_interfaces`` and renamed ``security`` to
``security_requirements``. ``from_sdk_card()`` is the way back, for a card read from another
agent.

**No credential ever enters a card.** A card says *which* scheme to authenticate with; the
secret lives in the caller's credential store. The only thing published here is a scheme name,
a header name and a URL.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Final
from urllib.parse import urljoin

from a2a.types import (
    AgentCapabilities as SDKCapabilities,
)
from a2a.types import (
    AgentCard as SDKAgentCard,
)
from a2a.types import (
    AgentExtension,
    AgentInterface,
    APIKeySecurityScheme,
    HTTPAuthSecurityScheme,
    MutualTlsSecurityScheme,
    OAuth2SecurityScheme,
    OpenIdConnectSecurityScheme,
    SecurityRequirement,
    SecurityScheme,
    StringList,
)
from a2a.types import (
    AgentProvider as SDKProvider,
)
from a2a.types import (
    AgentSkill as SDKSkill,
)
from a2a.utils.constants import PROTOCOL_VERSION_CURRENT, TransportProtocol
from trellis.contracts.a2a import AgentCapabilities, AgentCard, AgentProvider, AgentSkill
from trellis.contracts.descriptors import AgentDescriptor

from trellis.harness.registry.directory import AGENT_CARD_PATH, CARD_URL_METADATA
from trellis.harness_a2a.identity import EXTENSION_DESCRIPTION, EXTENSION_URI

#: The platform's default security scheme: the team's key, in the standard header. The value of
#: that key is never part of a card.
TEAM_KEY_SCHEME: Final = "teamKey"
TEAM_KEY_HEADER: Final = "Authorization"
DEFAULT_MODES: Final = ("text/plain", "application/json")
#: What the registry entity's version is recorded as, since ``version`` on a card is the agent's.
REGISTRY_VERSION_METADATA: Final = "registry_version"


def team_key_scheme(header: str = TEAM_KEY_HEADER) -> dict[str, dict[str, Any]]:
    """The default security schemes block: one API key, named, never valued."""
    return {TEAM_KEY_SCHEME: {"type": "apiKey", "in": "header", "name": header}}


def card_location(base_url: str, *, path: str = AGENT_CARD_PATH) -> str:
    """Where the card document is served for an agent reachable at ``base_url``."""
    return urljoin(base_url if base_url.endswith("/") else base_url + "/", path.lstrip("/"))


def agent_card(
    descriptor: AgentDescriptor,
    *,
    url: str,
    entity: Mapping[str, Any] | None = None,
    streaming: bool = True,
    push_notifications: bool = False,
    provider: AgentProvider | None = None,
    security_schemes: Mapping[str, Mapping[str, Any]] | None = None,
    security: Sequence[Mapping[str, Sequence[str]]] | None = None,
    extensions: Iterable[Mapping[str, Any]] = (),
    documentation_url: str | None = None,
    card_path: str = AGENT_CARD_PATH,
) -> AgentCard:
    """The contracts card for one served agent.

    ``url`` is where the agent answers A2A calls; the card document's own location is derived
    from it and travels in ``metadata`` so the Registry write-back and the client agree on one
    string. ``entity`` is a Registry entity as ``AIRegistryClient.agents()`` returns it (the
    chosen audience's spec already flattened): its description and skills overlay the
    descriptor's, because the catalogue is what was approved, while ``version`` stays the code's
    and the catalogue's is kept beside it.
    """
    facts = dict(entity or {})
    schemes = {k: dict(v) for k, v in (security_schemes or team_key_scheme()).items()}
    requirement = (
        [{name: []} for name in schemes] if security is None else [dict(s) for s in security]
    )
    card = AgentCard.from_descriptor(
        descriptor,
        url=url,
        capabilities=AgentCapabilities(
            streaming=streaming,
            push_notifications=push_notifications,
            extensions=[
                _extension(EXTENSION_URI, EXTENSION_DESCRIPTION),
                *(dict(e) for e in extensions),
            ],
        ),
        provider=provider,
        security_schemes=schemes,
        security=requirement,  # type: ignore[arg-type]  # list[dict[str, list[str]]]
    )
    overlay: dict[str, Any] = {
        "description": str(facts.get("description") or descriptor.description),
        "skills": _merged_skills(descriptor, facts.get("skills")),
        "default_input_modes": list(facts.get("input_modes") or DEFAULT_MODES),
        "default_output_modes": list(facts.get("output_modes") or DEFAULT_MODES),
        "metadata": {
            **card.metadata,
            CARD_URL_METADATA: card_location(url, path=card_path),
            REGISTRY_VERSION_METADATA: facts.get("version"),
            "entity_id": facts.get("id"),
        },
    }
    if documentation_url or facts.get("documentation_url"):
        overlay["documentation_url"] = documentation_url or str(facts["documentation_url"])
    return card.model_copy(update=overlay)


def to_sdk_card(
    card: AgentCard, *, transport: str = TransportProtocol.JSONRPC, tenant: str = ""
) -> SDKAgentCard:
    """The protocol's card (A2A v1.0, protobuf) for a platform card.

    The one translation worth reading twice: v1.0 has no ``url`` on a card. The endpoint and its
    transport are an ``AgentInterface``, so an agent that answers on several transports lists
    several — which is exactly what ``additional_interfaces`` already meant.
    """
    interfaces = [
        AgentInterface(
            url=card.url,
            protocol_binding=card.preferred_transport or transport,
            protocol_version=PROTOCOL_VERSION_CURRENT,
            tenant=tenant,
        ),
        *(
            AgentInterface(
                url=extra.url,
                protocol_binding=extra.transport,
                protocol_version=PROTOCOL_VERSION_CURRENT,
                tenant=tenant,
            )
            for extra in card.additional_interfaces
        ),
    ]
    sdk = SDKAgentCard(
        name=card.name,
        description=card.description,
        version=card.version,
        supported_interfaces=interfaces,
        capabilities=SDKCapabilities(
            streaming=card.capabilities.streaming,
            push_notifications=card.capabilities.push_notifications,
            extended_agent_card=card.supports_authenticated_extended_card,
            extensions=[_sdk_extension(e) for e in card.capabilities.extensions],
        ),
        default_input_modes=list(card.default_input_modes),
        default_output_modes=list(card.default_output_modes),
        skills=[_sdk_skill(skill) for skill in card.skills],
        security_schemes={name: _sdk_scheme(name, s) for name, s in card.security_schemes.items()},
        security_requirements=[
            SecurityRequirement(
                schemes={name: StringList(list=list(scopes)) for name, scopes in req.items()}
            )
            for req in card.security
        ],
    )
    if card.provider is not None:
        sdk.provider.CopyFrom(
            SDKProvider(organization=card.provider.organization, url=card.provider.url)
        )
    if card.documentation_url:
        sdk.documentation_url = card.documentation_url
    if card.icon_url:
        sdk.icon_url = card.icon_url
    return sdk


def from_sdk_card(sdk: SDKAgentCard, *, transport: str = TransportProtocol.JSONRPC) -> AgentCard:
    """A platform card for a card read from another agent.

    Lenient on purpose: a card is another system's data. What it cannot do is leave the caller
    without an endpoint — an interface for the transport we speak, or the first one offered.
    """
    interfaces = list(sdk.supported_interfaces)
    chosen = next((i for i in interfaces if i.protocol_binding == transport), None) or (
        interfaces[0] if interfaces else None
    )
    if chosen is None:
        raise ValueError(f"the card for {sdk.name!r} offers no interface to call")
    return AgentCard(
        name=sdk.name,
        description=sdk.description,
        url=chosen.url,
        version=sdk.version or "0",
        protocol_version=chosen.protocol_version or PROTOCOL_VERSION_CURRENT,
        preferred_transport=chosen.protocol_binding or transport,
        capabilities=AgentCapabilities(
            streaming=sdk.capabilities.streaming,
            push_notifications=sdk.capabilities.push_notifications,
            extensions=[
                {"uri": e.uri, "description": e.description, "required": e.required}
                for e in sdk.capabilities.extensions
            ],
        ),
        default_input_modes=list(sdk.default_input_modes) or list(DEFAULT_MODES),
        default_output_modes=list(sdk.default_output_modes) or list(DEFAULT_MODES),
        skills=[
            AgentSkill(
                id=s.id,
                name=s.name or s.id,
                description=s.description,
                tags=list(s.tags),
                examples=list(s.examples),
            )
            for s in sdk.skills
        ],
        security_schemes={name: _plain_scheme(s) for name, s in sdk.security_schemes.items()},
        security=[
            {name: list(scopes.list) for name, scopes in req.schemes.items()}
            for req in sdk.security_requirements
        ],
        supports_authenticated_extended_card=sdk.capabilities.extended_agent_card,
        metadata={
            "interfaces": [{"url": i.url, "transport": i.protocol_binding} for i in interfaces]
        },
    )


# ---------------------------------------------------------------------------- internals


def _extension(uri: str, description: str, *, required: bool = False) -> dict[str, Any]:
    return {"uri": uri, "description": description, "required": required}


def _merged_skills(descriptor: AgentDescriptor, declared: Any) -> list[AgentSkill]:
    """The descriptor's skills, with what the catalogue says about each overlaid on top."""
    skills = {s.skill_id: AgentSkill.from_descriptor(s) for s in descriptor.skills}
    for item in declared or ():
        if isinstance(item, str) and item.strip():
            skills.setdefault(item, AgentSkill(id=item, name=item))
        elif isinstance(item, Mapping):
            skill_id = str(item.get("id") or item.get("skill_id") or "")
            if not skill_id:
                continue
            existing = skills.get(skill_id)
            skills[skill_id] = AgentSkill(
                id=skill_id,
                name=str(item.get("name") or (existing.name if existing else skill_id)),
                description=str(
                    item.get("description") or (existing.description if existing else "")
                ),
                tags=[str(t) for t in (item.get("tags") or [])]
                or (list(existing.tags) if existing else []),
                examples=[str(e) for e in (item.get("examples") or [])],
            )
    return list(skills.values())


def _sdk_extension(extension: Mapping[str, Any]) -> AgentExtension:
    return AgentExtension(
        uri=str(extension.get("uri") or ""),
        description=str(extension.get("description") or ""),
        required=bool(extension.get("required")),
        params=extension.get("params") or None,
    )


def _sdk_skill(skill: AgentSkill) -> SDKSkill:
    return SDKSkill(
        id=skill.id,
        name=skill.name,
        description=skill.description,
        tags=list(skill.tags),
        examples=list(skill.examples),
        input_modes=list(skill.input_modes or []),
        output_modes=list(skill.output_modes or []),
    )


def _sdk_scheme(name: str, scheme: Mapping[str, Any]) -> SecurityScheme:
    """One security scheme, in the protocol's own union.

    An unsupported shape raises: a card is generated from this deployment's own configuration, so
    a scheme that could not be translated is a startup error, never a scheme quietly dropped from
    what callers are told to use.
    """
    kind = str(scheme.get("type") or "").lower()
    description = str(scheme.get("description") or "")
    if kind == "apikey":
        location = str(scheme.get("in") or scheme.get("location") or "header").lower()
        if location != "header":
            raise ValueError(f"security scheme {name!r}: only apiKey in a header is supported")
        return SecurityScheme(
            api_key_security_scheme=APIKeySecurityScheme(
                description=description,
                location=location,
                name=str(scheme.get("name") or TEAM_KEY_HEADER),
            )
        )
    if kind == "http":
        return SecurityScheme(
            http_auth_security_scheme=HTTPAuthSecurityScheme(
                description=description,
                scheme=str(scheme.get("scheme") or "bearer"),
                bearer_format=str(scheme.get("bearerFormat") or scheme.get("bearer_format") or ""),
            )
        )
    if kind in ("openidconnect", "openid_connect"):
        return SecurityScheme(
            open_id_connect_security_scheme=OpenIdConnectSecurityScheme(
                description=description,
                open_id_connect_url=str(
                    scheme.get("openIdConnectUrl") or scheme.get("open_id_connect_url") or ""
                ),
            )
        )
    if kind == "oauth2":
        metadata_url = str(
            scheme.get("oauth2MetadataUrl") or scheme.get("oauth2_metadata_url") or ""
        )
        if scheme.get("flows"):
            raise ValueError(
                f"security scheme {name!r}: OAuth2 flows are not translated by this surface; "
                "publish an oauth2MetadataUrl instead so callers read the flows from the issuer"
            )
        return SecurityScheme(
            oauth2_security_scheme=OAuth2SecurityScheme(
                description=description, oauth2_metadata_url=metadata_url
            )
        )
    if kind in ("mutualtls", "mtls"):
        return SecurityScheme(mtls_security_scheme=MutualTlsSecurityScheme(description=description))
    raise ValueError(f"security scheme {name!r}: unsupported type {kind!r}")


def _plain_scheme(scheme: SecurityScheme) -> dict[str, Any]:
    """The 0.3-spelled dict for a protocol scheme, for the contracts card."""
    which = scheme.WhichOneof("scheme")
    if which == "api_key_security_scheme":
        inner = scheme.api_key_security_scheme
        return {
            "type": "apiKey",
            "in": inner.location or "header",
            "name": inner.name,
            "description": inner.description,
        }
    if which == "http_auth_security_scheme":
        inner_http = scheme.http_auth_security_scheme
        return {
            "type": "http",
            "scheme": inner_http.scheme,
            "bearerFormat": inner_http.bearer_format,
            "description": inner_http.description,
        }
    if which == "oauth2_security_scheme":
        return {
            "type": "oauth2",
            "oauth2MetadataUrl": scheme.oauth2_security_scheme.oauth2_metadata_url,
            "description": scheme.oauth2_security_scheme.description,
        }
    if which == "open_id_connect_security_scheme":
        return {
            "type": "openIdConnect",
            "openIdConnectUrl": scheme.open_id_connect_security_scheme.open_id_connect_url,
            "description": scheme.open_id_connect_security_scheme.description,
        }
    if which == "mtls_security_scheme":
        return {"type": "mutualTLS", "description": scheme.mtls_security_scheme.description}
    return {"type": str(which or "unknown")}


__all__ = [
    "DEFAULT_MODES",
    "REGISTRY_VERSION_METADATA",
    "TEAM_KEY_HEADER",
    "TEAM_KEY_SCHEME",
    "agent_card",
    "card_location",
    "from_sdk_card",
    "team_key_scheme",
    "to_sdk_card",
]
