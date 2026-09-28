"""The Registry as the catalogue of agents another agent may call (design §9).

``RegistryAgentDirectory`` implements the contracts ``AgentDirectory`` port over the AI
Registry's manifest: it answers "which agents may this caller reach, and where is each one's
Agent Card?" from the ETag-cached, audience-filtered manifest, and it publishes a card's
location back onto the entity.

Two rules this module exists to keep:

* **An agent is discoverable only once its card location is published.** The manifest lists
  what exists; a caller needs somewhere to fetch the card from. An entity with no card URL is
  skipped and logged, because a planner that "found" an agent it cannot call is worse than one
  that found nothing.
* **A card that comes back from a registry is data, not instructions.** Unknown keys are
  dropped by the contracts model, URLs must be http(s), and nothing here executes anything it
  read. The authoritative card is the one the agent itself serves at the published location;
  what the registry holds is the location and the catalogue facts around it.

No ``a2a-sdk`` import: the card type is the contracts one, and the server and client live in
``trellis-harness-a2a``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Final
from urllib.parse import urljoin

from trellis.contracts.a2a import AgentCard, AgentSkill

from trellis.harness.runtime.logging import get_logger

log = get_logger("trellis.harness.registry.directory")

#: Where an A2A agent serves its card. Protocol text (A2A's well-known path), not an SDK
#: import: the core stays free of ``a2a-sdk`` and a test in ``trellis-harness-a2a`` asserts
#: this constant still equals the SDK's own.
AGENT_CARD_PATH: Final = "/.well-known/agent-card.json"
#: What ``publish`` and ``get`` agree on: the card's location travels in the card's metadata,
#: because the contracts card's ``url`` is the agent's service endpoint, not its document.
CARD_URL_METADATA: Final = "card_url"


class RegistryAgentDirectory:
    """Agents a caller may reach, as Agent Cards, from an :class:`AIRegistryClient`."""

    name = "ai-registry"

    def __init__(
        self, client: Any, *, audience: str | None = None, allow_local: bool = False
    ) -> None:
        """``audience`` is the view to read (the registry pre-resolves overlays per audience;
        ``None`` uses the manifest's own default). ``allow_local`` permits http and local card
        URLs when publishing, for development only."""
        for method in ("manifest", "agents", "card_url", "publish_card_url", "qualified"):
            if not callable(getattr(client, method, None)):
                raise TypeError("RegistryAgentDirectory needs an AIRegistryClient")
        self._client = client
        self.audience = audience
        self.allow_local = allow_local

    async def get(self, agent_id: str) -> AgentCard | None:
        """The card of one agent, or ``None`` when the registry does not list it (or lists it
        without a card location, which is logged as the deployment gap it is)."""
        wanted = self._client.qualified(agent_id)
        for card in await self._cards():
            if card.name == wanted:
                return card
        return None

    async def find(
        self, query: str | None = None, *, skill: str | None = None, limit: int = 20
    ) -> Sequence[AgentCard]:
        """Callable agents matching a free-text query (name or description) and/or a skill id."""
        needle = (query or "").strip().lower()
        found: list[AgentCard] = []
        for card in await self._cards():
            if skill is not None and card.skill(skill) is None:
                continue
            if needle and needle not in f"{card.name} {card.description}".lower():
                continue
            found.append(card)
            if len(found) >= max(1, limit):
                break
        return found

    async def publish(self, card: AgentCard) -> None:
        """Write this agent's card location onto its registry entity.

        Never raises: a registry that will not take the write leaves discovery on whatever the
        entity already held, and :meth:`AIRegistryClient.publish_card_url` says so loudly.
        """
        location = card.metadata.get(CARD_URL_METADATA) or urljoin(card.url, AGENT_CARD_PATH)
        await self._client.publish_card_url(card.name, str(location), allow_local=self.allow_local)

    # ------------------------------------------------------------------ internals
    async def _cards(self) -> list[AgentCard]:
        entities = await self._client.agents(audience=self.audience)
        cards: list[AgentCard] = []
        for entity in entities:
            agent_id = str(entity.get("agent_id") or "")
            location = await self._client.card_url(agent_id, refresh=False) if agent_id else None
            if not location:
                log.warning(
                    "registry.agent_without_card",
                    agent_id=agent_id,
                    detail="listed in the registry with no A2A card URL; it cannot be called",
                )
                continue
            card = _card(agent_id, location, entity)
            if card is not None:
                cards.append(card)
        return cards


def _card(agent_id: str, location: str, entity: dict[str, Any]) -> AgentCard | None:
    """A contracts card from one manifest entity. A malformed entity is skipped, not raised:
    one bad row in a catalogue must not stop discovery of the rest."""
    try:
        return AgentCard(
            name=agent_id,
            description=str(entity.get("description") or ""),
            url=location,
            version=str(entity.get("version") or "0"),
            skills=_skills(entity.get("skills")),
            metadata={CARD_URL_METADATA: location, "entity_id": entity.get("id")},
        )
    except ValueError as exc:
        log.warning("registry.agent_card_invalid", agent_id=agent_id, error=str(exc))
        return None


def _skills(declared: Any) -> list[AgentSkill]:
    """Skills as the registry spec declares them: ids, or objects with an id and a description.
    Anything else in the list is dropped — it is another system's data."""
    skills: list[AgentSkill] = []
    for item in declared or ():
        if isinstance(item, str) and item.strip():
            skills.append(AgentSkill(id=item, name=item))
        elif isinstance(item, dict):
            skill_id = str(item.get("id") or item.get("skill_id") or "")
            if not skill_id:
                continue
            skills.append(
                AgentSkill(
                    id=skill_id,
                    name=str(item.get("name") or skill_id),
                    description=str(item.get("description") or ""),
                    tags=[str(t) for t in (item.get("tags") or []) if isinstance(t, str)],
                )
            )
    return skills


__all__ = ["AGENT_CARD_PATH", "CARD_URL_METADATA", "RegistryAgentDirectory"]
