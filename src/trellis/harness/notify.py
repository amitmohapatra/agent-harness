"""Telling a person a run waits for them: :class:`Notifier`\\ s, called when a run pauses.

    class Pager:
        async def notify(self, interrupt: Interrupt, link: str | None) -> None:
            await page(interrupt.assignee, interrupt.question, link)

    h = Harness(notifiers=[Pager()])

Two ship with the harness, each on when the deployment names where it sends:

* :class:`Slack` — a Slack incoming webhook (``SLACK_WEBHOOK_URL``): the question, whose it
  is, the tool call under approval, and the link;
* :class:`Email` — an SMTP server (``SMTP_URL``, ``smtp://user:password@host:587`` or
  ``smtps://…`` for TLS from the start; ``SMTP_FROM`` the sender): to the assignee when it is an
  address (``user:ada@example.com``), else to ``SMTP_TO``.

``link`` is where the question is answered: ``TRELLIS_INBOX_URL`` (the reference inbox,
``h.serve_inbox``, or your own) with ``#<interrupt id>``; ``None`` when it is unset.

Best-effort, after the pause is recorded, in the background (the harness's background writes):
a notifier that fails is a ``warning`` event on the run (logged and counted
``trellis.notifications``), never a failed run, and is not retried. What a notifier gets is
redacted as everything leaving the process is (``redaction.py``): the question, the payload,
the screen's props and the tool call's arguments — not whose it is. A sub-agent's question is
notified once, as its parent's run's. agent-runs' webhooks (``run.paused``, signed) are the
other way: a receiver of your own, for every run of the tenant, wrapped or not.
"""

from __future__ import annotations

import asyncio
import json
import logging
import smtplib
import ssl
from email.message import EmailMessage
from typing import Any, Final, Protocol, runtime_checkable
from urllib.parse import unquote, urlsplit

import httpx

from trellis.contracts import ConfigurationError, Interrupt
from trellis.harness.redaction import DEFAULT as REDACTOR
from trellis.harness.settings import Settings

log = logging.getLogger("trellis.notify")

#: How long one notification may take.
NOTIFY_TIMEOUT_SECONDS: Final = 10.0
#: The SMTP ports by scheme, when the URL names none.
SMTP_PORTS: Final = {"smtp": 587, "smtps": 465}


@runtime_checkable
class Notifier(Protocol):
    """Something told when a run pauses for a person."""

    async def notify(self, interrupt: Interrupt, link: str | None) -> None: ...


def redacted(interrupt: Interrupt) -> Interrupt:
    """``interrupt`` as a notifier gets it: what it asks redacted, whose it is not."""
    call = interrupt.tool_call
    return interrupt.model_copy(
        update={
            "question": REDACTOR.redact_output(interrupt.question),
            "payload": REDACTOR.redact_input(interrupt.payload),
            "props": REDACTOR.redact_input(interrupt.props),
            "tool_call": None
            if call is None
            else call.model_copy(update={"args": REDACTOR.redact_input(call.args)}),
        }
    )


def summary(interrupt: Interrupt, link: str | None) -> list[str]:
    """The lines a notification says: the question, whose it is and by when, the call under
    approval, where to answer."""
    lines = [interrupt.question]
    if interrupt.assignee:
        lines.append(f"For: {interrupt.assignee}")
    if interrupt.deadline is not None:
        lines.append(f"Due: {interrupt.deadline.isoformat()}")
    call = interrupt.tool_call
    if call is not None:
        lines.append(f"Call: {call.tool} {json.dumps(call.args, default=str)}")
    if link:
        lines.append(f"Answer: {link}")
    return lines


class Slack:
    """A Slack incoming webhook (``SLACK_WEBHOOK_URL``)."""

    name: Final = "slack"

    def __init__(self, webhook_url: str, *, client: httpx.AsyncClient | None = None) -> None:
        self.webhook_url = webhook_url
        self._client = client

    async def notify(self, interrupt: Interrupt, link: str | None) -> None:
        text = "\n".join(summary(interrupt, link))
        body = {"text": f"A run is waiting: {text}"}
        if self._client is not None:
            response = await self._client.post(self.webhook_url, json=body)
        else:
            async with httpx.AsyncClient(timeout=NOTIFY_TIMEOUT_SECONDS) as client:
                response = await client.post(self.webhook_url, json=body)
        response.raise_for_status()


class Email:
    """An SMTP server (``SMTP_URL``, ``SMTP_FROM``; ``SMTP_TO`` for questions not assigned to
    an address)."""

    name: Final = "email"

    def __init__(self, url: str, sender: str, *, to: str | None = None) -> None:
        parts = urlsplit(url)
        if parts.scheme not in SMTP_PORTS or not parts.hostname:
            raise ConfigurationError(
                f"SMTP_URL must be smtp://host[:port] or smtps://host[:port], not {url!r}"
            )
        self.tls = parts.scheme == "smtps"
        self.host = parts.hostname
        self.port = parts.port or SMTP_PORTS[parts.scheme]
        self.user = unquote(parts.username) if parts.username else None
        self.password = unquote(parts.password) if parts.password else None
        self.sender = sender
        self.to = to

    def recipient(self, interrupt: Interrupt) -> str | None:
        """The assignee when it is an address, else ``SMTP_TO``."""
        named = (interrupt.assignee or "").partition(":")[2]
        return named if "@" in named else self.to

    async def notify(self, interrupt: Interrupt, link: str | None) -> None:
        to = self.recipient(interrupt)
        if to is None:
            raise ConfigurationError(
                f"{interrupt.interrupt_id} is for {interrupt.assignee or 'anyone'}, which is no "
                "address: set SMTP_TO for such questions"
            )
        message = EmailMessage()
        message["From"], message["To"] = self.sender, to
        message["Subject"] = f"Waiting for you: {interrupt.question}"[:200]
        message.set_content("\n".join(summary(interrupt, link)))
        await asyncio.to_thread(self._send, message)

    def _send(self, message: EmailMessage) -> None:
        context = ssl.create_default_context()
        connect = smtplib.SMTP_SSL if self.tls else smtplib.SMTP
        extra: dict[str, Any] = {"context": context} if self.tls else {}
        with connect(self.host, self.port, timeout=NOTIFY_TIMEOUT_SECONDS, **extra) as smtp:
            if not self.tls and smtp.has_extn("starttls"):
                smtp.starttls(context=context)
            if self.user is not None:
                smtp.login(self.user, self.password or "")
            smtp.send_message(message)


def from_env(settings: Settings) -> list[Notifier]:
    """The notifiers the deployment names: Slack with ``SLACK_WEBHOOK_URL``, email with
    ``SMTP_URL`` (which needs ``SMTP_FROM``)."""
    found: list[Notifier] = []
    if settings.slack_webhook_url:
        found.append(Slack(settings.slack_webhook_url))
    if settings.smtp_url:
        if not settings.smtp_from:
            raise ConfigurationError("SMTP_URL needs SMTP_FROM: the address mail is sent from")
        found.append(Email(settings.smtp_url, settings.smtp_from, to=settings.smtp_to))
    return found


def link_of(settings: Settings, interrupt: Interrupt) -> str | None:
    """Where ``interrupt`` is answered: ``TRELLIS_INBOX_URL#<interrupt id>``, when set."""
    if not settings.inbox_url:
        return None
    return f"{settings.inbox_url}#{interrupt.interrupt_id}"


def name_of(notifier: Notifier) -> str:
    return str(getattr(notifier, "name", type(notifier).__name__))


__all__ = ["Email", "Notifier", "Slack", "from_env", "link_of", "redacted", "summary"]
