"""Notifiers: what a notification says, redacted; Slack's incoming webhook; SMTP, to the
assignee's address or ``SMTP_TO``; the environment's notifiers and the answering link."""

from __future__ import annotations

import json
import smtplib
from datetime import UTC, datetime
from email.message import EmailMessage
from typing import Any, ClassVar

import httpx
import pytest
import respx

from trellis import Harness, Settings
from trellis.contracts import ConfigurationError, Interrupt, InterruptReason, ToolCall
from trellis.harness import notify
from trellis.harness.notify import Email, Notifier, Slack

WEBHOOK = "https://hooks.slack.test/services/T/B/x"


def asked(**fields: Any) -> Interrupt:
    base: dict[str, Any] = {
        "interrupt_id": "run_1.1.1",
        "tenant_id": "acme",
        "run_id": "run_1",
        "reason": InterruptReason.APPROVAL,
        "question": "Approve refund for bob@example.com?",
        "tool_call": ToolCall(tool="refund", args={"amount": 5, "password": "hunter2"}),
        "assignee": "user:ada@example.com",
        "deadline": datetime(2026, 10, 7, tzinfo=UTC),
        "payload": {"api_key": "sk-abcdefghijklmnop"},
    }
    return Interrupt(**{**base, **fields})


def test_a_notifier_gets_what_is_asked_redacted_but_whose_it_is() -> None:
    told = notify.redacted(asked(component="refund-review", props={"token": "t"}))
    assert told.question == "Approve refund for [email]?"
    assert told.tool_call is not None and told.tool_call.args["password"] == "[redacted]"
    assert told.payload == {"api_key": "[redacted]"} and told.props == {"token": "[redacted]"}
    assert told.assignee == "user:ada@example.com"  # where it goes is not redacted
    plain = notify.redacted(asked(tool_call=None, reason=InterruptReason.QUESTION))
    assert plain.tool_call is None
    lines = notify.summary(told, "https://ops.example/inbox#run_1.1.1")
    assert lines == [
        "Approve refund for [email]?",
        "For: user:ada@example.com",
        "Due: 2026-10-07T00:00:00+00:00",
        'Call: refund {"amount": 5, "password": "[redacted]"}',
        "Answer: https://ops.example/inbox#run_1.1.1",
    ]
    bare = asked(tool_call=None, reason=InterruptReason.QUESTION, assignee=None, deadline=None)
    assert notify.summary(bare, None) == [bare.question]


@respx.mock
async def test_slack_posts_the_summary_to_its_incoming_webhook() -> None:
    route = respx.post(WEBHOOK).mock(return_value=httpx.Response(200, text="ok"))
    await Slack(WEBHOOK).notify(asked(), "https://ops.example/inbox#run_1.1.1")
    text = json.loads(route.calls[0].request.content)["text"]
    assert text.startswith("A run is waiting: Approve refund") and "Answer: https://" in text
    async with httpx.AsyncClient() as client:
        respx.post(WEBHOOK).mock(return_value=httpx.Response(404, text="no_service"))
        with pytest.raises(httpx.HTTPStatusError):
            await Slack(WEBHOOK, client=client).notify(asked(), None)
    assert isinstance(Slack(WEBHOOK), Notifier)


class FakeSMTP:
    sent: ClassVar[list[tuple[str, int, bool, EmailMessage, tuple[str, str] | None]]] = []
    starttls_offered = True

    def __init__(self, host: str, port: int, timeout: float, **kwargs: Any) -> None:
        self.host, self.port, self.tls = host, port, "context" in kwargs
        self.login_as: tuple[str, str] | None = None
        self.upgraded = False

    def __enter__(self) -> FakeSMTP:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def has_extn(self, name: str) -> bool:
        return name == "starttls" and self.starttls_offered

    def starttls(self, context: Any) -> None:
        self.upgraded = True

    def login(self, user: str, password: str) -> None:
        self.login_as = (user, password)

    def send_message(self, message: EmailMessage) -> None:
        FakeSMTP.sent.append(
            (self.host, self.port, self.tls or self.upgraded, message, self.login_as)
        )


@pytest.fixture
def smtp(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    FakeSMTP.sent = []
    monkeypatch.setattr(smtplib, "SMTP", FakeSMTP)
    monkeypatch.setattr(smtplib, "SMTP_SSL", FakeSMTP)
    return FakeSMTP.sent


async def test_email_goes_to_the_assignees_address_else_to_smtp_to(smtp: list[Any]) -> None:
    mail = Email("smtp://bot%40ops:s3cret@mail.test", "trellis@ops.example", to="team@ops.example")
    assert (mail.host, mail.port, mail.tls, mail.user, mail.password) == (
        "mail.test",
        587,
        False,
        "bot@ops",
        "s3cret",
    )
    await mail.notify(asked(), "https://ops.example/inbox#run_1.1.1")
    await mail.notify(asked(assignee="role:finance"), None)
    [(host, port, secure, first, login), (_, _, _, second, _)] = smtp
    assert (host, port, secure, login) == ("mail.test", 587, True, ("bot@ops", "s3cret"))
    assert first["To"] == "ada@example.com" and second["To"] == "team@ops.example"
    assert first["Subject"].startswith("Waiting for you: Approve refund")
    assert "Answer: https://ops.example/inbox#run_1.1.1" in first.get_content()


async def test_email_over_tls_without_a_login_and_without_starttls(
    smtp: list[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    await Email("smtps://mail.test", "t@ops.example").notify(asked(), None)
    monkeypatch.setattr(FakeSMTP, "starttls_offered", False)
    await Email("smtp://mail.test:25", "t@ops.example").notify(asked(), None)
    assert [(port, secure, login) for _, port, secure, _, login in smtp] == [
        (465, True, None),
        (25, False, None),
    ]


async def test_email_refuses_what_it_cannot_send() -> None:
    with pytest.raises(ConfigurationError, match="SMTP_URL must be smtp://"):
        Email("https://mail.test", "t@x")
    with pytest.raises(ConfigurationError, match="set SMTP_TO"):
        await Email("smtp://mail.test", "t@x").notify(asked(assignee=None), None)


def test_the_environment_names_the_notifiers_and_the_link() -> None:
    assert notify.from_env(Settings()) == []
    both = notify.from_env(
        Settings(slack_webhook_url=WEBHOOK, smtp_url="smtp://mail.test", smtp_from="t@x")
    )
    assert [type(n) for n in both] == [Slack, Email]
    assert [notify.name_of(n) for n in both] == ["slack", "email"]
    with pytest.raises(ConfigurationError, match="SMTP_URL needs SMTP_FROM"):
        notify.from_env(Settings(smtp_url="smtp://mail.test"))
    assert notify.link_of(Settings(), asked()) is None
    link = notify.link_of(Settings(inbox_url="https://ops.example/inbox"), asked())
    assert link == "https://ops.example/inbox#run_1.1.1"


@respx.mock
async def test_a_harness_has_the_environments_notifiers_then_its_own() -> None:
    respx.post(WEBHOOK).mock(return_value=httpx.Response(500))

    class Mine:
        async def notify(self, interrupt: Interrupt, link: str | None) -> None:
            return None

    mine = Mine()
    async with Harness(config=Settings(slack_webhook_url=WEBHOOK), notifiers=[mine]) as h:
        assert [type(n) for n in h.notifiers] == [Slack, Mine]
        assert notify.name_of(mine) == "Mine"
        await h.notified(asked())  # Slack fails: logged; no run's events to warn on
        await h.writes.drain()
    async with Harness(config=Settings()) as quiet:
        await quiet.notified(asked())  # no notifier: nothing to do
