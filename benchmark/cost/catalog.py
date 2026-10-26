"""The benchmark's tasks and its tool catalogues of 10, 50, 100 and 500 tools.

**Tasks** — short (two parts, a change behind approval, a read that sounds like a change) and
long (four and five parts) — come from the tool-choice sets the live suite already uses — the store set
(``examples/06_scenarios/tool_choice.py``) and the held-out IT help desk and travel sets
(``tests/live/heldout.py``) — with the tools each needs written on the item
(``needed_tools``), so ``tool_coverage`` scores them. One more task reads a large result: the
help desk's ``ticket_export`` answers about 100,000 characters (a realistic export), past the
80,000 a ``ReAct`` pages instead of sending (``docs/tools.md#large-results``).

**Catalogues**: each task set's own tools, then *distractors* up to the size — synthetic but
realistic MCP-style tools (``<server>-<tool>``, the harness's MCP naming) of 25 servers a
company runs (code hosting, CRM, billing, HR, monitoring...), 20 tools each: list, get, create,
update and delete of four kinds of record, every one with a description and a JSON schema of
described, typed parameters (enums, bounds, required), the way an MCP server lists them. None
is about stock, the help desk or travel, so a task's needed tools stay the only right ones.
The order is fixed (a seeded shuffle), so a run is reproducible and a size's catalogue holds
the smaller sizes' tools.
"""

from __future__ import annotations

import importlib
import random
from dataclasses import dataclass, field
from typing import Any, Final

from tests.live import heldout

from trellis import tool
from trellis.contracts import ToolSpec
from trellis.harness.tools.base import Tool

scenario = importlib.import_module("examples.06_scenarios.tool_choice")

#: The catalogue sizes the benchmark runs by default.
SIZES: Final = (10, 50, 100, 500)
#: The seed of the distractors' order.
SEED: Final = 7


# --------------------------------------------------------------------------- distractors
@dataclass(frozen=True)
class Kind:
    """A kind of record a server keeps."""

    name: str
    plural: str
    example_id: str
    fields: str
    statuses: tuple[str, ...]


@dataclass(frozen=True)
class Server:
    name: str
    title: str
    kinds: tuple[Kind, ...]


def _k(name: str, plural: str, example: str, fields: str, statuses: str) -> Kind:
    return Kind(name, plural, example, fields, tuple(statuses.split()))


# fmt: off
SERVERS: Final = (
    Server("repo", "code hosting service", (
        _k("repository", "repositories", "web-app", "its default branch, visibility and topics", "active archived"),
        _k("pull_request", "pull requests", "PR-1042", "its author, reviewers, checks and diff size", "open merged closed"),
        _k("branch", "branches", "release-2.4", "its head commit and protection rules", "protected unprotected"),
        _k("release", "releases", "v2.4.1", "its tag, notes and assets", "draft published"),
    )),
    Server("tracker", "project tracker", (
        _k("issue", "issues", "ENG-311", "its assignee, labels, estimate and sprint", "todo in_progress done"),
        _k("sprint", "sprints", "SPR-48", "its dates, goal and committed points", "planned active closed"),
        _k("epic", "epics", "EP-12", "its child issues and progress", "open done"),
        _k("milestone", "milestones", "MS-3", "its due date and open issue count", "open closed"),
    )),
    Server("wiki", "team wiki", (
        _k("page", "pages", "PG-9001", "its title, body, author and last editor", "current draft archived"),
        _k("space", "spaces", "ENG", "its owners, permissions and page count", "open restricted"),
        _k("comment", "comments", "CM-552", "its author, text and the page it is on", "visible resolved"),
        _k("attachment", "attachments", "AT-77", "its file name, size and uploader", "current old"),
    )),
    Server("chat", "team chat", (
        _k("channel", "channels", "C-0042", "its topic, members and privacy", "public private archived"),
        _k("message", "messages", "M-88123", "its sender, text, reactions and thread", "sent edited"),
        _k("user_group", "user groups", "UG-oncall", "its handle and members", "enabled disabled"),
        _k("reminder", "reminders", "RM-19", "its text, time and recipient", "pending done"),
    )),
    Server("mail", "email service", (
        _k("email", "emails", "EM-4410", "its sender, recipients, subject and body", "unread read flagged"),
        _k("draft", "drafts", "DR-21", "its recipients, subject and body", "draft scheduled"),
        _k("label", "labels", "LB-finance", "its name, colour and email count", "shown hidden"),
        _k("filter", "filters", "FL-3", "its criteria and actions", "enabled disabled"),
    )),
    Server("calendar", "calendar service", (
        _k("event", "events", "EV-2210", "its time, attendees, room and video link", "confirmed tentative cancelled"),
        _k("calendar", "calendars", "CAL-team", "its owner, time zone and sharing", "shared private"),
        _k("room", "rooms", "RM-4B", "its capacity, floor and equipment", "bookable closed"),
        _k("booking_page", "booking pages", "BP-sales", "its slots, duration and buffer", "live paused"),
    )),
    Server("crm", "customer relationship manager", (
        _k("contact", "contacts", "CT-30211", "their name, email, company and owner", "lead customer churned"),
        _k("account", "accounts", "AC-881", "its industry, revenue tier and owner", "prospect customer former"),
        _k("deal", "deals", "DL-4410", "its amount, stage, close date and owner", "open won lost"),
        _k("lead", "leads", "LD-9902", "its source, score and owner", "new qualified disqualified"),
    )),
    Server("billing", "billing system", (
        _k("invoice", "invoices", "INV-2026-0412", "its customer, lines, tax and due date", "draft open paid void"),
        _k("subscription", "subscriptions", "SUB-771", "its plan, seats and renewal date", "active past_due cancelled"),
        _k("credit_note", "credit notes", "CN-55", "its invoice, amount and reason", "issued void"),
        _k("price_plan", "price plans", "PLAN-pro", "its price, interval and features", "active retired"),
    )),
    Server("payments", "payment processor", (
        _k("charge", "charges", "CH-90812", "its amount, currency, card and outcome", "succeeded failed pending"),
        _k("refund", "refunds", "RF-2231", "its charge, amount and reason", "pending succeeded failed"),
        _k("payout", "payouts", "PO-118", "its amount, bank account and arrival date", "pending paid failed"),
        _k("dispute", "disputes", "DP-41", "its charge, reason and evidence due date", "needs_response won lost"),
    )),
    Server("hr", "HR system", (
        _k("employee", "employees", "EMP-1209", "their title, manager, department and start date", "active on_leave terminated"),
        _k("leave_request", "leave requests", "LV-882", "its dates, type and approver", "pending approved rejected"),
        _k("department", "departments", "DEP-eng", "its head, cost centre and headcount", "active merged"),
        _k("performance_review", "performance reviews", "PR-2026-H1-77", "its rating, reviewer and goals", "draft submitted signed"),
    )),
    Server("recruiting", "applicant tracking system", (
        _k("candidate", "candidates", "CA-5521", "their CV, stage and source", "active hired rejected"),
        _k("job_posting", "job postings", "JOB-310", "its title, location, salary band and hiring manager", "open paused filled"),
        _k("interview", "interviews", "IV-7788", "its time, panel and scorecards", "scheduled done cancelled"),
        _k("offer", "offers", "OF-120", "its salary, start date and expiry", "draft sent accepted declined"),
    )),
    Server("docs", "document store", (
        _k("document", "documents", "DOC-7d2f", "its title, owner, size and last edit", "active trashed"),
        _k("folder", "folders", "FLD-legal", "its owner, parent and sharing", "active trashed"),
        _k("share_link", "share links", "SL-90", "its document, audience and expiry", "active revoked"),
        _k("template", "templates", "TPL-memo", "its title and fields", "published draft"),
    )),
    Server("analytics", "product analytics service", (
        _k("dashboard", "dashboards", "DB-growth", "its charts, filters and owner", "shared private"),
        _k("saved_report", "saved reports", "RP-retention", "its query, schedule and recipients", "active paused"),
        _k("metric", "metrics", "MT-dau", "its definition, unit and owner", "certified draft"),
        _k("segment", "segments", "SG-power-users", "its rules and size", "active archived"),
    )),
    Server("monitoring", "monitoring service", (
        _k("alert", "alerts", "AL-55120", "its monitor, severity and firing time", "firing resolved muted"),
        _k("monitor", "monitors", "MON-api-latency", "its query, thresholds and notification targets", "ok warn alert"),
        _k("slo", "service-level objectives", "SLO-checkout", "its target, window and error budget", "healthy at_risk breached"),
        _k("log_view", "saved log views", "LV-errors", "its query, columns and time range", "shared private"),
    )),
    Server("incidents", "incident manager", (
        _k("incident", "incidents", "INC-2026-118", "its severity, commander, timeline and impact", "investigating mitigated resolved"),
        _k("postmortem", "postmortems", "PM-118", "its causes, actions and owners", "draft in_review published"),
        _k("on_call_shift", "on-call shifts", "OC-2026-W43", "its rotation, person and hours", "upcoming active done"),
        _k("status_page", "status page notices", "SPN-31", "its components, message and audience", "scheduled live closed"),
    )),
    Server("cloud", "cloud infrastructure", (
        _k("instance", "virtual machines", "vm-0a91f", "its size, zone, image and tags", "running stopped terminated"),
        _k("bucket", "storage buckets", "logs-prod-eu", "its region, size, versioning and access policy", "active locked"),
        _k("load_balancer", "load balancers", "lb-web-1", "its listeners, targets and health checks", "active provisioning"),
        _k("snapshot", "disk snapshots", "snap-77ab", "its disk, size and creation time", "available pending"),
    )),
    Server("dns", "DNS and certificate service", (
        _k("zone", "DNS zones", "example.com", "its name servers and record count", "active pending"),
        _k("dns_record", "DNS records", "REC-4411", "its name, type, value and TTL", "active disabled"),
        _k("certificate", "TLS certificates", "CERT-2026-09", "its domains, issuer and expiry", "valid expiring expired"),
        _k("redirect", "redirects", "RD-12", "its source, target and status code", "active disabled"),
    )),
    Server("database", "managed database service", (
        _k("cluster", "database clusters", "pg-main", "its engine version, size and replicas", "available maintenance"),
        _k("backup", "backups", "BK-20261020", "its cluster, size and retention", "completed failed"),
        _k("migration", "schema migrations", "MG-0142", "its script, author and duration", "pending applied failed"),
        _k("db_user", "database users", "app_reader", "their roles and last login", "enabled locked"),
    )),
    Server("marketing", "marketing automation platform", (
        _k("campaign", "campaigns", "CP-autumn", "its channel, audience, budget and dates", "draft running finished"),
        _k("audience", "audiences", "AU-newsletter", "its rules and size", "active archived"),
        _k("email_template", "email templates", "ET-welcome", "its subject, body and variables", "published draft"),
        _k("landing_page", "landing pages", "LP-webinar", "its URL, form and conversion rate", "live draft"),
    )),
    Server("social", "social media manager", (
        _k("post", "posts", "PST-3301", "its text, media, network and engagement", "published scheduled draft"),
        _k("mention", "mentions", "MN-9012", "its author, text and sentiment", "new handled"),
        _k("social_account", "social accounts", "SA-company", "its network, handle and followers", "connected expired"),
        _k("content_queue", "content queues", "CQ-weekly", "its slots and pending posts", "active paused"),
    )),
    Server("legal", "contract management system", (
        _k("contract", "contracts", "CTR-2026-044", "its parties, value, term and renewal date", "draft signed expired"),
        _k("clause", "clause library entries", "CL-indemnity", "its text and fallback positions", "approved deprecated"),
        _k("nda", "NDAs", "NDA-883", "its counterparty, term and signatory", "sent signed expired"),
        _k("legal_matter", "legal matters", "LM-17", "its type, counsel and budget", "open closed"),
    )),
    Server("procurement", "procurement system", (
        _k("purchase_order", "purchase orders", "PO-2026-1203", "its vendor, lines, total and approver", "draft approved received"),
        _k("vendor", "vendors", "VN-acme", "its contacts, payment terms and risk rating", "approved blocked"),
        _k("quote_request", "requests for quote", "RFQ-88", "its items, invited vendors and deadline", "open awarded closed"),
        _k("requisition", "purchase requisitions", "REQ-551", "its requester, items and budget code", "pending approved rejected"),
    )),
    Server("facilities", "facilities manager", (
        _k("work_order", "maintenance work orders", "WO-3390", "its location, trade, priority and contractor", "open scheduled done"),
        _k("building", "buildings", "BLD-north", "its address, floors and capacity", "open closed"),
        _k("desk_booking", "desk bookings", "DSK-4-112", "its desk, person and date", "booked checked_in cancelled"),
        _k("parking_permit", "parking permits", "PK-0912", "its holder, vehicle and zone", "active expired"),
    )),
    Server("learning", "learning management system", (
        _k("course", "courses", "CRS-security-101", "its modules, duration and owner", "published draft retired"),
        _k("enrollment", "enrollments", "ENR-77120", "its learner, course and progress", "enrolled completed overdue"),
        _k("certification", "certifications", "CERTF-aws-saa", "its holder, issue and expiry dates", "valid expired"),
        _k("quiz", "quizzes", "QZ-31", "its questions and pass mark", "published draft"),
    )),
    Server("survey", "survey platform", (
        _k("survey", "surveys", "SV-enps-q4", "its questions, audience and close date", "draft open closed"),
        _k("survey_response", "survey responses", "SR-551920", "its answers, respondent and time", "complete partial"),
        _k("question_bank_item", "question bank entries", "QB-nps", "its text, type and options", "approved draft"),
        _k("panel", "respondent panels", "PNL-customers-eu", "its members and quotas", "active closed"),
    )),
)
# fmt: on


def _schema(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def _tools_of(server: Server) -> list[ToolSpec]:
    """A server's 20 tools, as an MCP server lists them."""
    specs = []
    for kind in server.kinds:
        ident = f"{kind.name}_id"
        id_param = {
            "type": "string",
            "description": f"The {kind.name.replace('_', ' ')}'s id, such as {kind.example_id}.",
        }
        status = {
            "type": "string",
            "enum": list(kind.statuses),
            "description": f"Only {kind.plural} in this status.",
        }
        noun = kind.name.replace("_", " ")
        made = [
            (
                f"list_{kind.name}s",
                f"List the {kind.plural} in the {server.title}, newest first, optionally only "
                f"those in one status. Returns up to `limit` of them and a cursor for the next "
                f"page.",
                "read",
                _schema(
                    {
                        "status": status,
                        "limit": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 100,
                            "default": 20,
                            "description": "The most to return.",
                        },
                        "cursor": {
                            "type": "string",
                            "description": "The cursor of the previous page, to read the next.",
                        },
                    },
                    [],
                ),
            ),
            (
                f"get_{kind.name}",
                f"Get one {noun} from the {server.title} by its id, with {kind.fields}.",
                "read",
                _schema({ident: id_param}, [ident]),
            ),
            (
                f"create_{kind.name}",
                f"Create a {noun} in the {server.title}. Returns the new {noun}'s id.",
                "write",
                _schema(
                    {
                        "title": {"type": "string", "description": f"The {noun}'s name or title."},
                        "details": {
                            "type": "string",
                            "description": f"What else to record: {kind.fields}.",
                        },
                        "owner": {
                            "type": "string",
                            "description": "Who owns it, as an email address.",
                        },
                    },
                    ["title"],
                ),
            ),
            (
                f"update_{kind.name}",
                f"Change an existing {noun} in the {server.title}: its title, details or "
                f"status. Only the fields given change.",
                "write",
                _schema(
                    {
                        ident: id_param,
                        "title": {"type": "string", "description": "The new name or title."},
                        "details": {"type": "string", "description": "The new details."},
                        "status": {**status, "description": "The new status."},
                    },
                    [ident],
                ),
            ),
            (
                f"delete_{kind.name}",
                f"Delete a {noun} from the {server.title} permanently. It cannot be undone.",
                "irreversible",
                _schema(
                    {
                        ident: id_param,
                        "reason": {
                            "type": "string",
                            "description": "Why it is deleted, kept in the audit log.",
                        },
                    },
                    [ident],
                ),
            ),
        ]
        specs.extend(
            ToolSpec(
                name=f"{server.name}-{name}",
                description=description,
                input_schema=schema,
                source="mcp",
                server=server.name,
                side_effects=effects,
            )
            for name, description, effects, schema in made
        )
    return specs


def distractor_specs() -> list[ToolSpec]:
    """Every distractor (500), in the benchmark's fixed order."""
    specs = [spec for server in SERVERS for spec in _tools_of(server)]
    random.Random(SEED).shuffle(specs)
    return specs


def _answer(spec: ToolSpec) -> Any:
    async def run(args: dict[str, Any]) -> Any:
        if spec.name.split("-", 1)[1].startswith("list_"):
            return {"items": [], "next_cursor": None}
        return {
            "ok": True,
            "tool": spec.name,
            **{k: v for k, v in args.items() if k.endswith("_id")},
        }

    return run


class Distractors:
    """A ``tools=`` source of the first ``count`` distractors: each answers a small result."""

    def __init__(self, count: int) -> None:
        self.specs = distractor_specs()[: max(count, 0)]

    async def resolve(self) -> list[Tool]:
        return [Tool(spec, _answer(spec)) for spec in self.specs]


# --------------------------------------------------------------------------- the large result
#: How many tickets the export holds, and roughly how long each line is: about 100,000
#: characters in all, past the 80,000 a ``ReAct`` pages.
EXPORT_TICKETS: Final = 1100
OLDEST: Final = "NET-0001"
TITLES: Final = (
    "VPN drops every hour on site B",
    "Switch port flapping in rack 4",
    "Wi-Fi slow on floor 3 after firmware update",
    "Printer queue stuck for accounting",
)


@tool(side_effects="read")
def ticket_export(team: str) -> str:
    """Export every ticket of a team, open and closed, as CSV lines, oldest first: id,
    opened date, priority, status, assignee and title. Large: use it when the whole history
    is needed, not for a count or a short list.

    Args:
        team: The team's name, such as network or desktop.
    """
    prefix = team[:3].upper() or "TKT"
    lines = ["id,opened,priority,status,assignee,title"]
    for n in range(1, EXPORT_TICKETS + 1):
        lines.append(
            f"{prefix}-{n:04d},2024-{1 + n % 12:02d}-{1 + n % 28:02d},"
            f"{('high', 'medium', 'low')[n % 3]},{('open', 'closed', 'waiting')[n % 3]},"
            f"agent{n % 17:02d},{TITLES[n % len(TITLES)]} (#{n})"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------- the tasks
@dataclass(frozen=True)
class TaskSet:
    """A domain: its instructions, its own tools and the tasks run on them."""

    name: str
    system: str
    tools: list[Any]
    tasks: list[dict[str, Any]] = field(default_factory=list)


def _task(item: dict[str, Any], name: str, length: str = "short") -> dict[str, Any]:
    return {**item, "metadata": {**item["metadata"], "task": name, "length": length}}


#: The long tasks: four and five parts, each served by its own tool.
LONG_HELP_DESK: Final = scenario.item(
    "For the network team: list its open high-priority tickets and its SLA breaches, tell me "
    "how to set up the VPN from the knowledge base, and who laptop LT-2290 is assigned to.",
    ["tickets_by_priority", "sla_breaches", "knowledge_base", "asset_owner"],
)
LONG_TRAVEL: Final = scenario.item(
    "For employee e-401: how many trips this year, which flights are booked, which expense "
    "claims are pending, and what the travel policy says about the meal allowance.",
    ["trip_summary", "flight_bookings", "expense_claims", "travel_policy"],
)


def task_sets() -> list[TaskSet]:
    """The three domains, ten tasks in all — six short (two parts, a change behind approval, a
    read that sounds like a change), three long (four and five parts) and the large export."""
    store_tools = [t for t in scenario.TOOLS if t.spec.name != "transfer_status"]  # it hangs
    data = scenario.DATASET
    five_parts = next(i for i in scenario.MORE if len(i["metadata"]["needed_tools"]) == 5)
    help_desk = heldout.HELPDESK
    travel = heldout.TRAVEL
    export = scenario.item(
        "Export all of the network team's tickets and tell me the id of the oldest one.",
        ["ticket_export"],
        expected=OLDEST,
    )
    return [
        TaskSet(
            "store",
            scenario.SYSTEM,
            store_tools,
            [
                _task(data[0], "store: specific tools, two parts"),
                _task(data[3], "store: a change behind approval"),
                _task(five_parts, "store: five parts", "long"),
            ],
        ),
        TaskSet(
            "help desk",
            heldout.HELPDESK_SYSTEM,
            [*heldout.HELPDESK_TOOLS, ticket_export],
            [
                _task(help_desk[0], "help desk: specific tools, two parts"),
                _task(help_desk[3], "help desk: a read that sounds like a change"),
                _task(LONG_HELP_DESK, "help desk: four parts", "long"),
                _task(export, "help desk: a large result (~100k characters)", "large"),
            ],
        ),
        TaskSet(
            "travel",
            heldout.TRAVEL_SYSTEM,
            heldout.TRAVEL_TOOLS,
            [
                _task(travel[0], "travel: specific tools, two parts"),
                _task(travel[3], "travel: a read that sounds like a change"),
                _task(LONG_TRAVEL, "travel: four parts", "long"),
            ],
        ),
    ]


#: The class suite's task classes (``--suite classes``), and the tasks ``mix`` marks for the
#: catalogues of 50 and 100 tools: four simple, three multi-step, three wide.
CLASSES: Final = ("simple", "multi", "wide", "large")
#: The teams whose ticket export the large tasks read, and its id prefix (``ticket_export``).
TEAMS: Final = (
    "network",
    "desktop",
    "security",
    "servers",
    "cloud",
    "database",
    "mobile",
    "printing",
    "identity",
    "storage",
)


def _classed(item: dict[str, Any], name: str, kind: str, *, mix: bool = False) -> dict[str, Any]:
    task = _task(item, name, kind)
    return {**task, "metadata": {**task["metadata"], "mix": mix}}


def class_sets() -> list[TaskSet]:
    """Ten tasks per class across the three domains — ``simple`` (one tool, a change behind
    approval, a read that sounds like a change, a record that does not exist, a capability),
    ``multi`` (two or three parts), ``wide`` (four or five parts, or four items), ``large`` (a
    ~100k-character export, the oldest or newest id of ten teams) — each a different task."""
    data, more = scenario.DATASET, scenario.MORE
    five_parts = next(i for i in more if len(i["metadata"]["needed_tools"]) == 5)
    hd, tr = heldout.HELPDESK, heldout.TRAVEL
    item = scenario.item
    store = [
        _classed(data[3], "store: a change behind approval", "simple"),
        _classed(data[4], "store: which store first", "simple", mix=True),
        _classed(more[1], "store: supplier delivery", "simple", mix=True),
        _classed(more[3], "store: the overview", "simple"),
        _classed(more[-1], "store: a read before a change", "simple"),
        _classed(data[0], "store: two parts", "multi", mix=True),
        _classed(data[1], "store: help and data", "multi"),
        _classed(data[2], "store: three parts", "multi"),
        _classed(
            item(
                "For store S40: which styles are in excess, and when does the supplier deliver "
                "more of ST-101?",
                ["excess_stock", "supplier_eta"],
            ),
            "store: excess and delivery",
            "multi",
        ),
        _classed(five_parts, "store: five parts", "wide", mix=True),
        _classed(
            item(
                "For store S40: what is in excess, which styles lost the most sales, what can be "
                "allocated, and what are its past allocations?",
                ["excess_stock", "lost_sales", "allocatable_stock", "allocation_history"],
            ),
            "store: four parts at S40",
            "wide",
        ),
        _classed(
            item(
                "For style ST-101: its weekly demand at S12, the supplier's next delivery, its "
                "stock at S40, and where in the app I create a stock rule.",
                ["store_demand", "supplier_eta", "style_stock", "app_help"],
            ),
            "store: four parts on a style",
            "wide",
        ),
        _classed(
            item(
                "Store S7: how is it doing overall, which styles lost the most sales, what is in "
                "excess, and what can be allocated?",
                ["store_overview", "lost_sales", "excess_stock", "allocatable_stock"],
            ),
            "store: four parts at S7",
            "wide",
        ),
    ]
    help_desk = [
        _classed(hd[2], "help desk: a change behind approval", "simple"),
        _classed(hd[3], "help desk: a read before a change", "simple", mix=True),
        _classed(hd[4], "help desk: no such record", "simple"),
        _classed(hd[0], "help desk: two parts", "multi", mix=True),
        _classed(hd[1], "help desk: knowledge and tickets", "multi"),
        _classed(
            item(
                "Who is laptop LT-2290 assigned to, and which of the network team's tickets "
                "breach their SLA?",
                ["asset_owner", "sla_breaches"],
            ),
            "help desk: owner and breaches",
            "multi",
        ),
        _classed(LONG_HELP_DESK, "help desk: four parts", "wide", mix=True),
        _classed(
            item(
                "For the desktop team: its open high-priority tickets, its SLA breaches, the "
                "knowledge base on printer setup, and who laptop LT-1001 is assigned to.",
                ["tickets_by_priority", "sla_breaches", "knowledge_base", "asset_owner"],
            ),
            "help desk: four parts for desktop",
            "wide",
        ),
        _classed(
            item(
                "Show me the desk dashboard, the security team's high-priority tickets, its SLA "
                "breaches, and how to reset MFA from the knowledge base.",
                ["desk_dashboard", "tickets_by_priority", "sla_breaches", "knowledge_base"],
            ),
            "help desk: four parts for security",
            "wide",
        ),
        *(
            _classed(
                item(
                    f"Export all of the {team} team's tickets and tell me the id of the "
                    f"{'oldest' if n % 2 == 0 else 'newest'} one.",
                    ["ticket_export"],
                    expected=f"{team[:3].upper()}-{1 if n % 2 == 0 else EXPORT_TICKETS:04d}",
                ),
                f"help desk: the {team} export",
                "large",
            )
            for n, team in enumerate(TEAMS)
        ),
    ]
    travel = [
        _classed(tr[2], "travel: a change behind approval", "simple"),
        _classed(tr[3], "travel: a read before a change", "simple", mix=True),
        _classed(tr[0], "travel: two parts", "multi", mix=True),
        _classed(tr[1], "travel: three parts", "multi"),
        _classed(
            item(
                "What does the travel policy say about hotels, and which expense claims does "
                "e-401 have pending?",
                ["travel_policy", "expense_claims"],
            ),
            "travel: policy and claims",
            "multi",
        ),
        _classed(LONG_TRAVEL, "travel: four parts", "wide", mix=True),
        _classed(
            item(
                "For employee e-512: trips this year, booked flights, pending expense claims and "
                "the travel policy on taxis.",
                ["trip_summary", "flight_bookings", "expense_claims", "travel_policy"],
            ),
            "travel: four parts for e-512",
            "wide",
        ),
        _classed(
            item(
                "For e-401 and for e-512: each one's booked flights and pending expense claims.",
                ["flight_bookings", "expense_claims"],
            ),
            "travel: four items for two employees",
            "wide",
        ),
    ]
    store_tools = [t for t in scenario.TOOLS if t.spec.name != "transfer_status"]  # it hangs
    return [
        TaskSet("store", scenario.SYSTEM, store_tools, store),
        TaskSet(
            "help desk",
            heldout.HELPDESK_SYSTEM,
            [*heldout.HELPDESK_TOOLS, ticket_export],
            help_desk,
        ),
        TaskSet("travel", heldout.TRAVEL_SYSTEM, heldout.TRAVEL_TOOLS, travel),
    ]


def catalogue(tools: list[Any], size: int) -> list[Any]:
    """A set's own tools and the distractors up to ``size`` tools (its own tools alone when
    they are already as many)."""
    return [*tools, Distractors(size - len(tools))]
