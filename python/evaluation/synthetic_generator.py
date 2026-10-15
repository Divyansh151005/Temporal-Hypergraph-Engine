#!/usr/bin/env python3
"""Reverse-generation of text / temporal-hypergraph pairs with multi-hop QA.

1. A seeded RNG builds a ground-truth hypergraph for one of four domains
   (corporate M&A, executive roles, customer support, medical care). Each
   graph contains a 2-3 hop chain from a seed entity to a fact that is later
   superseded (a CEO replaced, an on-call rotation, a new department head), so
   questions "as of" a date have a correct answer and a stale distractor.
2. An LLM writes an article, transcript or narrative conveying the graph
   (``--offline`` uses deterministic templates instead). Articles must mention
   every entity by label or alias, or they are regenerated.
3. ``--noise`` injects typos, alias swaps and irrelevant paragraphs.
4. Multi-hop QA items (question, gold answer, ``as_of``, seeds, gold edge
   ids, stale answer) are emitted for the benchmark suite.

Labels are unique across a dataset: names come from large combinatorial pools
and are allocated per document index, so documents never contradict each
other about the same entity.

    python python/evaluation/synthetic_generator.py --n 1000 --workers 8 --out datasets/synthetic/corp.jsonl
    python python/evaluation/synthetic_generator.py --n 100 --golden --out datasets/core_100_golden_dataset.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import random
import re
import sys
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any

from hypergraph_evaluation.llm import LLM, LLMError
from hypergraph_evaluation.schema import validate
from hypergraph_evaluation.temporal import add_months, day_phrase, epoch_of, month_phrase, to_iso

log = logging.getLogger("synthetic_generator")

DAY = 86_400
YEAR = 365 * DAY
DOMAINS = ("corporate_ma", "executive_roles", "customer_support", "medical_care")

# --- name pools ---------------------------------------------------------------

_FIRST_A = "Al Bri Ca Da El Fa Ga Ha Is Jo Ka Le Ma Na Ol Pe Ra Sa Ta Vi".split()
_FIRST_B = "na ra lia den mir ssa vin ron ela sha tan ley nor ric ya".split()
_LAST_A = "Ash Bren Cor Dal Ever Fen Gar Hol Ing Kel Lind Mor Nor Oak Pell Quin Ros Stan Thorn Vance Wex Yar Brook Hart Mar".split()
_LAST_B = "ford wick ton berg ley more field stone croft well ridge worth by man son gate holm dale wood shaw".split()
_BRAND_A = "Vant Nor Hel Cob Sum Aur Key Mer Atl Pin Cres Ever Iron Lum Saph Red Quan Silv Blu Harb Gran Falc Ori Bea Nim".split()
_BRAND_B = "ex ora ix alt ira eon ova ari yx enta ium os ara ella ion ica ent ary uma ida".split()
_SECTORS = (
    "Dynamics Systems Labs Logistics Analytics Robotics Energy Networks Health Foods "
    "Capital Materials Media Security Software Biotech Motors Telecom Retail Aerospace"
).split()
_PRODUCT_LINES = "Pro Air Max One Edge Flex Nova Core Pulse Lite".split()
_TEAMS = "Falcon Orion Cedar Delta Harbor Summit Willow Ember Atlas Comet Juniper Raven".split()
CITIES = (
    "Lisbon Toronto Denver Austin Oslo Dublin Seattle Lyon Porto Zurich Leeds Perth Osaka "
    "Krakow Tallinn Valencia Boston Calgary Utrecht Gothenburg Bilbao Aarhus Turin Ghent"
).split()
DRUGS = (
    "metformin lisinopril atorvastatin amlodipine sertraline levothyroxine omeprazole "
    "losartan gabapentin rosuvastatin bisoprolol apixaban"
).split()
_TITLES = ("CEO", "CFO", "CTO", "COO")
_DEPARTMENTS = ("cardiology", "oncology", "neurology", "orthopedics", "pediatrics")

_DISTRACTORS = [
    "Analysts noted that broader market conditions remained volatile throughout the period, "
    "with interest-rate expectations shifting several times.",
    "Separately, an industry survey found that most respondents expected hiring to remain flat "
    "over the coming year.",
    "The weather in the region was unusually mild that season, and several outdoor events drew "
    "record attendance.",
    "Commentators have long debated whether such arrangements create lasting value or merely "
    "redistribute existing resources.",
    "A spokesperson declined to comment on unrelated rumours circulating on social media.",
    "Local officials also announced a road-maintenance programme scheduled to start in the spring.",
]


@dataclass(frozen=True)
class Entity:
    id: str
    label: str
    type: str
    aliases: tuple[str, ...] = ()

    def to_json(self) -> dict[str, Any]:
        return {"id": self.id, "label": self.label, "type": self.type, "aliases": list(self.aliases)}


@dataclass
class Edge:
    id: str
    event_type: str
    participants: list[Entity]
    roles: list[str]
    valid_from: int | None
    valid_to: int | None
    summary: str
    sentence: str

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "event_type": self.event_type,
            "participants": [p.id for p in self.participants],
            "roles": self.roles,
            "valid_from": self.valid_from,
            "valid_to": self.valid_to,
            "summary": self.summary,
        }


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


@cache
def _pool(kind: str, seed: int) -> tuple[str, ...]:
    """A seeded permutation of every name of ``kind``."""
    combos: list[str]
    if kind == "person":
        combos = [f"{a}{b} {c}{d}" for a, b, c, d in itertools.product(_FIRST_A, _FIRST_B, _LAST_A, _LAST_B)]
    elif kind == "brand":
        combos = [f"{a}{b}" for a, b in itertools.product(_BRAND_A, _BRAND_B)]
    elif kind == "org":
        combos = [f"{a}{b} {s}" for a, b, s in itertools.product(_BRAND_A, _BRAND_B, _SECTORS)]
    else:
        raise ValueError(kind)
    random.Random(f"{seed}:{kind}").shuffle(combos)
    return tuple(combos)


class Names:
    """Allocates dataset-unique names: document ``index`` owns a fixed slice of each pool."""

    SLOTS = {"person": 8, "org": 8, "brand": 4}

    def __init__(self, seed: int, index: int) -> None:
        self.seed = seed
        self.index = index
        self.used = {kind: 0 for kind in self.SLOTS}

    def take(self, kind: str) -> str:
        slot = self.used[kind]
        if slot >= self.SLOTS[kind]:
            raise RuntimeError(f"document needs more than {self.SLOTS[kind]} {kind} names")
        self.used[kind] += 1
        pool = _pool(kind, self.seed)
        return pool[(self.index * self.SLOTS[kind] + slot) % len(pool)]


@dataclass
class Builder:
    """Per-document state shared by the domain builders."""

    rng: random.Random
    names: Names
    doc_key: str
    entities: list[Entity] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)
    qa: list[dict[str, Any]] = field(default_factory=list)

    def entity(self, prefix: str, label: str, type_: str, aliases: Sequence[str] = ()) -> Entity:
        taken = {a.lower() for e in self.entities for a in (e.label, *e.aliases)}
        kept = tuple(a for a in aliases if a.lower() not in taken and a.lower() != label.lower())
        entity = Entity(f"{prefix}:{slug(label)}", label, type_, kept)
        self.entities.append(entity)
        return entity

    def person(self, title: str = "") -> Entity:
        name = self.names.take("person")
        first, last = name.split(" ")
        aliases = [f"{first[0]}. {last}"] + ([f"{title} {last}"] if title else [last])
        return self.entity("person", name, "PERSON", aliases)

    def org(self) -> Entity:
        name = self.names.take("org")
        brand = name.split(" ")[0]
        return self.entity("org", name, "ORGANIZATION", [brand])

    def branded(self, prefix: str, suffix: str, type_: str, alias_suffix: str | None = None) -> Entity:
        brand = self.names.take("brand")
        aliases = [f"{brand} {alias_suffix}"] if alias_suffix else []
        return self.entity(prefix, f"{brand} {suffix}", type_, aliases)

    def shared(self, prefix: str, label: str, type_: str) -> Entity:
        existing = next((e for e in self.entities if e.label == label), None)
        return existing or self.entity(prefix, label, type_)

    def edge(
        self,
        event_type: str,
        members: Sequence[tuple[Entity, str]],
        valid_from: int | None,
        valid_to: int | None,
        summary: str,
        sentence: str,
    ) -> Edge:
        edge = Edge(
            id=f"edge:{self.doc_key}:{len(self.edges) + 1}",
            event_type=event_type,
            participants=[m for m, _ in members],
            roles=[r for _, r in members],
            valid_from=valid_from,
            valid_to=valid_to,
            summary=summary,
            sentence=sentence,
        )
        self.edges.append(edge)
        return edge

    def when(self, low_year: int, high_year: int, *, day: bool = False) -> int:
        """A random date; the first of a month unless ``day``."""
        return epoch_of(
            self.rng.randint(low_year, high_year),
            self.rng.randint(1, 12),
            self.rng.randint(1, 28) if day else 1,
        )

    def months(self, epoch: int, low: int, high: int) -> int:
        """The first of the month ``low..high`` months after ``epoch`` (negative: before)."""
        return add_months(epoch, self.rng.randint(low, high))

    def between(self, low: int, high: int) -> int:
        """A day strictly inside ``(low, high)``, at midnight UTC."""
        if high - low < 2 * DAY:
            raise ValueError("interval too short")
        return (self.rng.randint(low + DAY, high - DAY) // DAY) * DAY

    def ask(
        self,
        question: str,
        answer: Entity,
        stale: Entity | None,
        as_of: int,
        seeds: Sequence[Entity],
        gold: Sequence[Edge],
    ) -> None:
        self.qa.append(
            {
                "id": f"qa:{self.doc_key}:{len(self.qa) + 1}",
                "question": question,
                "answer": answer.label,
                "answer_id": answer.id,
                "as_of": as_of,
                "seeds": [s.id for s in seeds],
                "hops": len(gold),
                "gold_edge_ids": [e.id for e in gold],
                "stale_answer": stale.label if stale else None,
                "stale_answer_id": stale.id if stale else None,
            }
        )

    def supersession_questions(
        self,
        template: str,
        seeds: Sequence[Entity],
        chain: Sequence[Edge],
        old: Edge,
        new: Edge,
        old_holder: Entity,
        new_holder: Entity,
        phrase: Callable[[int], str] = month_phrase,
    ) -> None:
        """One question while ``old`` held and one after ``new`` took over."""
        start = max(e.valid_from or 0 for e in [*chain, old])
        assert new.valid_from is not None
        before = self.between(start, new.valid_from)
        after = self.between(new.valid_from, new.valid_from + 2 * YEAR)
        self.ask(template.format(date=phrase(before)), old_holder, new_holder, before, seeds, [*chain, old])
        self.ask(template.format(date=phrase(after)), new_holder, old_holder, after, seeds, [*chain, new])


# --- domains ------------------------------------------------------------------


def _since(epoch: int | None, phrase: Callable[[int], str] = month_phrase) -> str:
    return f"since {phrase(epoch)}" if epoch is not None else "for some time"


def build_corporate_ma(b: Builder) -> str:
    investor, acquirer, target, advisor = b.org(), b.org(), b.org(), b.org()
    ceo_old, ceo_new = b.person(), b.person()
    city = b.shared("loc", b.rng.choice(CITIES), "LOCATION")
    t_inv = b.when(2010, 2014)
    t_acq = b.months(t_inv, 12, 40)
    t_succ = b.months(t_acq, 12, 40)
    t_old = None if b.rng.random() < 0.2 else b.months(t_acq, -60, -12)
    stake = b.rng.choice([5, 8, 12, 15, 20, 25])

    e1 = b.edge(
        "INVESTMENT", [(investor, "INVESTOR"), (acquirer, "INVESTEE")], t_inv, None,
        f"{investor.label} took a {stake}% stake in {acquirer.label}.",
        f"In {month_phrase(t_inv)}, {investor.label} took a {stake}% stake in {acquirer.label}.",
    )
    e2 = b.edge(
        "ACQUISITION", [(acquirer, "ACQUIRER"), (target, "TARGET"), (advisor, "ADVISOR")], t_acq, None,
        f"{acquirer.label} acquired {target.label}, advised by {advisor.label}.",
        f"{acquirer.label} completed its acquisition of {target.label} in {month_phrase(t_acq)}, "
        f"with {advisor.label} acting as financial adviser.",
    )
    e3 = b.edge(
        "EXECUTIVE_ROLE", [(ceo_old, "CEO"), (target, "EMPLOYER")], t_old, t_succ,
        f"{ceo_old.label} was CEO of {target.label}.",
        f"{ceo_old.label} had led {target.label} as chief executive "
        f"{'from ' + month_phrase(t_old) if t_old else ''} until stepping down in {month_phrase(t_succ)}.".replace("  ", " "),
    )
    e4 = b.edge(
        "EXECUTIVE_ROLE", [(ceo_new, "CEO"), (target, "EMPLOYER")], t_succ, None,
        f"{ceo_new.label} became CEO of {target.label}.",
        f"{ceo_new.label} succeeded {ceo_old.aliases[-1] if ceo_old.aliases else ceo_old.label} as CEO "
        f"of {target.label} in {month_phrase(t_succ)}.",
    )
    b.edge(
        "HEADQUARTERS", [(target, "COMPANY"), (city, "LOCATION")], None, None,
        f"{target.label} is headquartered in {city.label}.",
        f"{target.label} is based in {city.label}.",
    )
    b.supersession_questions(
        f"As of {{date}}, who was the CEO of the company acquired by the firm {investor.label} invested in?",
        [investor], [e1, e2], e3, e4, ceo_old, ceo_new,
    )
    return "business news article"


def build_executive_roles(b: Builder) -> str:
    director = b.person("Dr.")
    org_x, org_y, org_prev = b.org(), b.org(), b.org()
    exec_old, exec_new = b.person(), b.person()
    city = b.shared("loc", b.rng.choice(CITIES), "LOCATION")
    title = b.rng.choice(_TITLES)
    t_prev = b.when(2008, 2012)
    t_board = b.months(t_prev, 24, 48)
    t_part = b.months(t_board, 12, 30)
    t_old = b.months(t_part, -36, -12)
    t_succ = b.months(t_part, 12, 40)

    b.edge(
        "EMPLOYMENT", [(director, "EMPLOYEE"), (org_prev, "EMPLOYER")], t_prev, t_board,
        f"{director.label} worked at {org_prev.label}.",
        f"{director.label} spent several years at {org_prev.label}, from {month_phrase(t_prev)} to {month_phrase(t_board)}.",
    )
    e1 = b.edge(
        "BOARD_MEMBERSHIP", [(director, "DIRECTOR"), (org_x, "COMPANY")], t_board, None,
        f"{director.label} joined the board of {org_x.label}.",
        f"In {month_phrase(t_board)}, {director.label} joined the board of directors of {org_x.label}.",
    )
    e2 = b.edge(
        "PARTNERSHIP", [(org_x, "PARTNER"), (org_y, "PARTNER"), (city, "LOCATION")], t_part, None,
        f"{org_x.label} and {org_y.label} formed a partnership in {city.label}.",
        f"{org_x.label} and {org_y.label} announced a strategic partnership in {city.label} in {month_phrase(t_part)}.",
    )
    e3 = b.edge(
        "EXECUTIVE_ROLE", [(exec_old, title), (org_y, "EMPLOYER")], t_old, t_succ,
        f"{exec_old.label} was {title} of {org_y.label}.",
        f"{exec_old.label} served as {title} of {org_y.label} from {month_phrase(t_old)} to {month_phrase(t_succ)}.",
    )
    e4 = b.edge(
        "EXECUTIVE_ROLE", [(exec_new, title), (org_y, "EMPLOYER")], t_succ, None,
        f"{exec_new.label} became {title} of {org_y.label}.",
        f"{exec_new.label} was named {title} of {org_y.label} in {month_phrase(t_succ)}.",
    )
    b.supersession_questions(
        f"As of {{date}}, who was the {title} of the company that partnered with the firm on whose "
        f"board {director.label} sits?",
        [director], [e1, e2], e3, e4, exec_old, exec_new,
    )
    return "executive profile in a trade publication"


def build_customer_support(b: Builder) -> str:
    customer = b.person()
    product = b.branded("product", f"{b.rng.choice(_PRODUCT_LINES)} {b.rng.randint(2, 9)}", "PRODUCT")
    ticket_no = 10_000 + b.names.index * 7 + b.rng.randint(0, 6)
    ticket = b.entity("ticket", f"ticket #{ticket_no}", "EVENT", [f"case {ticket_no}"])
    team = b.branded("team", f"{b.rng.choice(_TEAMS)} support team", "ORGANIZATION")
    agent_old, agent_new = b.person(), b.person()
    city = b.shared("loc", b.rng.choice(CITIES), "LOCATION")
    t_buy = b.when(2021, 2023, day=True)
    t_ticket = t_buy + b.rng.randint(20, 200) * DAY
    t_escalate = t_ticket + b.rng.randint(1, 5) * DAY
    t_shift_old = t_escalate - b.rng.randint(3, 20) * DAY
    t_shift_new = t_escalate + b.rng.randint(5, 30) * DAY

    b.edge(
        "PURCHASE", [(customer, "CUSTOMER"), (product, "PRODUCT"), (city, "STORE_LOCATION")], t_buy, None,
        f"{customer.label} bought a {product.label} in {city.label}.",
        f"{customer.label} bought a {product.label} at the {city.label} store on {day_phrase(t_buy)}.",
    )
    e1 = b.edge(
        "SUPPORT_TICKET", [(customer, "CUSTOMER"), (ticket, "TICKET"), (product, "PRODUCT")], t_ticket, None,
        f"{customer.label} opened {ticket.label} about the {product.label}.",
        f"On {day_phrase(t_ticket)}, {customer.label} opened {ticket.label} reporting that the {product.label} would not charge.",
    )
    e2 = b.edge(
        "ESCALATION", [(ticket, "TICKET"), (team, "ASSIGNED_TEAM")], t_escalate, None,
        f"{ticket.label} was escalated to the {team.label}.",
        f"{ticket.label} was escalated to the {team.label} on {day_phrase(t_escalate)}.",
    )
    e3 = b.edge(
        "ON_CALL", [(agent_old, "AGENT"), (team, "TEAM")], t_shift_old, t_shift_new,
        f"{agent_old.label} was the on-call agent for the {team.label}.",
        f"{agent_old.label} covered on-call duty for the {team.label} from {day_phrase(t_shift_old)} until {day_phrase(t_shift_new)}.",
    )
    e4 = b.edge(
        "ON_CALL", [(agent_new, "AGENT"), (team, "TEAM")], t_shift_new, None,
        f"{agent_new.label} took over on-call duty for the {team.label}.",
        f"{agent_new.label} took over the {team.label} on-call rotation on {day_phrase(t_shift_new)}.",
    )
    b.supersession_questions(
        f"As of {{date}}, which agent was on call for the team handling the ticket {customer.label} opened?",
        [customer], [e1, e2], e3, e4, agent_old, agent_new, phrase=day_phrase,
    )
    return "customer-support case log"


def build_medical_care(b: Builder) -> str:
    patient = b.person()
    gp = b.person("Dr.")
    clinic = b.branded("org", "Family Clinic", "ORGANIZATION", "clinic")
    hospital = b.branded("org", "General Hospital", "ORGANIZATION", "Hospital")
    head_old, head_new = b.person("Dr."), b.person("Dr.")
    department = b.rng.choice(_DEPARTMENTS)
    drug_old, drug_new = (b.shared("drug", d, "PRODUCT") for d in b.rng.sample(DRUGS, 2))
    t_reg = b.when(2015, 2018)
    t_aff = b.months(t_reg, 1, 30)
    t_head_old = b.months(t_aff, -48, -12)
    t_head_new = b.months(t_aff, 12, 40)
    t_rx = b.months(t_reg, 2, 13)
    t_rx_change = b.months(t_rx, 12, 36)

    e1 = b.edge(
        "PRIMARY_CARE", [(patient, "PATIENT"), (gp, "PHYSICIAN"), (clinic, "FACILITY")], t_reg, None,
        f"{patient.label} is a patient of {gp.label} at {clinic.label}.",
        f"{patient.label} registered with {gp.label} at {clinic.label} in {month_phrase(t_reg)}.",
    )
    e2 = b.edge(
        "AFFILIATION", [(clinic, "CLINIC"), (hospital, "HOSPITAL")], t_aff, None,
        f"{clinic.label} became affiliated with {hospital.label}.",
        f"In {month_phrase(t_aff)}, {clinic.label} became formally affiliated with {hospital.label}.",
    )
    e3 = b.edge(
        "DEPARTMENT_HEAD", [(head_old, f"HEAD_OF_{department.upper()}"), (hospital, "HOSPITAL")], t_head_old, t_head_new,
        f"{head_old.label} headed {department} at {hospital.label}.",
        f"{head_old.label} headed the {department} department at {hospital.label} from {month_phrase(t_head_old)} "
        f"to {month_phrase(t_head_new)}.",
    )
    e4 = b.edge(
        "DEPARTMENT_HEAD", [(head_new, f"HEAD_OF_{department.upper()}"), (hospital, "HOSPITAL")], t_head_new, None,
        f"{head_new.label} became head of {department} at {hospital.label}.",
        f"{head_new.label} took over as head of {department} at {hospital.label} in {month_phrase(t_head_new)}.",
    )
    b.edge(
        "PRESCRIPTION", [(gp, "PRESCRIBER"), (patient, "PATIENT"), (drug_old, "MEDICATION")], t_rx, t_rx_change,
        f"{gp.label} prescribed {drug_old.label} to {patient.label}.",
        f"{gp.label} started {patient.label} on {drug_old.label} in {month_phrase(t_rx)}.",
    )
    b.edge(
        "PRESCRIPTION", [(gp, "PRESCRIBER"), (patient, "PATIENT"), (drug_new, "MEDICATION")], t_rx_change, None,
        f"{gp.label} switched {patient.label} to {drug_new.label}.",
        f"In {month_phrase(t_rx_change)}, {patient.label} was switched from {drug_old.label} to {drug_new.label}.",
    )
    b.supersession_questions(
        f"As of {{date}}, who headed {department} at the hospital affiliated with {patient.label}'s clinic?",
        [patient], [e1, e2], e3, e4, head_old, head_new,
    )
    return "patient care narrative"


BUILDERS: dict[str, Callable[[Builder], str]] = {
    "corporate_ma": build_corporate_ma,
    "executive_roles": build_executive_roles,
    "customer_support": build_customer_support,
    "medical_care": build_medical_care,
}


# --- articles -----------------------------------------------------------------


def template_article(builder: Builder, rng: random.Random) -> str:
    """Deterministic article: edge sentences in timeline order, split into paragraphs."""
    ordered = sorted(builder.edges, key=lambda e: (e.valid_from or 0, e.id))
    sentences = [e.sentence for e in ordered]
    cut = max(1, len(sentences) // 2 + rng.randint(-1, 1))
    return " ".join(sentences[:cut]) + "\n\n" + " ".join(sentences[cut:])


def missing_mentions(text: str, entities: Sequence[Entity]) -> list[str]:
    lowered = text.lower()
    return [
        e.label
        for e in entities
        if not any(re.search(rf"(?<!\w){re.escape(n.lower())}(?!\w)", lowered) for n in (e.label, *e.aliases))
    ]


ARTICLE_SYSTEM = """You write realistic {style}s from structured facts.
Rules:
- Convey every hyperedge in the JSON graph accurately, including its dates. Phrase dates naturally \
("in March 2019", "on 5 March 2019"); never print ISO timestamps or epoch numbers.
- A null valid_from means the start date is unknown: do not invent one. A null valid_to means the fact still holds.
- Mention every entity by its label or one of its aliases.
- Do not introduce other people, organizations, dates or facts that contradict the graph.
- 150 to 400 words of plain prose (no lists, no headings, no JSON inside the article).
Return a JSON object {{"article": "..."}}."""


def graph_prompt(builder: Builder, reference_time: int) -> str:
    edges = [
        {**e.to_json(), "valid_from": to_iso(e.valid_from), "valid_to": to_iso(e.valid_to)}
        for e in builder.edges
    ]
    payload = {
        "reference_time": to_iso(reference_time),
        "entities": [e.to_json() for e in builder.entities],
        "hyperedges": edges,
    }
    return "Write the text for this temporal hypergraph:\n" + json.dumps(payload, indent=1)


def llm_article(
    builder: Builder, style: str, reference_time: int, llm: LLM, max_attempts: int, seed: int
) -> tuple[str, int, str]:
    """The article, the attempts it took and the ``provider:model`` that wrote it."""
    system = ARTICLE_SYSTEM.format(style=style)
    user = graph_prompt(builder, reference_time)
    missing: list[str] = []
    for attempt in range(1, max_attempts + 1):
        prompt = user
        if missing:
            prompt += "\n\nYour previous draft did not mention: " + ", ".join(missing) + ". Mention every entity."
        reply, writer = llm.complete_json_with_source(system, prompt, "article", temperature=0.7, seed=seed + attempt)
        article = reply["article"]
        missing = missing_mentions(article, builder.entities)
        if not missing:
            return article, attempt, writer
    raise LLMError(f"article still misses {missing} after {max_attempts} attempts")


# --- noise --------------------------------------------------------------------


def _typo(word: str, rng: random.Random) -> str:
    i = rng.randrange(1, len(word) - 1)
    match rng.randrange(3):
        case 0:
            return word[:i] + word[i + 1] + word[i] + word[i + 2:]
        case 1:
            return word[:i] + word[i + 1:]
        case _:
            return word[:i] + word[i] + word[i:]


def inject_noise(text: str, entities: Sequence[Entity], rate: float, rng: random.Random) -> str:
    """Alias swaps (rate), typos (rate / 6 of words) and irrelevant paragraphs (about 3 * rate)."""
    if rate <= 0:
        return text
    for entity in entities:
        if entity.aliases and rng.random() < rate:
            pattern = re.compile(rf"(?<!\w){re.escape(entity.label)}(?!\w)")
            matches = list(pattern.finditer(text))
            # Keep the first mention so the entity stays identifiable.
            for match in reversed(matches[1:]):
                text = text[: match.start()] + rng.choice(entity.aliases) + text[match.end():]
    words = re.split(r"(\s+)", text)
    for i, word in enumerate(words):
        if len(word) > 4 and word.isalpha() and rng.random() < rate / 6:
            words[i] = _typo(word, rng)
    paragraphs = "".join(words).split("\n\n")
    for _ in range(max(1, round(3 * rate))):
        paragraphs.insert(rng.randint(0, len(paragraphs)), rng.choice(_DISTRACTORS))
    return "\n\n".join(paragraphs)


# --- records ------------------------------------------------------------------


def generate_record(
    index: int,
    *,
    seed: int = 7,
    domains: Sequence[str] = DOMAINS,
    llm: LLM | None = None,
    noise: float = 0.0,
    max_attempts: int = 3,
) -> dict[str, Any]:
    """One dataset record; ``llm=None`` writes a template article."""
    domain = domains[index % len(domains)]
    rng = random.Random(f"{seed}:{index}")
    doc_key = f"{seed}-{index:05d}"
    builder = Builder(rng, Names(seed, index), doc_key)
    style = BUILDERS[domain](builder)
    latest = max(t for e in builder.edges for t in (e.valid_from, e.valid_to) if t is not None)
    latest = max([latest, *(q["as_of"] for q in builder.qa)])
    reference_time = latest + rng.randint(30, 365) * DAY

    if llm is None:
        article, attempts, writer = template_article(builder, rng), 0, "template"
    else:
        article, attempts, writer = llm_article(builder, style, reference_time, llm, max_attempts, seed * 100_003 + index)
    if noise:
        article = inject_noise(article, builder.entities, noise, rng)

    record = {
        "dataset_id": f"syn-{domain.replace('_', '-')}-{doc_key}",
        "domain": domain,
        "reference_time": reference_time,
        "raw_text": article,
        "noise": noise,
        "ground_truth_graph": {
            "nodes": [e.to_json() for e in builder.entities],
            "hyperedges": [e.to_json() for e in builder.edges],
        },
        "qa": builder.qa,
        "generator": {"writer": writer, "seed": seed, "index": index, "attempts": attempts},
    }
    validate("hypergraph", record)
    return record


def generate_dataset(
    n: int,
    *,
    seed: int = 7,
    domains: Sequence[str] = DOMAINS,
    llm: LLM | None = None,
    noise: float = 0.0,
    workers: int = 1,
    max_attempts: int = 3,
) -> list[dict[str, Any]]:
    """``n`` records in index order. Documents whose article cannot be verified
    are replaced by later indices, up to ``n`` extra attempts."""

    def attempt(index: int) -> dict[str, Any] | None:
        try:
            return generate_record(index, seed=seed, domains=domains, llm=llm, noise=noise, max_attempts=max_attempts)
        except LLMError as error:
            log.warning("document %d skipped: %s", index, error)
            return None

    records: list[dict[str, Any]] = []
    next_index = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        while len(records) < n and next_index < 2 * n:
            batch = range(next_index, min(next_index + n - len(records), 2 * n))
            next_index = batch.stop
            records += [r for r in pool.map(attempt, batch) if r is not None]
            log.info("%d / %d documents", len(records), n)
    if len(records) < n:
        raise RuntimeError(f"only {len(records)} of {n} documents could be generated")
    return records


def write_records(records: Sequence[dict[str, Any]], path: Path, *, as_array: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if as_array:
        path.write_text(json.dumps(list(records), indent=1) + "\n")
    else:
        path.write_text("".join(json.dumps(r) + "\n" for r in records))


def load_records(path: Path) -> list[dict[str, Any]]:
    """Reads a JSON array or JSONL dataset."""
    text = path.read_text()
    if text.lstrip().startswith("["):
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=1000, help="documents to generate")
    parser.add_argument("--out", type=Path, required=True, help=".jsonl, or .json for an array")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--workers", type=int, default=8, help="parallel LLM requests")
    parser.add_argument("--noise", type=float, default=0.0, help="adversarial noise rate in [0, 1]")
    parser.add_argument("--domains", default=",".join(DOMAINS), help="comma-separated subset of " + ", ".join(DOMAINS))
    parser.add_argument("--golden", action="store_true", help="noise-free golden set written as a JSON array")
    parser.add_argument("--offline", action="store_true", help="template articles instead of the LLM")
    parser.add_argument("--max-attempts", type=int, default=3, help="article regenerations per document")
    parser.add_argument("--llm", help="provider chain, e.g. xai:grok-4.7,groq:qwen/qwen3.8-27b "
                        "(default: $GENERATOR_LLM, else the built-in generator chain)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    domains = [d.strip() for d in args.domains.split(",") if d.strip()]
    unknown = sorted(set(domains) - set(DOMAINS))
    if unknown:
        parser.error(f"unknown domains {unknown}")
    if not 0 <= args.noise <= 1:
        parser.error("--noise must be in [0, 1]")
    noise = 0.0 if args.golden else args.noise
    llm = None
    if not args.offline:
        llm = LLM.for_role("generator", args.llm)
        log.info("writing articles with %s", llm.label)
    records = generate_dataset(
        args.n, seed=args.seed, domains=domains, llm=llm, noise=noise,
        workers=args.workers, max_attempts=args.max_attempts,
    )
    write_records(records, args.out, as_array=args.golden or args.out.suffix == ".json")
    questions = sum(len(r["qa"]) for r in records)
    print(f"wrote {len(records)} documents and {questions} questions to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
