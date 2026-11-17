"""Build OOD benchmark and intents_v3 augmentation datasets."""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"

V2_PATH = DATA / "intents_v2.jsonl"
OOD_PATH = DATA / "intent_ood_test.jsonl"
V3_PATH = DATA / "intents_v3.jsonl"
OOD_REPORT = ARTIFACTS / "ood_dataset_report.json"
V3_REPORT = ARTIFACTS / "intents_v3_report.json"

rng = random.Random(20260322)


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def uniq_extend(
    bucket: list[str],
    seen: set[str],
    candidates: list[str],
    blocked: set[str],
) -> None:
    for text in candidates:
        key = norm(text)
        if not key or key in seen or key in blocked:
            continue
        seen.add(key)
        bucket.append(text.strip())


# ---------------------------------------------------------------------------
# OOD generators (evaluation only)
# ---------------------------------------------------------------------------

def ood_ecommerce(n: int, blocked: set[str]) -> list[str]:
    brands = [
        "adidas", "puma", "reebok", "skechers", "asics", "new balance",
        "xiaomi", "oneplus", "motorola", "sony", "bose", "jbl", "anker",
        "decathlon", "unbranded", "local-market",
    ]
    products = [
        "running shoes", "casual sneakers", "wireless earbuds", "over-ear headphones",
        "android phone", "tablet", "power bank", "smartwatch", "backpack",
        "rain jacket", "kitchen blender", "air fryer", "study lamp", "laptop stand",
        "mechanical keyboard", "webcam", "fitness band",
    ]
    prices = [
        "under 1500", "below 2999", "around 2k", "under 30k", "below fifty euros",
        "under 80 dollars", "around four thousand rupees", "max budget 12000",
        "cheaper than 4500", "in the 7 to 9k range",
    ]
    patterns = [
        "show me {b} {p} {price}",
        "need a {b} {p} {price}",
        "looking for {p} from {b} {price}",
        "best {p} for commuting {price}",
        "suggest a {p} that stays quiet and fits {price}",
        "anything like a {b} {p} but in another color",
        "compare two {p} options without opening ten tabs",
        "recommend {p} for college that survives rain {price}",
        "want {p} with long battery {price}",
        "find me {b} gear similar to what influencers push {price}",
        "is there a deal on {p} right now {price}",
        "help me pick {p} for a small kitchen {price}",
        "search {b} {p} with free returns if possible",
        "got any {p} recommendations {price} that ship this week",
        "looking for a quieter {p} than my old one {price}",
        "{b} {p} with waterproof rating {price}",
        "need gift-worthy {p} {price} and not flashy packaging",
        "show alternatives to {b} {p} if sold out",
        "which {p} should I order first if I only want one",
        "find {p} listings that mention noise reduction {price}",
    ]
    seen: set[str] = set()
    out: list[str] = []
    guard = 0
    while len(out) < n and guard < 20000:
        guard += 1
        text = rng.choice(patterns).format(
            b=rng.choice(brands), p=rng.choice(products), price=rng.choice(prices)
        )
        if rng.random() < 0.2:
            text = text.replace("looking for", "lookin for").replace("recommend", "reccomend")
        uniq_extend(out, seen, [text], blocked)
    if len(out) < n:
        raise RuntimeError(f"OOD ecommerce shortfall {len(out)}/{n}")
    return out[:n]


def ood_retail(n: int, blocked: set[str]) -> list[str]:
    patterns = [
        "inventory level of outlet {n}",
        "where is aisle {a}",
        "move {q} units to the front shelf",
        "check whether the backroom has this size",
        "which branch has stock nearby",
        "fitting room needs more hangers",
        "transfer this sku to outlet {n}",
        "restock endcap before noon",
        "receiving dock is blocked by pallets",
        "shelf {a} label does not match the barcode",
        "do we have extra mediums in the cage",
        "pull five cartons from backstock",
        "showroom floor model is damaged near the armrest",
        "can someone cover lane {a} for ten minutes",
        "backroom inventory count is off by {q}",
        "put returns on the sorting table",
        "aisle {a} needs a price gun",
        "is outlet {n} still open for employee pickup",
        "transfer leftover seasonal to the clearance wall",
        "check handheld stock for this sku before I walk the floor",
        "front shelf looks empty but system says {q} on hand",
        "need a manager override at the service desk",
        "hangers missing in fitting room {a}",
        "move overstock from receiving to aisle {a}",
        "which cage holds the rain checks",
    ]
    seen: set[str] = set()
    out: list[str] = []
    guard = 0
    while len(out) < n and guard < 20000:
        guard += 1
        text = rng.choice(patterns).format(
            n=rng.randint(2, 40),
            a=rng.randint(1, 24),
            q=rng.randint(2, 18),
        )
        uniq_extend(out, seen, [text], blocked)
    if len(out) < n:
        raise RuntimeError(f"OOD retail shortfall {len(out)}/{n}")
    return out[:n]


def ood_legal(n: int, blocked: set[str]) -> list[str]:
    seeds = [
        "what is an arbitration clause",
        "define negligence",
        "can a verbal contract count",
        "is this clause enforceable",
        "what does indemnity mean",
        "can my landlord do this",
        "they changed the terms after I signed",
        "when does this notice expire",
        "what is a noncompete",
        "define consideration in a contract",
        "is a text message enough to amend an agreement",
        "what is constructive dismissal",
        "can silence count as acceptance",
        "what does without prejudice mean",
        "is a handwritten change valid",
        "define liquidated damages",
        "what is a demand letter",
        "can they enforce a renewal I never saw",
        "what is a cooling-off period",
        "define fiduciary duty simply",
        "is an unsigned pdf binding",
        "what is a holdover tenancy",
        "can an email create a contract",
        "what does severability mean",
        "define promissory estoppel in plain words",
        "is a clickwrap agreement real",
        "what is a statute of limitations",
        "can my employer change commission after the fact",
        "what is a warranty disclaimer",
        "define force majeure without jargon",
        "is arbitration optional here",
        "what is a three-day notice",
        "can they keep the deposit without an itemized list",
        "what does assignment of rights mean",
        "is a verbal lease enforceable for a year",
        "what is mutual assent",
        "define breach of warranty",
        "can a minor sign this",
        "what is an integration clause",
        "when must I answer a complaint",
        "what is a cease and desist",
        "define unjust enrichment",
        "can they record me if I object",
        "what is a power of attorney",
        "is a settlement offer confidential",
        "what does indemnity shift onto me",
        "can a landlord enter without notice",
        "what is a class action waiver",
        "define material breach",
        "what happens if I ignore a subpoena",
        "is a non-solicitation different from a noncompete",
        "what is quiet enjoyment",
        "define anticipatory repudiation",
        "can they change fees mid-term",
        "what is a notice period",
        "define duty of care",
        "is a screenshot of terms enough proof",
        "what is a liquidated damages clause",
        "can silence after a deadline waive my claim",
        "what is a tenancy at will",
        "define specific performance",
        "what is a merger clause",
        "can they assign the contract to another company",
        "what does hold harmless mean",
        "is a verbal side deal enforceable",
        "what is a condition precedent",
        "define equitable estoppel briefly",
        "can they accelerate the whole balance",
        "what is a cure period",
        "define constructive notice",
        "what is a warranty of habitability",
        "can an auto-renewal stand without clear notice",
        "what is a release of claims",
        "define joint and several liability",
        "what is a non-disclosure agreement",
        "can they ban pets after I moved in with one",
        "what is an as-is clause",
        "define mitigation of damages",
        "what is a forum selection clause",
        "can they fire me for reporting safety issues",
        "what is a severance agreement",
        "define at-will employment simply",
        "what is a mechanic's lien",
        "can an oral modification override a written ban",
        "what is a right of first refusal",
        "define quiet title",
        "what is a demand for assurance",
        "can they withhold wages for a uniform",
        "what is a non-disparagement clause",
        "define unjust enrichment with an example",
        "what is a bill of sale",
        "can they enforce a one-sided arbitration fee",
        "what is a notice to quit",
        "define warranty of merchantability",
        "what is a personal guarantee",
        "can they change the venue after filing",
        "what is a protective order",
        "define proximate cause simply",
        "what is a contingent fee",
        "can they keep funds in escrow past closing",
        "what is a novation",
        "define anticipatory breach",
        "what is a security deposit itemization",
        "can an unsigned addendum control",
        "what is a tolling agreement",
        "define constructive eviction",
        "what is a choice-of-law clause",
        "can they retract an offer after I accepted",
    ]
    # Ensure uniqueness and fill to n with variants that stay short/legal.
    extras = []
    bases = [
        "briefly explain {x}",
        "plain meaning of {x}",
        "is {x} something I should worry about",
        "when people say {x} what do they mean",
        "does {x} usually favor one side",
    ]
    topics = [
        "indemnity", "novation", "estoppel", "subrogation", "rescission",
        "injunction", "garnishment", "lien priority", "quiet enjoyment",
        "material adverse change", "cure notice", "waiver", "ratification",
    ]
    for topic in topics:
        for base in bases:
            extras.append(base.format(x=topic))

    seen: set[str] = set()
    out: list[str] = []
    uniq_extend(out, seen, seeds + extras, blocked)
    guard = 0
    while len(out) < n and guard < 10000:
        guard += 1
        text = f"what is {rng.choice(topics)} in simple terms"
        if rng.random() < 0.5:
            text = f"define {rng.choice(topics)}"
        uniq_extend(out, seen, [text], blocked)
    if len(out) < n:
        raise RuntimeError(f"OOD legal shortfall {len(out)}/{n}")
    return out[:n]


def ood_generic(n: int, blocked: set[str]) -> list[str]:
    seeds = [
        "how are running shoes manufactured",
        "why do stores arrange products by aisle",
        "what is the history of contract bridge",
        "how does a barcode work",
        "why is the sky blue",
        "what is the capital of france",
        "explain how online payments work technically",
        "how does a credit card authorization hold work",
        "why do warehouses use barcodes instead of only names",
        "what is the difference between retail and wholesale pricing in economics",
        "how are sneakers vulcanized in a factory",
        "why do shopping carts have different wheel designs",
        "what is the origin of the word mortgage",
        "how does arbitration differ from mediation in history class terms",
        "explain supply chain at a high level for a school project",
        "how does a thermal receipt printer work",
        "why do shoes have different last shapes",
        "what is the capital of kazakhstan",
        "how do loyalty points systems calculate balances behind the scenes",
        "why does ice float",
        "what causes ocean tides",
        "how does wifi hand off between access points",
        "explain compound interest with a tiny example",
        "how are cardboard packages folded on an assembly line",
        "why do some countries use different voltage",
        "what is a monsoon",
        "how do noise cancelling headphones cancel sound",
        "why do leaves change color",
        "what is entropy in everyday language",
        "how does gps trilateration work",
        "why do metal rails leave gaps for expansion",
        "what is the greenhouse effect",
        "how do vaccines train immune memory",
        "why is the dead sea so salty",
        "what is a prime number",
        "how does a qr code store data",
        "why do fans cool people if they move warm air",
        "what is the difference between weather and climate",
        "how does a touchscreen sense a finger",
        "why do airplanes leave contrails",
        "what is the fibonacci sequence used for outside math class",
        "how does a microwave heat food",
        "why do we see phases of the moon",
        "what is dark matter according to physicists",
        "how does two factor authentication stop password reuse attacks",
        "why do bridges use expansion joints",
        "what is photosynthesis for a middle schooler",
        "how does a heat pump move warmth",
        "why do currencies have exchange rates",
        "what is the silk road historically",
        "how are online ads targeted at a technical level",
        "why do stores use endcaps according to merchandising research",
        "what is a barcode checksum",
        "how does a sewing machine form a lockstitch",
        "why do running shoes have foam midsoles",
        "what is the history of the shipping container",
        "how does a POS terminal talk to a payment processor",
        "why do legal systems distinguish civil and criminal cases academically",
        "what is the capital of canada",
        "how do rifled barrels improve accuracy",
        "why do some paints need primer",
        "what is a glacier",
        "how does bluetooth pair devices",
        "why do soap bubbles show colors",
        "what is the doppler effect",
        "how does a refrigerator compressor work",
        "why do we need leap years",
        "what is a polynomial",
        "how are glass bottles recycled industrially",
        "why do magnets attract iron",
        "what is the difference between RAM and storage",
        "how does a search engine rank pages at a high level",
        "why do coastal cities stay milder in winter",
        "what is a black hole event horizon",
        "how does distillation separate liquids",
        "why do batteries eventually lose capacity",
        "what is the speed of light in vacuum",
        "how do bees find flowers",
        "why do we dream according to common theories",
        "what is a continent versus a country",
        "how does a fuse protect a circuit",
        "why do paper cuts hurt so much",
        "what is the difference between a virus and bacteria",
        "how does a wind turbine generate electricity",
        "why do rainbows form arcs",
        "what is inflation in economics class terms",
        "how does a camera lens focus light",
        "why do hot air balloons rise",
        "what is the capital of japan",
        "how does composting break down food scraps",
        "why do ships float despite steel being dense",
        "what is a histogram used for",
        "how does a thermostat decide to call for heat",
        "why do we have time zones",
        "what is the difference between a novel and a novella",
        "how does a catalytic converter reduce emissions",
        "why do espresso machines use pressure",
        "what is a supply curve",
        "how do libraries assign call numbers",
        "why do some animals migrate seasonally",
        "what is the capital of kenya",
        "how does a smoke detector sense particles",
        "why do we use base ten counting historically",
        "what is a feedback loop in systems thinking",
        "how does a bicycle gearing ratio change effort",
        "why do concrete sidewalks crack",
        "what is the difference between mass and weight",
    ]
    seen: set[str] = set()
    out: list[str] = []
    uniq_extend(out, seen, seeds, blocked)
    fillers = [
        "can you explain {t} briefly",
        "I am curious about {t}",
        "give a short answer on {t}",
        "what should a beginner know about {t}",
    ]
    topics = [
        "osmosis", "inertia", "capacitance", "erosion", "mitosis",
        "albedo", "refraction", "momentum", "viscosity", "coriolis effect",
    ]
    guard = 0
    while len(out) < n and guard < 10000:
        guard += 1
        text = rng.choice(fillers).format(t=rng.choice(topics))
        uniq_extend(out, seen, [text], blocked)
    if len(out) < n:
        raise RuntimeError(f"OOD generic shortfall {len(out)}/{n}")
    return out[:n]


# ---------------------------------------------------------------------------
# V3 augmentation (training only; must not overlap OOD)
# ---------------------------------------------------------------------------

def aug_generic(n: int, blocked: set[str]) -> list[str]:
    products = [
        "shoes", "sneakers", "phones", "backpacks", "jackets", "mixers",
        "laptops", "headphones", "tablets", "cameras", "watches", "blenders",
    ]
    places = [
        "france", "kenya", "japan", "canada", "peru", "norway", "egypt",
        "chile", "ghana", "vietnam", "portugal", "finland",
    ]
    topics = [
        "photosynthesis", "entropy", "inflation", "momentum", "refraction",
        "contract bridge", "the silk road", "leap years", "prime numbers",
        "osmosis", "inertia", "erosion", "mitosis", "albedo", "viscosity",
        "capacitance", "coriolis effect", "doppler effect", "greenhouse gases",
    ]
    openers = [
        "how are {p} manufactured in a factory",
        "why do stores group {p} by aisle according to merchandising research",
        "what is the history of {topic}",
        "how does a barcode identify {p}",
        "explain online payments technically without teaching me to buy {p}",
        "what is the capital of {place}",
        "why do people use contracts in business history",
        "how does arbitration differ from mediation in a civics lesson",
        "what does indemnity mean in dictionary terms only",
        "how are shopping carts designed for stability",
        "why do warehouses scan {p} instead of writing names",
        "what is {topic} for a beginner",
        "how does wifi differ from cellular data when discussing {topic}",
        "why is the sky blue at noon",
        "explain {topic} with one small example",
        "how does a receipt printer work in a store setting",
        "what is the difference between retail and wholesale in economics",
        "why do running shoes use foam midsoles scientifically",
        "how do loyalty ledgers update balances in software systems",
        "what causes tides near {place}",
        "can you summarize {topic} without jargon",
        "I am curious how {p} packaging is recycled",
        "give a short school-level answer about {topic}",
        "how would a textbook introduce {topic}",
        "what should someone know first about {topic}",
        "why do engineers care about {topic}",
        "how is {topic} measured in practice",
        "where does {topic} show up outside classrooms",
        "compare popular misconceptions about {topic}",
        "explain the basic idea behind {topic}",
    ]
    seen: set[str] = set()
    out: list[str] = []
    guard = 0
    while len(out) < n and guard < 100000:
        guard += 1
        text = rng.choice(openers).format(
            p=rng.choice(products),
            place=rng.choice(places),
            topic=rng.choice(topics),
        )
        if rng.random() < 0.2:
            text = text + " Keep it concise."
        uniq_extend(out, seen, [text], blocked)
    if len(out) < n:
        raise RuntimeError(f"aug generic shortfall {len(out)}/{n}")
    return out[:n]


def aug_ecommerce(n: int, blocked: set[str]) -> list[str]:
    brands = [
        "nike", "adidas", "samsung", "apple", "lenovo", "hp", "dell",
        "sony", "boat", "noise", "realme", "vivo", "canon", "nikon",
        "puma", "reebok", "xiaomi", "oneplus", "asus", "acer",
    ]
    products = [
        "shoes", "sneakers", "phone", "earbuds", "headphones", "laptop",
        "tablet", "watch", "mixer", "blender", "backpack", "jacket",
        "camera", "monitor", "keyboard", "mouse", "router", "speaker",
    ]
    verbs = [
        "looking for", "need", "show me", "find", "suggest", "recommend",
        "help me pick", "search for", "want", "got any",
    ]
    constraints = [
        "under {price}",
        "below {price}",
        "around {price}",
        "with free returns",
        "that ships this week",
        "in blue",
        "with long battery",
        "for college",
        "for a small kitchen",
        "with overnight delivery",
        "priced in rupees under {price}",
        "priced in dollars under {price}",
        "refurbished with warranty under {price}",
        "that pairs with android under {price}",
        "similar to last year under {price}",
    ]
    prices = [
        999, 1499, 1999, 2499, 2999, 3499, 3999, 4499, 4999, 5499,
        5999, 6999, 7999, 8999, 9999, 11999, 14999, 19999, 24999, 29999,
    ]
    seen: set[str] = set()
    out: list[str] = []
    guard = 0
    while len(out) < n and guard < 100000:
        guard += 1
        text = (
            f"{rng.choice(verbs)} {rng.choice(brands)} {rng.choice(products)} "
            f"{rng.choice(constraints).format(price=rng.choice(prices))}"
        )
        uniq_extend(out, seen, [text], blocked)
    if len(out) < n:
        raise RuntimeError(f"aug ecommerce shortfall {len(out)}/{n}")
    return out[:n]


def aug_retail(n: int, blocked: set[str]) -> list[str]:
    actions = [
        "check stock for store {n}",
        "aisle {a} needs restock on {item}",
        "backroom count for {item}",
        "transfer {item} to branch {n}",
        "fitting room {a} is out of hangers",
        "front shelf count for {item} looks wrong",
        "move {q} cases of {item} to the endcap",
        "is sku {sku} on the floor or only in the cage",
        "showroom sample of {item} has a scratch",
        "receiving says {item} arrived but handheld disagrees",
        "pull {item} from overstock for aisle {a}",
        "outlet {n} asked for a stock transfer of {item}",
        "price label on shelf {a} does not match {item}",
        "customer waiting for {item} size check in backroom",
        "restock {item} before the evening rush",
        "lane {a} needs change for cash returns",
        "clearance wall needs more {item}",
        "inventory discrepancy on {item} after cycle count",
        "put returns of {item} on the sorting table",
        "garden center needs {item} before weekend",
        "service desk override for {item}",
        "cage inventory for {item} is short by {q}",
        "move {item} from receiving to aisle {a}",
        "branch {n} is low on {item}",
        "floor walk: missing {item} near aisle {a}",
    ]
    items = [
        "shoes", "sneakers", "detergent", "batteries", "paint", "milk",
        "chargers", "socks", "mugs", "towels", "cereal", "light bulbs",
        "diapers", "coffee", "tea", "rice", "oil", "soap",
    ]
    seen: set[str] = set()
    out: list[str] = []
    guard = 0
    while len(out) < n and guard < 100000:
        guard += 1
        text = rng.choice(actions).format(
            n=rng.randint(3, 60),
            a=rng.randint(1, 30),
            q=rng.randint(2, 20),
            sku=rng.randint(10000, 99999),
            item=rng.choice(items),
        )
        uniq_extend(out, seen, [text], blocked)
    if len(out) < n:
        raise RuntimeError(f"aug retail shortfall {len(out)}/{n}")
    return out[:n]


def aug_legal(n: int, blocked: set[str]) -> list[str]:
    frames = [
        "what is an {x}",
        "define {x} in plain language",
        "is a {x} enforceable after I already signed",
        "can a {x} be verbal",
        "what does {x} mean for my rights",
        "they changed the {x} after I agreed",
        "when does a {x} expire",
        "do I have to follow this {x}",
        "short answer: what is {x}",
        "explain {x} like I am preparing for a consult",
        "is {x} the same as a waiver",
        "can my landlord rely on this {x}",
        "what notice rules apply to a {x}",
        "does an email count toward a {x}",
        "help me understand this {x} without jargon",
        "is {x} optional or mandatory here",
        "what happens if I ignore a {x}",
        "can silence accept a {x}",
        "plain-English reading of a {x}",
        "does {x} survive if I change jobs",
        "quick question about a {x}",
        "is this {x} binding",
        "need a simple definition of {x}",
        "someone mentioned a {x} today",
        "how should I read a {x}",
    ]
    topics = [
        "arbitration clause", "indemnity clause", "noncompete", "lease renewal",
        "eviction notice", "demand letter", "power of attorney", "warranty disclaimer",
        "settlement offer", "clickwrap agreement", "cooling-off period",
        "holdover tenancy", "force majeure clause", "severability clause",
        "integration clause", "class action waiver", "notice to quit",
        "security deposit itemization", "non-disclosure agreement",
        "choice-of-law clause", "forum selection clause", "non-solicitation",
        "personal guarantee", "mechanics lien", "cure notice",
    ]
    seen: set[str] = set()
    out: list[str] = []
    guard = 0
    while len(out) < n and guard < 100000:
        guard += 1
        text = rng.choice(frames).format(x=rng.choice(topics))
        if rng.random() < 0.25:
            text = text + "?"
        uniq_extend(out, seen, [text], blocked)
    if len(out) < n:
        raise RuntimeError(f"aug legal shortfall {len(out)}/{n}")
    return out[:n]


def token_len(text: str) -> int:
    return len(re.findall(r"[a-z0-9']+", text.lower()))


QUESTION_PREFIXES = (
    "what ",
    "why ",
    "how ",
    "when ",
    "where ",
    "which ",
    "who ",
    "can ",
    "is ",
    "are ",
    "do ",
    "does ",
    "did ",
    "should ",
    "could ",
    "would ",
)


def is_question(text: str) -> bool:
    stripped = text.strip().lower()
    if "?" in stripped:
        return True
    return any(stripped.startswith(prefix) for prefix in QUESTION_PREFIXES)


def report_dataset(
    rows: list[dict],
    blocked_norms: set[str],
    name: str,
    extra: dict | None = None,
) -> dict:
    counts = Counter(r["intent"] for r in rows)
    norms = [norm(r["text"]) for r in rows]
    dup = len(norms) - len(set(norms))
    overlap = sum(1 for n in norms if n in blocked_norms)
    avg = {}
    qratio = {}
    statement_ratio = {}
    for intent in ("GENERIC", "ECOMMERCE", "RETAIL", "LEGAL"):
        subset = [r["text"] for r in rows if r["intent"] == intent]
        lengths = [token_len(t) for t in subset]
        avg[intent] = round(sum(lengths) / max(len(lengths), 1), 2)
        n_q = sum(is_question(t) for t in subset)
        qratio[intent] = round(n_q / max(len(subset), 1), 3)
        statement_ratio[intent] = round(1.0 - qratio[intent], 3)
    means = list(avg.values())
    report = {
        "name": name,
        "total_samples": len(rows),
        "per_class_counts": dict(counts),
        "duplicate_count": dup,
        "overlap_with_blocked": overlap,
        "average_length_by_class": avg,
        "question_ratio_by_class": qratio,
        "statement_ratio_by_class": statement_ratio,
        "length_spread": round(max(means) - min(means), 2) if means else 0,
        "validation": {
            "no_duplicates": dup == 0,
            "balanced": len(set(counts.values())) == 1,
            "no_blocked_overlap": overlap == 0,
        },
    }
    if extra:
        report.update(extra)
    return report


def build_ood(v2_norms: set[str]) -> list[dict]:
    generic = ood_generic(100, v2_norms)
    ecommerce = ood_ecommerce(100, v2_norms)
    retail = ood_retail(100, v2_norms)
    legal = ood_legal(100, v2_norms)
    rows = (
        [{"text": t, "intent": "GENERIC"} for t in generic]
        + [{"text": t, "intent": "ECOMMERCE"} for t in ecommerce]
        + [{"text": t, "intent": "RETAIL"} for t in retail]
        + [{"text": t, "intent": "LEGAL"} for t in legal]
    )
    rng.shuffle(rows)
    return rows


def build_v3(v2_rows: list[dict], ood_norms: set[str]) -> list[dict]:
    blocked = {norm(r["text"]) for r in v2_rows} | ood_norms
    # Augment only blocked by OOD + existing texts while generating,
    # but final set is v2 + new (v2 already unique).
    gen_block = ood_norms | {norm(r["text"]) for r in v2_rows}
    added = (
        [{"text": t, "intent": "GENERIC"} for t in aug_generic(200, gen_block)]
        + [{"text": t, "intent": "ECOMMERCE"} for t in aug_ecommerce(200, gen_block)]
        + [{"text": t, "intent": "RETAIL"} for t in aug_retail(200, gen_block)]
        + [{"text": t, "intent": "LEGAL"} for t in aug_legal(200, gen_block)]
    )
    # Ensure added don't collide with each other
    seen = {norm(r["text"]) for r in v2_rows} | ood_norms
    clean_added = []
    for row in added:
        key = norm(row["text"])
        if key in seen:
            continue
        seen.add(key)
        clean_added.append(row)
    if len(clean_added) != 800:
        # top up if collisions within added
        deficit = 800 - len(clean_added)
        raise RuntimeError(f"augmentation produced {len(clean_added)} unique rows, need 800")
    rows = list(v2_rows) + clean_added
    rng.shuffle(rows)
    return rows


def ambiguous_candidates(rows: list[dict]) -> list[dict]:
    out = []
    for row in rows:
        text = row["text"].lower()
        flags = []
        if row["intent"] == "GENERIC" and any(
            w in text for w in ("contract", "arbitration", "shoes", "store", "payment")
        ):
            flags.append("generic_with_domain_nouns")
        if row["intent"] == "LEGAL" and text.startswith("what is"):
            flags.append("short_legal_definition")
        if row["intent"] == "ECOMMERCE" and any(ch.isdigit() for ch in text):
            flags.append("brand_or_price_style")
        if row["intent"] == "RETAIL" and any(w in text for w in ("aisle", "backroom", "sku", "outlet")):
            flags.append("ops_style")
        if flags:
            out.append({"text": row["text"], "intent": row["intent"], "flags": flags})
    rng.shuffle(out)
    return out[:30]


def main() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    v2_rows = load_jsonl(V2_PATH)
    v2_norms = {norm(r["text"]) for r in v2_rows}

    ood_rows = build_ood(v2_norms)
    write_jsonl(OOD_PATH, ood_rows)
    ood_report = report_dataset(
        ood_rows,
        blocked_norms=v2_norms,
        name="intent_ood_test",
        extra={
            "purpose": "evaluation_only",
            "never_use_for": [
                "training",
                "validation",
                "vocabulary",
                "early_stopping",
                "checkpoint_selection",
            ],
            "examples_by_category": {
                intent: [r["text"] for r in ood_rows if r["intent"] == intent][:8]
                for intent in ("GENERIC", "ECOMMERCE", "RETAIL", "LEGAL")
            },
            "ambiguous_candidates": ambiguous_candidates(ood_rows),
        },
    )
    # Optional UNK vs current vocab if present
    vocab_path = ARTIFACTS / "vocab.json"
    if vocab_path.exists():
        import sys

        sys.path.insert(0, str(ROOT / "src"))
        from tokenizer import encode, load_vocab

        vocab = load_vocab(str(vocab_path))
        unk_id = vocab["<UNK>"]
        per_class_unk = {}
        total = unk = 0
        for intent in ("GENERIC", "ECOMMERCE", "RETAIL", "LEGAL"):
            texts = [r["text"] for r in ood_rows if r["intent"] == intent]
            t_tok = u_tok = 0
            for text in texts:
                ids = encode(text, vocab)
                t_tok += len(ids)
                u_tok += sum(i == unk_id for i in ids)
            per_class_unk[intent] = round(100 * u_tok / max(t_tok, 1), 2)
            total += t_tok
            unk += u_tok
        ood_report["unk_rate_overall_with_current_vocab"] = round(100 * unk / max(total, 1), 2)
        ood_report["unk_rate_by_class_with_current_vocab"] = per_class_unk

    OOD_REPORT.write_text(json.dumps(ood_report, indent=2) + "\n", encoding="utf-8")
    assert ood_report["validation"]["no_duplicates"]
    assert ood_report["validation"]["balanced"]
    assert ood_report["validation"]["no_blocked_overlap"]
    assert ood_report["per_class_counts"] == {
        "GENERIC": 100,
        "ECOMMERCE": 100,
        "RETAIL": 100,
        "LEGAL": 100,
    }

    ood_norms = {norm(r["text"]) for r in ood_rows}
    v3_rows = build_v3(v2_rows, ood_norms)
    write_jsonl(V3_PATH, v3_rows)
    v3_report = report_dataset(
        v3_rows,
        blocked_norms=ood_norms,
        name="intents_v3",
        extra={
            "source": "intents_v2 + 800 targeted augmentations",
            "augmentation_counts": {"GENERIC": 200, "ECOMMERCE": 200, "RETAIL": 200, "LEGAL": 200},
            "ambiguous_candidates": ambiguous_candidates(v3_rows),
            "confusion_candidates": ambiguous_candidates(v3_rows)[:25],
        },
    )
    # Vocabulary coverage of the 800 added rows vs a train-only-style vocab from v2
    added_rows = [r for r in v3_rows if norm(r["text"]) not in v2_norms]
    try:
        import sys

        sys.path.insert(0, str(ROOT / "src"))
        from tokenizer import build_vocab, encode

        v2_vocab = build_vocab([r["text"] for r in v2_rows])
        unk_id = v2_vocab["<UNK>"]
        covered = total = 0
        for row in added_rows:
            ids = encode(row["text"], v2_vocab)
            total += len(ids)
            covered += sum(1 for i in ids if i != unk_id)
        v3_report["added_example_count"] = len(added_rows)
        v3_report["vocab_coverage_of_added_vs_v2_vocab"] = {
            "token_coverage_pct": round(100 * covered / max(total, 1), 2),
            "total_tokens": total,
            "known_tokens": covered,
            "unk_tokens": total - covered,
        }
    except Exception as exc:  # pragma: no cover
        v3_report["vocab_coverage_error"] = str(exc)

    V3_REPORT.write_text(json.dumps(v3_report, indent=2) + "\n", encoding="utf-8")
    assert len(v3_rows) == 2800
    assert v3_report["per_class_counts"] == {
        "GENERIC": 700,
        "ECOMMERCE": 700,
        "RETAIL": 700,
        "LEGAL": 700,
    }
    assert v3_report["validation"]["no_duplicates"]
    assert v3_report["validation"]["no_blocked_overlap"]

    print("OOD:", OOD_PATH, ood_report["per_class_counts"])
    print("V3:", V3_PATH, v3_report["per_class_counts"], "total", len(v3_rows))
    print("Reports:", OOD_REPORT, V3_REPORT)


if __name__ == "__main__":
    main()
