"""Generate intents_v2.jsonl (2000 unique examples) and dataset_report.json."""

from __future__ import annotations

import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT_JSONL = DATA / "intents_v2.jsonl"
OUT_REPORT = ROOT / "dataset_report.json"

rng = random.Random(20260322)


def uniq_add(bucket: list[str], seen: set[str], text: str) -> bool:
    key = re.sub(r"\s+", " ", text.strip().lower())
    if not key or key in seen:
        return False
    seen.add(key)
    bucket.append(text.strip())
    return True


def pick(seq):
    return rng.choice(seq)


# ---------------------------------------------------------------------------
# GENERIC
# ---------------------------------------------------------------------------

GENERIC_SCIENCE = [
    "What is gravity?",
    "How do vaccines train the immune system without causing the full disease?",
    "Why is the sky blue during the day but red near sunset?",
    "Explain photosynthesis in plain language for a middle schooler.",
    "I am trying to understand why satellites stay in orbit even though gravity is constantly pulling them toward the Earth.",
    "Does boiling water kill every kind of microorganism or only some of them?",
    "What actually happens inside a star when it begins to run out of hydrogen?",
    "How can ice float on water when most solids sink in their liquid form?",
    "Is light a wave, a particle, or somehow both depending on the experiment?",
    "Why do humans need sleep if the brain never fully shuts down?",
    "Walk me through what DNA replication means without jargon.",
    "How does a greenhouse gas trap heat differently from ordinary air?",
    "What is the difference between weather and climate in one careful paragraph?",
    "Why does metal feel colder than wood even when both are at room temperature?",
    "Can you explain entropy without turning it into a textbook page?",
    "How do antibiotics differ from antiviral medicines?",
    "What causes earthquakes along tectonic plate boundaries?",
    "Why do we see phases of the Moon instead of a permanently full disk?",
    "How does GPS figure out a phone's location from satellites?",
    "What is dark matter, and why do scientists think it exists?",
]

GENERIC_GEO = [
    "Capital of Kazakhstan?",
    "Which river runs through Cairo?",
    "Where is Patagonia relative to Chile and Argentina?",
    "What mountain range separates Europe from Asia in western Russia?",
    "Name three landlocked countries in Africa.",
    "Why is the Dead Sea so much saltier than the open ocean?",
    "Which desert is larger, the Sahara or Antarctica's ice desert?",
    "I need a quick overview of Southeast Asia's major island groups.",
    "How far is it roughly from Lisbon to Madrid by road?",
    "What is the capital of Canada, and why do people often guess Toronto?",
    "Which ocean sits between Africa and Australia?",
    "Explain why Iceland has so many volcanoes and geysers.",
    "Where would I find the Great Barrier Reef on a map?",
    "What country has both Mediterranean and Atlantic coastlines and shares a border with Spain?",
    "Is Greenland politically part of Europe or North America?",
]

GENERIC_HISTORY = [
    "Who painted the Sistine Chapel ceiling?",
    "When did the Berlin Wall fall?",
    "What triggered the start of World War I beyond the assassination alone?",
    "Who was the first person to reach the South Pole?",
    "Summarize the Renaissance without reducing it to a list of famous names.",
    "Why did the industrial revolution begin in Britain rather than elsewhere?",
    "What was the Silk Road actually used for day to day?",
    "How did the printing press change access to books in Europe?",
    "Who led India's independence movement most associated with nonviolent protest?",
    "What caused the Dust Bowl in the American Midwest?",
    "Explain the difference between the American Revolution and the Civil War.",
    "When did women gain the right to vote in New Zealand?",
    "What was the purpose of the Magna Carta in its own time?",
    "Who wrote the Declaration of Independence's first draft?",
    "How did the Black Death reshape European labor and wages?",
]

GENERIC_TECH = [
    "How does Wi-Fi differ from cellular data under the hood?",
    "What does an API do in everyday software terms?",
    "Explain cloud storage like I'm lending files to a remote locker.",
    "Why do phones need regular software updates?",
    "What is the difference between RAM and storage on a laptop?",
    "How does two-factor authentication actually stop most password theft?",
    "What is open-source software, in one honest paragraph?",
    "Why does a computer fan spin harder during video calls?",
    "How do noise-cancelling headphones cancel sound without blocking ears completely?",
    "What is a VPN useful for on public Wi-Fi?",
    "Explain machine learning without promising magic.",
    "How does a QR code store information?",
    "What happens when you empty the recycle bin on Windows?",
    "Why do some websites still refuse to load on older browsers?",
    "What is the difference between HTTP and HTTPS for a regular user?",
]

GENERIC_MATH = [
    "72 fahrenheit in celsius",
    "How many tablespoons are in a quarter cup?",
    "Explain compound interest as if I were fifteen.",
    "What is a prime number, with two examples?",
    "How do you find the area of a circle from its radius?",
    "Walk through solving 3x + 5 = 20 step by step.",
    "What does standard deviation tell you about a set of scores?",
    "How is a median different from a mean when outliers appear?",
    "Convert 2.5 kilometers into meters.",
    "Why does dividing by zero break ordinary arithmetic?",
    "Explain fractions versus decimals for the same quantity.",
    "What is the Pythagorean theorem used for in real life?",
    "How do percentages relate to proportions?",
    "What is the next number in a simple Fibonacci walk-through?",
    "How do you calculate a tip of 18 percent on a 64 dollar bill?",
]

GENERIC_GK = [
    "Who took the World Series in 2016?",
    "How would you say see you tomorrow in casual Spanish?",
    "Latte versus flat white for someone who just says coffee.",
    "Why do leaves change color before they fall?",
    "Polite way to decline a dinner invite without inventing drama.",
    "What should I pack for Iceland in March if I run cold?",
    "A haiku about Monday mornings that is dry, not inspirational.",
    "Define serendipity and put it in a sentence that is not corny.",
    "How many players are on the field for one soccer side during open play?",
    "What does it mean when a book is in the public domain?",
    "Rough population of Canada.",
    "Why is the ocean salty if rivers taste fresh?",
    "Morning or evening for watering tomatoes in a heat wave?",
    "One clean joke about accountants.",
    "How long should leftover cooked chicken stay in the fridge?",
]


def more_generic(n: int, seen: set[str]) -> list[str]:
    out: list[str] = []
    topics = [
        ("science", ["orbit", "cells", "atoms", "magnetism", "evaporation", "enzymes", "fossils", "neurons", "pressure", "friction"]),
        ("geo", ["delta", "archipelago", "tundra", "monsoon", "fjord", "plateau", "strait", "peninsula", "savanna", "glacier"]),
        ("history", ["treaty", "empire", "revolution", "expedition", "manuscript", "suffrage", "colony", "armistice", "dynasty", "migration"]),
        ("tech", ["browser", "server", "pixel", "bandwidth", "firmware", "cache", "encryption", "bluetooth", "processor", "firewall"]),
        ("math", ["ratio", "equation", "angle", "volume", "probability", "graph", "average", "square root", "percentage", "coordinate"]),
        ("life", ["sleep", "stretching", "hydration", "decluttering", "focus", "commute", "cooking", "reading", "walking", "budgeting"]),
    ]
    openers = [
        "What is {x}?",
        "Can you explain {x} simply?",
        "I keep forgetting how {x} works.",
        "Why does {x} matter in everyday life?",
        "Give me a short answer about {x}.",
        "How would you describe {x} to a curious beginner?",
        "Is there a clear difference between {x} and related ideas people confuse?",
        "People mention {x} a lot. What does it actually mean?",
        "Help me understand {x} without a lecture.",
        "I read something confusing about {x}. Can you straighten it out?",
        "{x} still does not click for me.",
        "In one paragraph, what should someone know about {x}?",
        "Where does the idea of {x} show up outside school?",
        "I'm curious about {x} after a conversation last night.",
        "Does {x} have a practical example I would recognize?",
    ]
    long_bits = [
        "I tried reading about it and the explanations kept looping.",
        "Keep the answer grounded and skip the motivational tone.",
        "Assume I know the basics but not the jargon.",
        "A friend asked me and I blanked.",
        "I want something accurate enough to repeat later.",
    ]
    while len(out) < n:
        _, words = pick(topics)
        word = pick(words)
        roll = rng.random()
        if roll < 0.25:
            text = pick(openers).format(x=word)
        elif roll < 0.65:
            text = f"{pick(openers).format(x=word)} {pick(long_bits)}"
        else:
            text = (
                f"Lately I have been wondering about {word}. "
                f"{pick(long_bits)} What is the cleanest way to explain it?"
            )
        if rng.random() < 0.15:
            text = text[0].lower() + text[1:] if text else text
        uniq_add(out, seen, text)
    return out


ECOMMERCE_SEED = [
    "Need waterproof hiking footwear under 7000",
    "Looking for wireless earbuds with long battery life",
    "Can you suggest a gaming laptop for college that will not overheat on Zoom?",
    "The navy hoodie I added to the cart last night is missing now that I'm back on the site.",
    "Can the total be split between a gift card and Visa? The page only shows one payment box.",
    "Marked delivered at 2:14. The stoop is empty and the photo is a house two doors down.",
    "Did the charcoal blender jump back to full price after midnight, or is the slash still real?",
    "This desk lamp has to arrive before Friday's video call. Which shipping option actually makes that?",
    "The welcome code errors out the second I try to pay, even though the banner accepted it.",
    "Renewal hits on the 3rd. I want it stopped, and the cancel button just spins.",
    "Do these ceramic mugs ship internationally, or does the page hide that until checkout?",
    "No confirmation email, and the bank shows a pending charge that does not match the cart total.",
    "It has not left the fulfillment center. Switch the ship-to address to my office.",
    "On the product page, the noise-cancelling pair under 200 dollars says it works with Android.",
    "The photos show a zipper pocket. What arrived is just a slip pocket.",
    "Boots are going back. I still have the order email with the number.",
    "Two charges for the same tote on the same afternoon. One should not be there.",
    "Tracking still shows the order sitting in a warehouse. Which facility is that?",
    "The deal clock hit zero while I was typing the card. Did I lose the sale price?",
    "The candle set is a gift. Wrap it and do not print the price on anything inside.",
    "Sofa filters reset every time I sort low to high on the website.",
    "The download link died overnight. I bought the files yesterday and never saved them.",
    "June is on the preorder. If it slides to August I want out with no penalty.",
    "Saved card was declined. The bank app does not even show an attempt.",
    "The package from this order arrived last week. How long do I have to leave a product review?",
    "The bundle looks cheaper than the pieces. Is that before tax or after?",
    "I checked out as a guest. How do I find that purchase on Monday?",
    "The swatch was burgundy. What showed up is closer to rust in daylight.",
    "Leave it with the lobby desk, not the stoop. Where do I put that delivery note?",
    "Things I hearted say unavailable on my phone and look fine on desktop.",
    "The cable from the box frayed in a week. I need a replacement, not a care speech.",
    "They missed the day promised on the fast option. Does that fee come back?",
    "Trying to buy two skillets. The quantity field refuses to leave 1.",
    "Yesterday this code knocked off the delivery fee. Today the fee is back.",
    "If I pay with PayPal at the last step, do loyalty points still land?",
    "The unboxing clip shows a case and a strap. The bullet list only mentions the device.",
    "The invoice needs my company name for expenses. It currently has my personal name.",
    "After I paid, Tuesday quietly became the following Monday on the tracking page.",
    "Turn off promo texts from this shop's account. I still want order history.",
    "The carry-on in my cart is the soft one from the listing. Does it pass the airline sizer in the photos?",
    "The return QR in the order email expired. Send a new label to that inbox.",
    "These should have been king, not queen. Can the right set ship before I send these back?",
    "Is that phone number field at checkout actually required?",
    "Coffee plan: skip March, keep April. Do not cancel the whole subscription.",
    "The only-three-left banner has been up since the week before last.",
    "The refund came back as store credit. I asked for the card I paid with.",
    "The mixer order is in my account. Did warranty register with that purchase?",
    "Send it to the hotel so it arrives the morning I check in, under my name.",
    "The app died on the final tap. Now there is a pending charge and no receipt.",
    "On the product pages for the two standing desks, which listing is the walnut one?",
    "Both codes claim they combine. The total only accepts one.",
    "The seller flipped the status to shipped four days ago. The carrier still has zero scans.",
    "compare these sofa listings online",
    "Which size chart should I trust for this brand's online jackets?",
    "Add a second ship-to address without wiping the items already in my cart.",
    "Is free returns still available on this marketplace seller?",
    "The preorder page took payment. When do I get a firm ship date?",
]


def more_ecommerce(n: int, seen: set[str]) -> list[str]:
    out: list[str] = []
    products = [
        "trail runners", "ceramic cookware", "mechanical keyboard", "wool coat",
        "portable monitor", "yoga mat", "espresso machine", "kids bike helmet",
        "linen duvet", "noise cancelling headset", "stainless water bottle",
        "gaming mouse", "desk chair", "cast iron skillet", "smartwatch band",
        "backpack with laptop sleeve", "wireless charger", "cotton sheets",
    ]
    actions = [
        "Looking for {p} that will survive daily use",
        "Need {p} under a mid-range budget",
        "Can you help me compare {p} options on this site?",
        "The {p} in my cart disappeared after I logged in",
        "Does this {p} listing include free shipping past a certain total?",
        "I want to return the {p} I ordered last week",
        "Is there a discount code that still works for {p}?",
        "The review photos for this {p} do not match the main gallery",
        "Track the {p} that was marked shipped yesterday",
        "Swap the color on my {p} order before it leaves the warehouse",
        "Why is checkout blocking the {p} because of an address check?",
        "Add gift wrap to the {p} without printing the price",
        "The {p} page says in stock but the cart says unavailable",
        "I need the invoice PDF for the {p} purchase from Tuesday",
        "Will the {p} arrive before the weekend if I pick express?",
        "Someone left a review saying the {p} sizing runs small. Is that still true?",
        "Cancel the {p} order if it has not been picked yet",
        "The seller of this {p} has low ratings. Any safer listing?",
        "Apply store credit to the {p} sitting in checkout",
        "Why did the price of this {p} jump after I refreshed?",
    ]
    long = [
        "I am on mobile and the desktop site behaves differently.",
        "Please keep the answer about this order, not a generic FAQ.",
        "The confirmation email never showed up in spam either.",
        "I already tried another browser and the same error appears.",
        "It is a gift so timing matters more than saving a few dollars.",
    ]
    while len(out) < n:
        p = pick(products)
        text = pick(actions).format(p=p)
        if rng.random() < 0.35:
            text = f"{text}. {pick(long)}"
        elif rng.random() < 0.2:
            text = text.rstrip(".") + "?"
        uniq_add(out, seen, text)
    return out


# ---------------------------------------------------------------------------
# RETAIL
# ---------------------------------------------------------------------------

RETAIL_SEED = [
    "What time does the downtown branch actually lock up on Sundays?",
    "Blue runners in athletic. Still on the wall, or did the morning crowd finish them?",
    "Unscented detergent. I already walked household twice.",
    "I'm coming to the Maple store today. If the item is in the back, I'll take it at the counter.",
    "The fitting rooms are a line. Any larger stall, or is this the only bank of them?",
    "The garage is full. Is the lot across the street validated if I buy something?",
    "The tag in home goods says 24.99 and the shelf strip says 19.99. I need a person.",
    "The window sign is lower than the sticker in my hand. Do you honor the sign at the register?",
    "Same-day swap. Which desk do I walk up to?",
    "Does the pharmacy counter stay open straight through lunch?",
    "Seasonal mugs were on an endcap near the front. I've looped the entrance twice.",
    "Hold these boots at the counter while I grab socks?",
    "The machine hates my tote. Who's the person with the badge near lane 4?",
    "Do you still cut keys at this branch?",
    "The sign by the hook says three on hand. I've checked the hook and the back of the rack.",
    "Which door drops me closest to grocery?",
    "The jacket on the mannequin by shoes. Is a large actually on the floor?",
    "Bought it here last month on my card. The slip is gone. Can the counter still take it back?",
    "Restroom on this floor, or do I go upstairs?",
    "The perfume testers are dry. Any idea when they refill that table?",
    "Bridal appointment this Saturday, late morning if you have it.",
    "These pants need a hem before dinner tonight. Is the tailor in?",
    "The deli line is backed into the freezers. Are you handing out numbers?",
    "Travel-size toiletries. I always end up in the regular shampoo aisle.",
    "The floor-model sofa by the cafe. Is that one actually for sale?",
    "The door sign says 9. The gate was halfway down at 8:40 and nobody was there.",
    "At the jewelry counter, is cash fine or card only?",
    "The circular showed a clearance rack near the entrance. I don't see it.",
    "Can I walk up to optical for an adjustment, or do I need a time?",
    "This cart's front wheel is stuck. Where are the decent ones?",
    "Somewhere to sit while the prescription is counted. The benches by the entrance are full.",
    "Which lane is for bringing stuff back, as opposed to just paying?",
    "The baker said sourdough hits the case at 10. Still true on weekdays?",
    "I need automotive and I keep landing in seasonal. Can someone walk me over?",
    "Any outlet near the upstairs cafe I can borrow for ten minutes?",
    "I left a black jacket in a fitting room about twenty minutes ago.",
    "It's raining. Is the garden-center door open, or only the main entrance?",
    "The shelf price didn't match the beep. Can we sort that out at this register?",
    "Kids' shoes with a wide toe. Which wall?",
    "I want to hand a return off at the curb so I don't have to park and come in.",
    "Is gift wrap still at the service desk?",
    "The advertised chicken was gone by noon. Do you still write rain checks?",
    "Where's lost and found?",
    "I sprayed something at the fragrance table and now I'm lightheaded. Is there a water fountain close?",
    "A personal shopper on Thursday evening, after work.",
    "Tire center today. Do I need a time slot, or do I just pull in?",
    "The light is out in stall 4. The other mirrors are fine.",
    "Your brand of oat milk. Cooler by dairy, or did it move to the plant-based shelf?",
    "I'm meeting someone at the coffee kiosk. Which floor is that on?",
    "The security tag is still on this shirt. I already paid at lane 2.",
    "compare these sofas on the showroom floor",
    "Can you check backroom stock for SKU 44821 before I drive over?",
    "We need a stock transfer from the warehouse to aisle 12 by opening.",
    "Shelf availability for the white paint quarts looks wrong on the handheld.",
    "Who handles SKU management for seasonal toys this week?",
]


def more_retail(n: int, seen: set[str]) -> list[str]:
    out: list[str] = []
    places = [
        "aisle 7", "the back room", "customer service", "the garden center",
        "pharmacy", "electronics", "home goods", "the fitting rooms",
        "lane 3", "the loading dock", "seasonal", "automotive", "the deli case",
        "the endcap by the entrance", "optical", "the tire bay",
    ]
    items = [
        "paper towels", "wide-width shoes", "gluten-free pasta", "AA batteries",
        "indoor paint", "phone chargers", "winter gloves", "baby wipes",
        "extension cords", "cast iron pans", "gift cards", "plant food",
    ]
    patterns = [
        "Where do you keep {item} in this store?",
        "Is {item} still on the shelf in {place}?",
        "Can someone check {place} for {item}?",
        "I need {item}. Point me toward {place} if that is right.",
        "The handheld says {item} is in {place}, but I do not see it.",
        "Transfer {item} from the warehouse to {place} if you can.",
        "Do you have backroom stock of {item}, or only what is on the floor?",
        "Who can override the shelf label for {item} at the register?",
        "I am standing near {place}. Where should I look next for {item}?",
        "Store ops question: how often do you restock {item} during the day?",
        "SKU for {item} looks mismatched on the shelf strip in {place}.",
        "Can I pick up {item} at curbside from this location today?",
        "The display in {place} is empty. Any {item} left in the cage?",
        "I need help finding {item} without walking the whole store again.",
        "Is inventory for {item} shared with the other branch, or only here?",
        "Before I leave, can you confirm {place} has {item} in my size?",
        "The associate said {item} would be on the floor by noon. Still true?",
        "I want to return {item} at the desk by {place}. Is that the right counter?",
        "Are you open early enough that I can grab {item} before work?",
        "Showroom question: can I sit on the floor model before deciding on {item}?",
    ]
    extras = [
        "I am already in the building.",
        "Parking is terrible so I need a fast answer.",
        "No need to place an online order; I will take it today.",
        "A manager would help if associates are busy.",
        "This is about the physical store, not the website.",
    ]
    while len(out) < n:
        text = pick(patterns).format(item=pick(items), place=pick(places))
        if rng.random() < 0.3:
            text = f"{text} {pick(extras)}"
        uniq_add(out, seen, text)
    return out


# ---------------------------------------------------------------------------
# LEGAL
# ---------------------------------------------------------------------------

LEGAL_SEED = [
    "The deposit has been with the landlord for six weeks. No itemized list, no reply.",
    "The NDA from my old job. Can a portfolio show the work if I never name the client?",
    "Unpaid invoice, small claims. How long do I actually have before a filing is late?",
    "The landlord says the lease renewal notice was mailed. I never received it, and the tenancy rolled anyway.",
    "Will or living trust for a small estate?",
    "The kitchen is half finished, the contractor stopped answering, and the completion date was the 12th.",
    "The non-compete says two years. I'm moving states for a similar role. Does the clause follow me?",
    "Their insurer is offering far less than the body shop for a crash the other driver admitted. What can I do if I reject that amount?",
    "The brand name goes on packaging next month. Do I file before that, or can it wait?",
    "The association fined me for a paint color that is on their own approved sheet. I want the appeal path.",
    "What if I miss a jury summons?",
    "The parenting order says alternate weekends. We've followed a different rhythm for a year and nobody filed anything.",
    "A client is threatening a suit over a late handoff. The thread shows they changed the spec twice.",
    "Handshake on a used car, no money exchanged yet. Does that agreement mean anything?",
    "I'm salaried with no direct reports, and extra hours vanished from the last three paychecks.",
    "A collector sent a letter for an account I closed. I need to answer without restarting the clock by accident.",
    "We both want a prenup and neither of us has counsel yet. What's a sane first step?",
    "They changed the written terms a week after I signed. I never agreed to the new version. Does the original still control?",
    "I want the property manager on a recording, and I'll say so at the start. Is telling them enough?",
    "A neighbor's tree came down on my fence in the storm. Their insurer says the fence is my problem.",
    "I got a cease-and-desist over a review. Everything in it is something I can document.",
    "Written agreement, they missed payment in March. When does the window to file actually close?",
    "A cofounder is leaving. The operating agreement never says how to buy back their shares.",
    "Can I fight this ticket in writing?",
    "My mother died in May and her name is still on the deed. Which filing comes first?",
    "The school went through my kid's backpack and didn't call me until after. Was that allowed?",
    "A client is still running my photos in ads. The license ended in June.",
    "A three-day notice is taped to the door. Rent was late once, not unpaid for months. Is that notice valid?",
    "My father is lucid and wants me on the bank accounts. Full power of attorney, or something lighter?",
    "A subpoena just arrived for our email. What has to be frozen before anyone cleans out an inbox?",
    "A clinic billed me for a visit I canceled a week ahead. The no-show line was a footnote in the intake packet.",
    "The dates on my visa stamp don't match the petition. Who do I contact before I travel?",
    "The mediator wrote up something we never agreed to. I haven't signed. Am I bound anyway?",
    "I'm trying to clear an old misdemeanor. Which records do I pull before I apply?",
    "This publishing contract wants every right, forever, including formats that don't exist yet. Which pieces are out of line?",
    "The membership contract has a year left. I moved fifty miles and the cancellation clause never mentions distance. Can they enforce the rest?",
    "The tenant left mid-lease, rent stopped, and furniture is still in the unit. What are we allowed to do with it?",
    "I saw the crash. Counsel for one of the drivers wants a statement this week. Do I have to give one?",
    "The city cited my home office because clients come by twice a week. The block is already mixed use.",
    "I need a formal demand for the balance before anyone files. What has to be in it?",
    "My manager summarized my medical leave in a team channel. About a dozen people saw it.",
    "We need paperwork for a contractor that doesn't quietly turn them into staff.",
    "The warranty contract is still in force. They denied hail damage, and hail is not listed in the exclusions. Does that denial breach the warranty?",
    "Grandparents want time with the kids and both parents are refusing. Is a court even an option?",
    "I co-signed, the borrower defaulted, and I haven't answered. How close is a wage garnishment?",
    "A letter says our app icon infringes a patent. The drawing they attached looks nothing like it.",
    "The decree splits the refund. They already filed as single and kept the whole amount.",
    "We both emailed yes to moving the deadline to May. The original is a signed PDF. Does the email amend it?",
    "A new pet rule would push out a dog the prior lease explicitly allowed. I've lived here six years.",
    "What does without prejudice mean on a settlement draft?",
    "they changed the agreement after I signed it",
    "Is arbitration mandatory under this employment offer, or can I keep court access?",
    "Consumer rights question: the store sold a defective appliance and refuses a cash refund.",
    "Negligence claim after a fall on an unmarked wet floor. What evidence should I preserve?",
    "Litigation timeline: how long after service does a response usually need to be filed?",
]


def more_legal(n: int, seen: set[str]) -> list[str]:
    out: list[str] = []
    subjects = [
        "lease", "employment contract", "arbitration clause", "noncompete",
        "settlement offer", "eviction notice", "power of attorney", "NDA",
        "warranty", "demand letter", "custody order", "contractor agreement",
        "consumer complaint", "subpoena", "insurance denial", "copyright notice",
    ]
    patterns = [
        "What does this {s} actually require me to do next?",
        "Can they enforce the {s} after the facts changed?",
        "I need plain language on whether the {s} is still binding.",
        "Deadlines on the {s} are unclear. How do I read them?",
        "They ignored the {s}. What options do I have without escalating too fast?",
        "Is the {s} valid if I never received a signed copy?",
        "How should I respond to a {s} without creating new liability?",
        "Does a handwritten note modify the {s}, or only a formal amendment?",
        "The other side cited the {s}. I think they misread it.",
        "Who usually has standing to challenge a {s} like this?",
        "Preserve-or-lose question: what records matter for a dispute over this {s}?",
        "I signed the {s} under time pressure. Can that matter later?",
        "Explain the {s} like I am preparing for a consult, not asking for free advice.",
        "Is arbitration locked in by this {s}, or is court still available?",
        "Landlord-tenant angle: how does the {s} interact with local housing rules?",
        "Employment law angle: does the {s} survive if I change roles at the same company?",
        "Consumer rights angle: can the {s} waive a refund I am otherwise owed?",
        "Litigation posture: should a {s} go out before filing?",
        "They served a {s} at my workplace. What do I do in the next 48 hours?",
        "Negligence context: does the {s} limit damages even if they were careless?",
        "Signed the {s} already.",
        "Need help reading this {s}.",
        "Is this {s} enforceable?",
        "{s} arrived today.",
        "Question about my {s}.",
    ]
    tails = [
        "I am not asking you to represent me, just to map the issues.",
        "Dates and documents are available if that changes the analysis.",
        "Keep it practical; I have a consult scheduled next week.",
        "Please flag what depends on local law.",
        "This started after they changed terms without a clear yes from me.",
    ]
    while len(out) < n:
        text = pick(patterns).format(s=pick(subjects))
        roll = rng.random()
        if roll < 0.35:
            text = f"{text} {pick(tails)}"
        elif roll < 0.55 and text.endswith("?"):
            text = text[:-1] + "."
        uniq_add(out, seen, text)
    return out


def balance_lengths(items: list[str], target_mean: float, rounds: int = 8) -> list[str]:
    """Lightly reshuffle selection already built; lengths handled at fill time."""
    return items


def fill_class(seed: list[str], builder, need: int, seen: set[str]) -> list[str]:
    bucket: list[str] = []
    for text in seed:
        uniq_add(bucket, seen, text)
    # Prefer a mix of short and long while filling.
    extra = builder(need * 3, seen)
    # Sort candidates by distance from class-agnostic length bands to encourage mix
    shorts = [t for t in extra if len(t.split()) <= 8]
    medium = [t for t in extra if 9 <= len(t.split()) <= 18]
    longs = [t for t in extra if len(t.split()) >= 19]
    rng.shuffle(shorts)
    rng.shuffle(medium)
    rng.shuffle(longs)
    # Interleave to keep length distributions similar across classes
    pools = [shorts, medium, longs]
    idx = 0
    while len(bucket) < need:
        pool = pools[idx % 3]
        idx += 1
        if pool:
            text = pool.pop()
            if text not in bucket:
                bucket.append(text)
        elif any(pools):
            continue
        else:
            # emergency: generate more
            more = builder(need, seen)
            rng.shuffle(more)
            for text in more:
                if len(bucket) >= need:
                    break
                if text not in bucket:
                    bucket.append(text)
            break
    # If still short, keep generating
    guard = 0
    while len(bucket) < need and guard < 50:
        guard += 1
        more = builder(need * 2, seen)
        for text in more:
            if len(bucket) >= need:
                break
            if text not in bucket:
                bucket.append(text)
    if len(bucket) < need:
        raise RuntimeError(f"could not fill class to {need}, got {len(bucket)}")
    return bucket[:need]


def confusion_candidates(rows: list[dict]) -> list[dict]:
    """Heuristic pairs that look like boundary cases."""
    keywords = {
        "ECOMMERCE": ["online", "cart", "checkout", "shipping", "listing", "order", "refund", "website"],
        "RETAIL": ["store", "aisle", "shelf", "showroom", "register", "fitting", "warehouse", "sku", "floor"],
        "LEGAL": ["contract", "lease", "arbitration", "signed", "court", "notice", "liability", "agreement"],
        "GENERIC": ["explain", "what is", "why", "how does", "capital", "meaning"],
    }
    scored = []
    for row in rows:
        text_l = row["text"].lower()
        hits = []
        for intent, words in keywords.items():
            if intent == row["intent"]:
                continue
            if any(w in text_l for w in words):
                hits.append(intent)
        if hits:
            scored.append({"text": row["text"], "intent": row["intent"], "also_suggests": hits})
    rng.shuffle(scored)
    return scored[:25]


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    need = 500

    generic = fill_class(
        GENERIC_SCIENCE + GENERIC_GEO + GENERIC_HISTORY + GENERIC_TECH + GENERIC_MATH + GENERIC_GK,
        more_generic,
        need,
        seen,
    )
    ecommerce = fill_class(ECOMMERCE_SEED, more_ecommerce, need, seen)
    retail = fill_class(RETAIL_SEED, more_retail, need, seen)
    legal = fill_class(LEGAL_SEED, more_legal, need, seen)

    rows: list[dict] = []
    for intent, texts in (
        ("GENERIC", generic),
        ("ECOMMERCE", ecommerce),
        ("RETAIL", retail),
        ("LEGAL", legal),
    ):
        for text in texts:
            rows.append({"text": text, "intent": intent})

    rng.shuffle(rows)

    # Validation
    texts = [r["text"] for r in rows]
    assert len(rows) == 2000, len(rows)
    counts = Counter(r["intent"] for r in rows)
    assert counts == {"GENERIC": 500, "ECOMMERCE": 500, "RETAIL": 500, "LEGAL": 500}, counts
    norm = [re.sub(r"\s+", " ", t.strip().lower()) for t in texts]
    dup = len(norm) - len(set(norm))
    assert dup == 0, dup

    avg_len = {}
    q_ratio = {}
    for intent in ("GENERIC", "ECOMMERCE", "RETAIL", "LEGAL"):
        subset = [r["text"] for r in rows if r["intent"] == intent]
        lengths = [len(t.split()) for t in subset]
        avg_len[intent] = round(sum(lengths) / len(lengths), 2)
        q_ratio[intent] = round(sum(1 for t in subset if "?" in t) / len(subset), 3)

    # Length distributions should be in a similar band (not class-predictive)
    means = list(avg_len.values())
    if max(means) - min(means) > 8:
        print("WARNING: average lengths differ a lot across classes:", avg_len)

    report = {
        "total_samples": 2000,
        "per_class_counts": dict(counts),
        "average_length_by_class": avg_len,
        "duplicate_count": dup,
        "question_ratio_by_class": q_ratio,
        "length_spread_across_class_means": round(max(means) - min(means), 2),
        "class_confusion_candidates": confusion_candidates(rows),
        "validation": {
            "no_duplicates": dup == 0,
            "balanced_classes": all(v == 500 for v in counts.values()),
            "length_distribution_similar": (max(means) - min(means)) <= 8,
            "mixture_of_questions_and_statements": all(0.15 <= q_ratio[i] <= 0.85 for i in q_ratio),
        },
    }

    with OUT_JSONL.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    OUT_REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUT_JSONL} ({len(rows)} rows)")
    print(f"Wrote {OUT_REPORT}")
    print(json.dumps(report["validation"], indent=2))
    print("avg lengths:", avg_len)
    print("question ratios:", q_ratio)


if __name__ == "__main__":
    main()
