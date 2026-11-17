# Intent classification data

This project will classify a user query into one of four intents. The dataset is ready. Model code is not included yet.

## Intents

| Intent | What it covers |
| --- | --- |
| `GENERIC` | Everyday questions: weather, how-tos, travel, food, language, trivia, and small tasks that are not about buying or the law. |
| `ECOMMERCE` | Online buying: product pages, checkout, payment, delivery, digital goods, subscriptions, refunds, and marketplace sellers. |
| `RETAIL` | A physical store: hours, aisles, fitting rooms, the service desk, in-store stock, curbside handoff, and counter exchanges. |
| `LEGAL` | Rights, contracts, courts, housing, work, family, and intellectual property. These are phrased as real questions, not as advice. |

Online checkout and shipping belong to `ECOMMERCE`. Standing in a store, or going there to pick something up, belongs to `RETAIL`.

## Data

Labeled queries live in `data/intents.jsonl`. Each line is one JSON object:

```json
{"text": "the user query", "intent": "GENERIC"}
```

`intent` is one of `GENERIC`, `ECOMMERCE`, `RETAIL`, or `LEGAL`.

There are 70 GENERIC, 50 ECOMMERCE, 50 RETAIL, and 70 LEGAL examples (240 lines total). GENERIC includes longer everyday queries, and LEGAL includes short ones, so length is not a stand-in for the class. Queries mix fragments and conversational sentences so the classes are not tied to a single keyword.

## Layout

```
project/
├── data/
│   └── intents.jsonl
├── src/
├── tests/
├── notebooks/
└── README.md
```

`src/`, `tests/`, and `notebooks/` are empty on purpose. No training code, PyTorch, or Hugging Face usage is in this repository yet.
"# intent-transformer" 
