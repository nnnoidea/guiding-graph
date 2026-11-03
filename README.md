# Guiding Graph

Code for **[Guiding Graph: Explicit Query-Structure Guidance for Knowledge Graph Question Answering](https://aclanthology.org/2026.emnlp-main.2312/)** (EMNLP 2026).
This repository contains the Freebase implementation for WebQSP and CWQ.

## Usage

1. Copy `.env.example` to `.env` and set the model API key, API base URL, model name,
   and Freebase SPARQL endpoint. Queries require a Virtuoso-compatible Freebase service.
2. Set `DATA_PATH` and `OUTPUT_PATH` at the top of `main.py`. Input is a JSON list
   of question records; output is a new JSONL file. All questions are processed by default.
3. Run question answering:

```sh
python main.py
```

WebQSP records use `RawQuestion`; CWQ records use `question`.
`topic_entity` maps Freebase IDs to entity names for disambiguation.

`MAX_ANSWER_TRIPLES` controls the evidence limit (`None` uses all retrieved triples).
`ALLOW_OWN_KNOWLEDGE` controls whether the answer prompt permits model knowledge
beyond the retrieved evidence.
