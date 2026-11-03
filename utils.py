"""Prompt templates, model calls and bidirectional Freebase queries.

KG query return formats:
- get_related_nodes_by_id: (relation, name, entity_id)
- get_related_nodes: (label, name, entity_id, relation, source_id)
Neither layout encodes edge direction; both directions are intentionally merged.
"""

import json
import logging
import os
import re
from pathlib import Path

logger = logging.getLogger(__name__)

MODEL_NAME = "deepseek-v3"
API_BASE_URL = None  # Set OPENAI_BASE_URL for an OpenAI-compatible provider.
SPARQLPATH = "http://localhost:1111/sparql"
PROMPT_LOG_PATH = None  # Optional Path to a JSONL prompt/response log.
_client = None


def configure():
    """Read .env once at startup; importing this module makes no service calls."""
    from dotenv import load_dotenv

    global MODEL_NAME, API_BASE_URL, SPARQLPATH, _client
    load_dotenv(Path(__file__).resolve().parent / ".env")
    MODEL_NAME = os.getenv("MODEL_NAME", MODEL_NAME)
    API_BASE_URL = os.getenv("OPENAI_BASE_URL") or API_BASE_URL
    SPARQLPATH = os.getenv("SPARQL_ENDPOINT", SPARQLPATH)
    _client = None


def chat(prompt):
    """Call the configured model and return text plus token usage."""
    global _client
    if _client is None:
        from openai import OpenAI

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Set OPENAI_API_KEY in .env or your environment.")
        _client = OpenAI(api_key=api_key, base_url=API_BASE_URL)
    response = _client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
    )
    text = response.choices[0].message.content
    if text is None:
        raise ValueError("The model returned no text content.")
    if response.usage is None:
        raise ValueError("The model endpoint did not return token usage.")
    usage = {
        "total": response.usage.total_tokens,
        "input": response.usage.prompt_tokens,
        "output": response.usage.completion_tokens,
    }
    return text, usage


def _ask(stage, prompt):
    response, usage = chat(prompt)
    if PROMPT_LOG_PATH is not None:
        path = Path(PROMPT_LOG_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(
                {"stage": stage, "prompt": prompt, "response": response},
                ensure_ascii=False,
            ) + "\n")
    return response, usage


def prepare_dataset(file_path):
    """Read a JSON list of WebQSP or CWQ question records."""
    with Path(file_path).open(encoding="utf-8") as handle:
        dataset = json.load(handle)
    if not isinstance(dataset, list):
        raise ValueError("Expected a JSON list of question records; see README.md.")
    return dataset


def _sparql():
    from SPARQLWrapper import JSON, SPARQLWrapper

    client = SPARQLWrapper(SPARQLPATH)
    client.setReturnFormat(JSON)
    return client

complete_info_prompt = """
Rephrase the question as a statement, ensuring:  
1. **All explicit and logically implied information is included** (e.g., location, time, scope if hinted in the question).  
2. Split compound noun phrases into individual entities.  
3. Classify each entity as generic/specific.  
Do not answer the question. 
question: What team did Payton Manning's father play for?
answer:
statement: Identify the team payton manning's father played for.
entities: 
  - team (generic)
  - payton manning (specific)
  - father (generic)
question: what did james k polk do before he was president?
answer:
statement: Identify the job of James K. Polk before he became president.
entities: 
  - job (generic)
  - James K. Polk (specific)
  - president (generic)
question: Where did the "Country Nation World Tour" concert artist go to college?
answer:
statement: Identify the college attended by the concert artist of the "Country Nation World Tour."
entities:
    - college (generic)
    - Country Nation World Tour (specific)
    - concert artist (generic)
Now answer with the format of the example above. Be brief and precise.
question: {}
answer:"""

structure_prompt = """
Please analyze the relationships between the following entities in the given sentence. Represent each relationship as a triple in the format (subject, relation, object). Then identify the target entity of the sentence. 
These keywords can be used as entities or relationships in the triples, but please do not modify the keywords. If a generic keyword and another keyword refer to the same entity, the generic keyword should be treated as a relation rather than an entity.
sentence: Identify the father of Keyshia Cole.
generic keywords: father
specific keywords: Keyshia Cole
answer:
triples: [(father, of, Keyshia Cole)]
target entity: father
sentence: Identify the main trading partner of China that appointed Abdelaziz Bouteflika to a governmental position.
generic keywords: main trading partner; governmental position
specific keywords: China; Abdelaziz Bouteflika
answer:
triples: [(main trading partner, of, China), (main trading partner, appointed to governmental position, Abdelaziz Bouteflika)]
Now answer with the format of the example above. Be brief and precise.
sentence: {}
generic keywords: {}
specific keywords: {}
answer:"""

prune_prompt = """
Select the most semantically matching relation from the candidate relations to replace the given relation in the theme information while maintaining its essential meaning. Rate it on a scale of 0 to 10. Be brief and precise.
Below is an example:
theme information:  main spoken language of country
relation: of
candidate relations: ['language.human_language.main_country', 'language.human_language.human_language', 'language.human_language.language_family', 'language.human_language.iso_639_3_code', 'base.rosetta.languoid.parent', 'language.human_language.writing_system', 'base.rosetta.languoid.languoid_class'] 
assess: 
- 'language.human_language.human_language' can perfectly replace the relation, (9) score.
- 'language.human_language.main_country' can replace the relation, (8) score.
- other relations are not related to any information in the sentence, (0) score.
Now answer with the format of the example above. Be brief and precise.
theme information: {}
relation: {}
candidate relations: {}
assess: """

choose_prompt = """
Based on the target sentence below, determine which of the provided pieces of knowledge better matches the target sentence. Be breif and precise.
Target sentence: Identify the country that Turkey trades with which contains the Annaba Province.
Knowledge 1: country location.adjoining_relationship.adjoins turkey
Knowledge 2: country location.contain.contains Annaba
Answer: Knowledge (2) better matches the target sentence.
Target sentence: {}
Knowledge 1: {}
Knowledge 2: {}
Answer:"""

answer_by_triples_prompt = """
Please answer the question. Triples are available for reference. If there are not enough information in the triples, please answer with your own knowledge.
question: Which place is the madam satan located?
triplets: [('madam satan', 'film.film.country', 'the USA'), ('madam satan', 'film.film.language', 'English')]
answer: 'madam satan' is located in 'the USA'.
question: {}
triplets: {}
answer: """

def complete_info(question):
    """Normalize the question and extract specific/generic clues."""
    response, usage = _ask("complete_info", complete_info_prompt.format(question))
    statement = ""
    specific_entities = []
    generic_entities = []
    in_entities = False
    # Specific clues anchor retrieval; generic clues represent unknown concepts.
    for line in response.splitlines():
        line = line.strip().lower()
        if line.startswith("statement:"):
            statement = line.split(":", 1)[1].strip()
        elif line.startswith("entities:"):
            in_entities = True
        elif in_entities and line.startswith("-"):
            parts = line.split("(", 1)
            if len(parts) == 2:
                # Remove the list marker, not hyphens inside an entity name.
                entity = parts[0][1:].strip()
                entity_type = parts[1].rstrip(")").strip()
                if entity_type == "specific":
                    specific_entities.append(entity)
                elif entity_type == "generic":
                    generic_entities.append(entity)
        elif in_entities and not line:
            in_entities = False
    if not statement:
        raise ValueError("Missing statement in the information-extraction response.")
    return statement, specific_entities, generic_entities, usage


def structure(statement, generic, specific):
    """Build question-graph triples and identify the answer placeholder."""
    prompt = structure_prompt.format(statement, "; ".join(generic), "; ".join(specific))
    response, usage = _ask("structure", prompt)
    # Parse the graph triples separately from the answer placeholder.
    triples = []
    for match in re.findall(r"\(([^)]+)\)", response):
        elements = tuple(element.strip() for element in match.split(","))
        if len(elements) != 3 or not all(elements):
            raise ValueError(f"Malformed question-graph triple: {match!r}")
        triples.append(elements)
    target = None
    for line in response.splitlines():
        line = line.strip()
        if line.lower().startswith("target entity:"):
            target = line.split(":", 1)[1].strip()
            break
    if not triples or not target:
        raise ValueError("Missing triples or target entity in the graph response.")
    return triples, target, usage


def choose(target_sentence, knowledge1, knowledge2):
    """Return the selected branch number, or None for an unparseable response."""
    prompt = choose_prompt.format(target_sentence, knowledge1, knowledge2)
    response, usage = _ask("choose", prompt)
    matches = re.findall(r"\((.*?)\)", response)
    return (matches[0].strip() if matches else None), usage


def prune_node(matched_clues, relations, pruned_info):
    """Score candidate relations under the current question-graph triple.

    Targets use underscore-delimited triples. Relations scoring below 6
    are discarded.
    """
    theme_parts = matched_clues[0].split("_")
    if len(theme_parts) != 3:
        raise ValueError("Expected a three-part target clue separated by underscores.")
    prompt = prune_prompt.format(" ".join(theme_parts), theme_parts[1], relations)
    response, usage = _ask("prune_node", prompt)
    # Accept only returned predicates present in the candidate relation set.
    for line in response.splitlines():
        line = line.lower()
        scores = re.findall(r"\((\d+)\)", line)
        if not scores or int(scores[0]) < 6:
            continue
        matches = re.findall(r"[\"'“”‘’](.*?)[\"'“”‘’]", line)
        for relation in matches:
            if relation in relations:
                pruned_info.setdefault(matched_clues[0], []).append(
                    (relation, relation.split(".")[-1], int(scores[0]))
                )
    return pruned_info, usage


def answer_question(question, triples, allow_own_knowledge=True):
    """Generate an answer from retrieved triples and optional model knowledge."""
    evidence = ", ".join(str(triple) for triple in triples)
    prompt = answer_by_triples_prompt
    if not allow_own_knowledge:
        prompt = prompt.replace(
            "If there are not enough information in the triples, please answer with your own knowledge.",
            "Use only the supplied triples. If they are insufficient, state that the answer cannot be determined.",
        )
    return _ask("answer", prompt.format(question, evidence))


def query_node_by_name(name):
    # Combine Virtuoso full-text lookup with case-insensitive exact label matching.
    sparql = _sparql()
    safe_name = name.replace("'", "\\'")
    safe_name = safe_name.replace('"', '\\"')
    sparql_txt = 'SELECT ?entity \n    WHERE {\n    ?entity rdfs:label ?label .\n    ?label bif:contains \'"%s"\' .\n    FILTER(LCASE(STR(?label)) = LCASE("%s"))\n    }' % (safe_name, safe_name)
    sparql.setQuery(sparql_txt)
    try:
        results = sparql.query().convert()
        nodes = []
        for item in results['results']['bindings']:
            node_id = item['entity']['value'].split('/')[-1]
            nodes.append((name, node_id))
        return nodes
    except Exception:
        logger.exception('Entity lookup failed for %r', name)
        raise

def get_name_by_id(id):
    sparql = _sparql()
    sparql_txt = 'SELECT ?label \n    WHERE {\n    <id> rdfs:label ?label .\n    FILTER(LANG(?label) = "en")\n    }'.replace('id', 'http://rdf.freebase.com/ns/' + id)
    sparql.setQuery(sparql_txt)
    results = sparql.query().convert()
    if len(results['results']['bindings']) == 0:
        return 'UnName_Entity'
    return results['results']['bindings'][0]['label']['value']

def count_relations(id):
    sparql = _sparql()
    sparql_txt = '\n    SELECT (COUNT(DISTINCT ?r) AS ?relationCount) WHERE {\n        {\n            <id> ?r ?e.\n        } UNION {\n            ?e ?r <id>.\n        }\n    }\n    '.replace('id', 'http://rdf.freebase.com/ns/' + id)
    sparql.setQuery(sparql_txt)
    results = sparql.query().convert()
    return int(results['results']['bindings'][0]['relationCount']['value'])

def count_nodes(id):
    sparql = _sparql()
    sparql_txt = '\n    SELECT (COUNT(DISTINCT ?e) AS ?entityCount) WHERE {\n        {\n            <id> ?r ?e.\n        } UNION {\n            ?e ?r <id>.\n        }\n    }\n    '.replace('id', 'http://rdf.freebase.com/ns/' + id)
    sparql.setQuery(sparql_txt)
    results = sparql.query().convert()
    return int(results['results']['bindings'][0]['entityCount']['value'])

def get_relations_by_id(id, tail=True):
    sparql = _sparql()
    if tail:
        sparql_txt = 'select DISTINCT ?r where {<id> ?r ?e.} \n            '.replace('id', 'http://rdf.freebase.com/ns/' + id)
    else:
        sparql_txt = 'select DISTINCT ?r where {?e ?r <id>.} \n            '.replace('id', 'http://rdf.freebase.com/ns/' + id)
    sparql.setQuery(sparql_txt)
    results = sparql.query().convert()
    result = []
    for item in results['results']['bindings']:
        relation = item['r']['value'].replace('http://rdf.freebase.com/ns/', '')
        if item['r']['value'].split('/')[-1] in ['type.object.name', 'owl#sameAs', 'rdf-schema#label', 'common.topic.description', 'type.object.type', 'type.object.key']:
            continue
        if item['r']['value'].startswith('http://rdf.freebase.com/key/'):
            continue
        if item['r']['value'].startswith('http://www.w3.org/1999/02/22-rdf-syntax-ns#type'):
            continue
        if item['r']['value'].split('/')[-1].startswith('common.') or item['r']['value'].split('/')[-1].startswith('freebase.'):
            continue
        result.append((relation, id))
    return result

def get_related_nodes_by_id_directed(id, tail=True):
    sparql = _sparql()
    if tail:
        sparql_txt = 'select DISTINCT ?r ?e where {<id> ?r ?e.} Limit 10000\n            '.replace('id', 'http://rdf.freebase.com/ns/' + id)
    else:
        sparql_txt = 'select DISTINCT ?r ?e where {?e ?r <id>.} Limit 10000\n            '.replace('id', 'http://rdf.freebase.com/ns/' + id)
    sparql.setQuery(sparql_txt)
    results = sparql.query().convert()
    result = []
    for item in results['results']['bindings']:
        node_id = ''
        name = ''
        if item['r']['value'].split('/')[-1] in ['type.object.name', 'owl#sameAs', 'rdf-schema#label', 'common.topic.description', 'type.object.type', 'type.object.key']:
            continue
        if item['r']['value'].startswith('http://rdf.freebase.com/key/'):
            continue
        if item['r']['value'].startswith('http://www.w3.org/1999/02/22-rdf-syntax-ns#type'):
            continue
        if item['r']['value'].split('/')[-1].startswith('common.') or item['r']['value'].split('/')[-1].startswith('freebase.'):
            continue
        if item['e']['type'] == 'uri':
            node_id = item['e']['value'].split('/')[-1]
            if node_id == '' or node_id[0] != 'm':
                continue
            name = get_name_by_id(node_id)
        elif item['e'].get('xml:lang') == 'en':
            name = item['e']['value']
        result.append((item['r']['value'].replace('http://rdf.freebase.com/ns/', ''), name, node_id))
    return result

def get_related_nodes_by_id_and_relation(id, relation, tail=True):
    sparql = _sparql()
    if tail:
        sparql_txt = 'select DISTINCT ?e where {<id> <relation> ?e.} \n            '.replace('id', 'http://rdf.freebase.com/ns/' + id).replace('relation', 'http://rdf.freebase.com/ns/' + relation)
    else:
        sparql_txt = 'select DISTINCT ?e where {?e <relation> <id>.} \n            '.replace('id', 'http://rdf.freebase.com/ns/' + id).replace('relation', 'http://rdf.freebase.com/ns/' + relation)
    sparql.setQuery(sparql_txt)
    results = sparql.query().convert()
    result = []
    for item in results['results']['bindings']:
        node_id = ''
        if item['e']['type'] == 'uri':
            node_id = item['e']['value'].split('/')[-1]
            if not node_id or node_id[0] != 'm':
                continue
            name = get_name_by_id(node_id)
        else:
            name = item['e']['value']
        result.append((relation.split('.')[-1], name, node_id, relation, id))
    return result

def get_relations(id):
    # Merge both directions intentionally to avoid missing relevant relations.
    results = get_relations_by_id(id, True)
    results.extend(get_relations_by_id(id, False))
    return results

def get_related_nodes(id, r):
    # Expand the selected relation in both directions; direction is not a filter.
    results = get_related_nodes_by_id_and_relation(id, r, True)
    results.extend(get_related_nodes_by_id_and_relation(id, r, False))
    return results

def get_related_nodes_by_id(id):
    # Structural intersection uses the union of incoming and outgoing neighbors.
    results = get_related_nodes_by_id_directed(id, True)
    results.extend(get_related_nodes_by_id_directed(id, False))
    return results
