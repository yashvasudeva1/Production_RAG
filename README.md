# Production RAG

A production-oriented Retrieval-Augmented Generation system designed around **adaptive retrieval** rather than a single retrieval strategy.

The system does not treat every user query the same way. It first understands the query, extracts reliable document-level constraints, reduces the search space using metadata, and then routes the query to a retrieval strategy suited to the type of information being requested.

The goal is not simply to build a RAG demo that produces answers. The goal is to build a **production RAG architecture whose individual stages are modular, testable, replaceable, and measurable**.

---

## Architecture

```text
                              USER QUERY
                                  |
                                  v
                       +---------------------+
                       |   Query Classifier  |
                       +----------+----------+
                                  |
                     +------------+------------+
                     |            |            |
                    FACT       DETAILED      GENERAL
                     |            |            |
                     +------------+------------+
                                  |
                                  v
                    +--------------------------+
                    | Query Metadata Extraction|
                    +------------+-------------+
                                 |
                                 v
                    +--------------------------+
                    |    Metadata Filtering    |
                    +------------+-------------+
                                 |
                       Candidate Document Set
                                 |
              +------------------+------------------+
              |                  |                  |
              v                  v                  v
            FACT             DETAILED            GENERAL
              |                  |                  |
              v                  v                  v
        BM25 / Keyword      Parent-Child         Multi-Query
           Retrieval          Retrieval           Retrieval
              |                  |                  |
              +------------------+------------------+
                                 |
                                 v
                         +---------------+
                         |   Reranker    |
                         +-------+-------+
                                 |
                                 v
                              CONTEXT
                                 |
                                 v
                         +---------------+
                         |      LLM      |
                         +-------+-------+
                                 |
                                 v
                              ANSWER
```

---

# Why this architecture?

A conventional RAG pipeline often looks like:

```text
Query
  |
  v
Embedding
  |
  v
Vector Search
  |
  v
Top-K
  |
  v
LLM
```

That approach is simple, but it assumes that one retrieval strategy is appropriate for every query.

This project takes a different approach.

Different questions have different retrieval requirements.

A query asking for a specific fact can benefit from precise lexical retrieval.

A detailed question may require a small relevant passage together with the larger context surrounding it.

A broad question may benefit from multiple interpretations of the original query.

Therefore, the system separates:

```text
"What documents should I search?"
```

from:

```text
"What information should I retrieve from those documents?"
```

and from:

```text
"Which retrieved results are actually the most relevant?"
```

---

# Core Design Principles

## 1. Retrieval is adaptive

The system does not force every query through the same retriever.

The query is classified into one of three retrieval categories:

```text
fact
detailed
general
```

These categories determine the retrieval strategy.

```text
FACT
    -> Keyword / BM25 retrieval

DETAILED
    -> Parent-Child retrieval

GENERAL
    -> Multi-Query retrieval
```

---

## 2. Metadata filtering is candidate-space reduction

Metadata filtering is intentionally separate from retrieval.

The metadata layer answers questions such as:

```text
Who?
Where?
When?
What type of document?
Which department?
```

For example:

```text
"What did Google researchers propose in the 2017 Transformer paper?"
```

can produce:

```json
{
  "document_type": "research_paper",
  "organizations": ["Google"],
  "dates": ["2017"],
  "topics": ["Transformer"]
}
```

The filtering layer uses the reliable document-level constraints:

```text
document_type = research_paper
organization = Google
date = 2017
```

and produces a smaller candidate set.

The topic:

```text
Transformer
```

is not used as a hard metadata constraint.

Instead, it is passed forward as a retrieval signal.

This distinction is fundamental:

```text
Metadata filtering
    |
    v
WHERE should I search?

Retrieval
    |
    v
WHAT information should I retrieve?
```

---

# Query Processing Pipeline

## 1. User Query

The system receives a natural-language query.

Example:

```text
What did Google researchers propose in the 2017 Transformer paper?
```

The query is not immediately sent to a vector store or keyword retriever.

It first passes through the query-understanding layer.

---

# 2. Query Classification

The query classifier assigns one of three categories:

```text
fact
detailed
general
```

The classifier also provides a confidence score.

Conceptually:

```json
{
  "category": "fact",
  "confidence": 0.94
}
```

The category determines which specialized retrieval path will eventually be used.

### FACT

Used for queries that primarily seek a specific piece of information.

Example:

```text
What year was the Transformer paper published?
```

### DETAILED

Used when the question requires a deeper or more contextual answer.

Example:

```text
Explain how the Transformer architecture works and why self-attention
is important to it.
```

### GENERAL

Used for broader information-seeking questions.

Example:

```text
Tell me about machine learning.
```

These three categories form the current routing model for the system.

---

# 3. Query Metadata Extraction

After classification, the query is analyzed for explicit document-level constraints.

The current metadata representation is:

```python
{
    "document_type": str,
    "organizations": list[str],
    "locations": list[str],
    "dates": list[str],
    "department": str,
    "topics": list[str]
}
```

The extractor is deliberately conservative.

It should extract a constraint only when there is sufficient evidence that the query is explicitly referring to that metadata.

For example:

```text
Which research papers discuss machine translation?
```

becomes approximately:

```json
{
  "document_type": "research_paper",
  "organizations": [],
  "locations": [],
  "dates": [],
  "department": "",
  "topics": ["machine translation"]
}
```

The system should not convert every noun or keyword into a hard filter.

---

# 4. Metadata Filtering

The metadata filtering layer operates over document metadata.

Its responsibility is to reduce the candidate document set before retrieval.

The current hard-filter fields are:

```text
document_type
organizations
locations
dates
department
```

The current soft retrieval field is:

```text
topics
```

For example, if there are four documents:

```text
Document A
Document B
Document C
Document D
```

and the query specifies:

```text
document_type = research_paper
organization = Google
date = 2017
```

the filter can reduce:

```text
4 documents
     |
     v
1 candidate document
```

The retrieval system can then operate on that reduced candidate space.

This becomes particularly important as the corpus grows.

---

# Metadata Filtering Philosophy

Metadata filtering is designed to be **recall-oriented and conservative**.

Incorrect filtering is dangerous because once a document is removed from the candidate set, the downstream retriever can never recover it.

Therefore:

```text
False negative
    |
    v
Document disappears permanently
```

is much more harmful than:

```text
False positive
    |
    v
Document survives
    |
    v
Retriever / reranker can reject it later
```

For this reason, topics are not currently treated as hard filters.

---

# Fallback and Filter Relaxation

Metadata constraints can sometimes be too restrictive.

For example, the query may explicitly mention several constraints, but the corpus may not contain a document satisfying all of them simultaneously.

The metadata filter therefore supports controlled relaxation.

The relaxation order is:

```text
dates
locations
department
organizations
document_type
```

The system first attempts the full constraint intersection.

If no candidates are found, it progressively relaxes hard constraints until candidates are found.

The result records which filters were relaxed.

Example:

```json
{
  "relaxed_filters": [
    "dates"
  ],
  "used_fallback": true
}
```

This makes fallback behavior observable rather than silently changing the search scope.

---

# Metadata Filter Output

The filtering component produces a structured result:

```python
FilterResult(
    document_ids=[...],
    matched_count=...,
    total_documents=...,
    applied_filters={...},
    retrieval_signals={...},
    relaxed_filters=[...],
    used_fallback=...
)
```

### `document_ids`

The documents that survived metadata filtering.

### `matched_count`

Number of candidate documents.

### `total_documents`

Total number of documents considered.

### `applied_filters`

Hard metadata constraints that were actually applied.

### `retrieval_signals`

Information that should be passed to retrieval rather than used for hard filtering.

### `relaxed_filters`

Hard constraints that had to be removed to find candidates.

### `used_fallback`

Indicates whether relaxation was necessary.

---

# Document Processing

The system maintains document-level metadata separately from the retrieval process.

Documents can currently include:

```text
.txt
.md
.markdown
.pdf
.docx
.json
.csv
```

The metadata extraction stage reads the documents and produces structured metadata.

The document metadata schema includes:

```python
{
    "title": str,
    "summary": str,
    "document_type": str,
    "topics": list[str],
    "keywords": list[str],
    "people": list[str],
    "organizations": list[str],
    "locations": list[str],
    "dates": list[str],
    "language": str,
    "department": str,
    "author": str,
    "version": str,
    "entities": list[str]
}
```

The generated metadata is stored centrally in:

```text
metadata/
└── documents.json
```

Each document also has a stable `document_id`.

That identifier becomes the connection between document-level metadata and the downstream chunk-level retrieval system.

---

# Chunk-Level Retrieval

Metadata exists at the document level.

Retrieval, however, should normally operate over chunks.

The intended relationship is:

```text
Document
   |
   +-- Chunk 1
   +-- Chunk 2
   +-- Chunk 3
   +-- Chunk N
```

Each chunk should retain the originating:

```text
document_id
```

Therefore:

```text
Metadata Filter
      |
      v
Candidate document IDs
      |
      v
Chunks belonging to those documents
      |
      v
Specialized Retriever
```

This provides a clean boundary between candidate selection and passage retrieval.

---

# Retrieval Layer

The retrieval layer contains specialized retrievers.

All retrievers should operate behind a common interface so they can be replaced without changing the overall pipeline.

Conceptually:

```python
class Retriever:
    def retrieve(
        self,
        query,
        document_ids=None,
        top_k=5
    ):
        ...
```

Possible implementations include:

```text
BM25Retriever
ParentChildRetriever
MultiQueryRetriever
```

The routing layer should not need to know how each retriever internally works.

---

# Fact Retrieval

Fact-oriented queries use keyword-based retrieval.

The intended first implementation is:

```text
BM25
```

Pipeline:

```text
Query
  |
  v
Metadata Filter
  |
  v
Candidate Document IDs
  |
  v
Candidate Chunks
  |
  v
BM25
  |
  v
Top-K Chunks
```

This branch is intended for queries where precise lexical matching is valuable.

Example:

```text
What year was the Transformer paper published?
```

The retriever can exploit terms such as:

```text
Transformer
2017
published
```

rather than relying exclusively on semantic similarity.

---

# Detailed Retrieval

Detailed questions use parent-child retrieval.

The basic pattern is:

```text
Parent Document
      |
      +-- Child Chunk
      +-- Child Chunk
      +-- Child Chunk
      +-- Child Chunk
```

The system retrieves a highly relevant child chunk first.

Then it uses the relationship between the child and its parent to recover larger contextual information.

Conceptually:

```text
Query
  |
  v
Small child chunks
  |
  v
Precise retrieval
  |
  v
Identify parent
  |
  v
Return larger contextual content
```

This allows retrieval precision and contextual completeness to be handled separately.

---

# General Retrieval

General questions use multi-query retrieval.

The system takes the original query and creates multiple retrieval perspectives.

Conceptually:

```text
Original Query
      |
      v
Alternative Queries
      |
      +-- Query A
      +-- Query B
      +-- Query C
      +-- Query D
      |
      v
Independent Retrieval
      |
      v
Result Combination
      |
      v
Deduplication
      |
      v
Candidate Results
```

This is intended for broader information needs where one formulation of a query may not adequately cover the relevant content.

---

# Reranking

The specialized retrievers optimize the candidate set.

The reranker then improves the ordering of those candidates.

The intended pipeline is:

```text
Specialized Retriever
        |
        v
      Top-N
        |
        v
     Reranker
        |
        v
      Top-K
```

For example:

```text
BM25
 |
 v
20 candidate chunks
 |
 v
Reranker
 |
 v
5 highest-quality chunks
```

The retriever is therefore primarily responsible for recall.

The reranker is responsible for improving precision.

---

# Context Construction

After reranking, the selected results are assembled into the context supplied to the language model.

The context should retain document and chunk identity so that retrieved information remains traceable to its source.

Conceptually:

```text
Retrieved Chunk
      |
      +-- document_id
      +-- chunk_id
      +-- document metadata
      +-- text
```

The final context is then constructed from the highest-ranked relevant material.

---

# Generation

The LLM receives:

```text
User Query
+
Retrieved Context
```

and produces the final answer.

The generation layer is intentionally downstream of retrieval.

The LLM should not be responsible for discovering the entire corpus itself.

Instead:

```text
Retrieval
    |
    v
Find relevant evidence

LLM
    |
    v
Reason over that evidence
and generate the response
```

---

# End-to-End Flow

The complete system can therefore be viewed as:

```text
USER
 |
 v
QUERY
 |
 v
QUERY CLASSIFIER
 |
 +-------------+-------------+
 v             v             v
FACT        DETAILED       GENERAL
 |             |             |
 +-------------+-------------+
               |
               v
      QUERY METADATA EXTRACTION
               |
               v
       METADATA FILTERING
               |
               v
       CANDIDATE DOCUMENTS
               |
        +------+------+
        |      |      |
        v      v      v
      BM25   PARENT  MULTI
             CHILD   QUERY
        |      |      |
        +------+------+
               |
               v
            RERANKER
               |
               v
             CONTEXT
               |
               v
              LLM
               |
               v
             ANSWER
```

---

# Component Boundaries

One of the central goals of the project is to maintain clear boundaries between components.

## Query Classifier

Responsible for:

```text
Query
  |
  v
fact / detailed / general
```

It does not retrieve documents.

---

## Query Metadata Extractor

Responsible for:

```text
Query
  |
  v
document-level constraints
+
retrieval signals
```

It does not perform filtering itself.

---

## Metadata Filter

Responsible for:

```text
Document metadata
+
query metadata
  |
  v
candidate documents
```

It does not retrieve chunks.

---

## Retriever

Responsible for:

```text
Query
+
candidate documents
  |
  v
relevant chunks
```

It does not generate answers.

---

## Reranker

Responsible for:

```text
Retrieved candidates
  |
  v
better-ranked candidates
```

---

## LLM

Responsible for:

```text
Query
+
retrieved context
  |
  v
final answer
```

This separation makes each stage independently testable.

---

# Project Structure

```text
production rag/
|
+-- .env
|
+-- documents/
|   +-- ...
|
+-- metadata/
|   +-- documents.json
|
+-- prompts/
|   +-- query_classification.json
|   +-- metadata_extraction.json
|   +-- query_metadata_extraction.json
|
+-- src/
    +-- __init__.py
    |
    +-- components/
        +-- __init__.py
        +-- query_categorization.py
        +-- query_metadata_extraction.py
        +-- metadata_extraction.py
        +-- metadata_filtering.py
```

The project is being developed as a collection of independent components rather than a single monolithic RAG script.

---

# Current Components

### Query Categorization

Determines whether a query is:

```text
fact
detailed
general
```

and produces a confidence value.

---

### Query Metadata Extraction

Extracts:

```text
document type
organizations
locations
dates
department
topics
```

while remaining conservative about hard constraints.

---

### Document Metadata Extraction

Processes source documents and generates structured metadata.

The process supports incremental processing so unchanged documents do not need to be repeatedly processed.

---

### Metadata Filtering

Reduces the candidate document space using reliable metadata constraints.

It can also return retrieval signals that are intentionally left for the retrieval layer.

---

# Configuration Philosophy

The components are designed to avoid embedding project-specific assumptions wherever possible.

For example, the metadata filtering component supports configurable:

```python
organization_aliases
document_type_aliases
metadata_path
relaxation_order
```

This means domain-specific behavior can be supplied by the application rather than hard-coded into the filtering logic.

For example:

```python
MetadataFilterConfig(
    organization_aliases={
        "Tata Power": {
            "Tata Power Delhi Distribution Limited"
        }
    }
)
```

The same component can therefore be adapted to a different corpus without rewriting its core filtering logic.

---

# Observability

The project is designed with observability in mind rather than treating it as an afterthought.

The LLM-based components are traced through LangSmith.

The architecture also exposes structured information such as:

```text
query category
classifier confidence
query metadata
applied metadata filters
candidate count
retrieval signals
relaxed filters
fallback usage
retrieved chunks
reranked results
```

This is important in a production system because an answer being wrong is not enough information.

The system needs to make it possible to investigate:

```text
Was the query classified incorrectly?

Was metadata extracted incorrectly?

Did metadata filtering eliminate the correct document?

Did the retriever fail?

Did the reranker choose the wrong result?

Did generation misinterpret the retrieved context?
```

The modular architecture makes those failure points distinguishable.

---

# Production Focus

This project is explicitly being developed as a **production RAG system**, not merely as a proof-of-concept chatbot.

Production readiness here means designing for:

```text
Modularity
Reliability
Observability
Testability
Replaceability
Fallback behavior
Retrieval quality
Latency awareness
Cost awareness
Evaluation
```

The architecture therefore avoids coupling the entire application to one retrieval method or one model.

A retrieval strategy can be changed independently.

A query classifier can be replaced independently.

A reranker can be changed independently.

Metadata filtering can be improved without rewriting generation.

This makes the system easier to evolve as the corpus, requirements, and models change.

---

# Failure Handling

The system is designed to avoid silently failing.

### Missing metadata

A document with incomplete metadata should not automatically disappear from the corpus.

### Overly restrictive filters

The filtering layer can relax hard constraints when the strict intersection produces no candidates.

### No candidates after relaxation

The filtering component can return an empty candidate set rather than silently pretending that an unrelated document is relevant.

The downstream system can then decide how it wants to handle that case.

### Retrieval failure

The architecture leaves room for retrieval-level fallback strategies rather than making one retrieval algorithm a single point of failure.

---

# What This Project Is Trying to Prove

The central idea is not:

> "Can an LLM answer questions from documents?"

A basic RAG system already demonstrates that.

The deeper question is:

> **Can an adaptive retrieval architecture retrieve better evidence by selecting the retrieval strategy according to the user's information need while reducing unnecessary search through metadata filtering?**

This project is therefore structured around experimentation and evaluation.

The architecture can ultimately be compared against simpler approaches such as a single retrieval strategy.

The important outcome is not simply whether the system produces an answer.

The important outcome is whether the architecture improves:

```text
retrieval quality
answer quality
context relevance
latency
resource usage
```

without introducing unnecessary complexity.

---

# Development Roadmap

The current system has reached the end of the metadata-processing stage.

```text
[COMPLETED]
Document metadata extraction
Query classification
Query metadata extraction
Metadata filtering

        |
        v

[NEXT]
Chunking / indexing layer
BM25 fact retriever
Retrieval interface
BM25 evaluation

        |
        v

Parent-child detailed retriever
Parent-child evaluation

        |
        v

Multi-query general retriever
Multi-query evaluation

        |
        v

Reranking layer

        |
        v

Adaptive retrieval router

        |
        v

End-to-end RAG

        |
        v

Retrieval evaluation
Answer evaluation
Failure analysis

        |
        v

Production hardening
Performance optimization
Observability
```

---

# Long-Term Architecture

The final system is intended to converge toward:

```text
                           USER
                             |
                             v
                      Query Processing
                             |
              +--------------+--------------+
              |                             |
       Query Classification          Query Metadata
              |                     Extraction
              +--------------+--------------+
                             |
                             v
                    Metadata Filtering
                             |
                             v
                    Candidate Documents
                             |
                             v
                     Adaptive Router
                             |
             +---------------+---------------+
             |               |               |
             v               v               v
           BM25        Parent-Child      Multi-Query
             |               |               |
             +---------------+---------------+
                             |
                             v
                          Reranker
                             |
                             v
                       Context Builder
                             |
                             v
                            LLM
                             |
                             v
                          Answer
                             |
                             v
                     Observability
                     + Evaluation
```

The core principle remains:

```text
Understand the query
        |
        v
Reduce the search space
        |
        v
Choose the right retrieval strategy
        |
        v
Retrieve broadly enough
        |
        v
Rerank precisely
        |
        v
Generate from evidence
        |
        v
Measure the system
```

---

# Status

This project is currently in the **retrieval architecture development stage**.

The query-understanding and metadata-filtering foundation is implemented.

The next major milestone is the implementation and evaluation of the specialized retrieval layer, beginning with **BM25-based fact retrieval**.

---

# Objective

Build a modular, adaptive, observable, and measurable **production RAG system** that treats retrieval as an engineering problem rather than simply attaching an LLM to a vector database.
