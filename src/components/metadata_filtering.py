from __future__ import annotations

import json
import re
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from pydantic import BaseModel, Field

from .query_metadata_extraction import QueryMetadata, extract_query_metadata


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_METADATA_PATH = (
    Path(__file__).resolve().parents[2] / "metadata" / "documents.json"
)

# These fields are suitable for conservative hard filtering because they
# describe document-level properties rather than the user's information need.
HARD_FILTER_FIELDS = {
    "document_type",
    "organizations",
    "locations",
    "dates",
    "department",
}

# These fields should normally be passed to a retrieval component rather than
# used to eliminate documents here.
SOFT_RETRIEVAL_FIELDS = {
    "topics",
}


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class FilterResult(BaseModel):
    document_ids: list[str] = Field(default_factory=list)
    matched_count: int = 0
    total_documents: int = 0

    # Only constraints that actually affected the candidate set.
    applied_filters: dict[str, Any] = Field(default_factory=dict)

    # Signals that should be forwarded to the retrieval layer.
    retrieval_signals: dict[str, Any] = Field(default_factory=dict)

    # Human-readable names of hard filters relaxed during fallback.
    relaxed_filters: list[str] = Field(default_factory=list)

    used_fallback: bool = False


@dataclass
class MetadataFilterConfig:
    metadata_path: str | Path = DEFAULT_METADATA_PATH

    # Order in which hard constraints are relaxed when the exact intersection
    # produces zero candidates.
    relaxation_order: tuple[str, ...] = (
        "dates",
        "locations",
        "department",
        "organizations",
        "document_type",
    )

    # Organization aliases can be supplied by the application/domain.
    # Example:
    # {"Tata Power": {"Tata Power Delhi Distribution Limited"}}
    organization_aliases: Mapping[str, set[str]] = field(default_factory=dict)

    # Optional document-type normalization map.
    # Example:
    # {"paper": "research_paper", "research paper": "research_paper"}
    document_type_aliases: Mapping[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

def _normalize(value: Any) -> str:
    """Normalize a scalar for case-insensitive comparison."""
    if value is None:
        return ""

    text = str(value).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def _normalize_date(value: Any) -> str:
    """Normalize dates while preserving useful year/range information."""
    text = _normalize(value)
    text = text.replace("–", "-").replace("—", "-")
    return text


def _clean_list(values: Iterable[Any] | None) -> list[str]:
    result: list[str] = []

    if not values:
        return result

    for value in values:
        normalized = str(value).strip()
        if not normalized:
            continue

        if normalized not in result:
            result.append(normalized)

    return result


def _as_list(value: Any) -> list[str]:
    """Convert a metadata field into a list without losing scalar values."""
    if value is None:
        return []

    if isinstance(value, (list, tuple, set)):
        return _clean_list(value)

    text = str(value).strip()
    return [text] if text else []


def _document_id(document: Mapping[str, Any]) -> str | None:
    value = document.get("document_id")

    if value:
        return str(value)

    # Support applications that use another common ID field.
    for key in ("id", "doc_id"):
        value = document.get(key)
        if value:
            return str(value)

    return None


# ---------------------------------------------------------------------------
# Field matching
# ---------------------------------------------------------------------------

def _canonical_document_type(
    value: Any,
    aliases: Mapping[str, str],
) -> str:
    normalized = _normalize(value)

    if not normalized:
        return ""

    return _normalize(aliases.get(normalized, normalized))


def _organization_matches(
    requested: str,
    document_values: Sequence[str],
    aliases: Mapping[str, set[str]],
) -> bool:
    requested_norm = _normalize(requested)

    if not requested_norm:
        return False

    accepted = {requested_norm}

    # Expand aliases in both directions:
    # requested -> canonical/known variants.
    for key, values in aliases.items():
        key_norm = _normalize(key)
        value_norms = {_normalize(v) for v in values}

        if requested_norm == key_norm or requested_norm in value_norms:
            accepted.add(key_norm)
            accepted.update(value_norms)

    normalized_document_values = {
        _normalize(value)
        for value in document_values
        if _normalize(value)
    }

    # Exact normalized match first.
    if accepted & normalized_document_values:
        return True

    # Conservative containment for names such as:
    # "Tata Power" vs "Tata Power Delhi Distribution Limited".
    for requested_variant in accepted:
        for document_value in normalized_document_values:
            if (
                requested_variant in document_value
                or document_value in requested_variant
            ):
                return True

    return False


def _scalar_matches(
    requested: Any,
    actual: Any,
    *,
    field_name: str,
    config: MetadataFilterConfig,
) -> bool:
    requested_norm = _normalize(requested)
    actual_norm = _normalize(actual)

    if not requested_norm:
        return True

    if not actual_norm:
        return False

    if field_name == "document_type":
        requested_norm = _canonical_document_type(
            requested_norm,
            config.document_type_aliases,
        )
        actual_norm = _canonical_document_type(
            actual_norm,
            config.document_type_aliases,
        )
        return requested_norm == actual_norm

    if field_name == "dates":
        return _date_matches(requested_norm, actual_norm)

    return requested_norm == actual_norm


def _date_matches(requested: str, actual: str) -> bool:
    """
    Match dates conservatively but support useful cases such as:

        query: 2017
        document: 2017

        query: 2024
        document: 2024-2028

    We avoid fuzzy matching unrelated dates.
    """
    if requested == actual:
        return True

    requested_years = set(re.findall(r"\b(?:19|20)\d{2}\b", requested))
    actual_years = set(re.findall(r"\b(?:19|20)\d{2}\b", actual))

    if requested_years & actual_years:
        return True

    return False


# ---------------------------------------------------------------------------
# Document extraction
# ---------------------------------------------------------------------------

def _metadata_for_document(document: Mapping[str, Any]) -> Mapping[str, Any]:
    """
    Supports the current structure:

        {
            "document_id": "...",
            "extracted_metadata": {
                "document_type": "...",
                ...
            }
        }

    and also a flatter structure:

        {
            "document_id": "...",
            "document_type": "...",
            ...
        }
    """
    nested = document.get("extracted_metadata")

    if isinstance(nested, Mapping):
        return nested

    return document


def _field_values(
    document: Mapping[str, Any],
    field_name: str,
) -> list[str]:
    metadata = _metadata_for_document(document)

    return _as_list(metadata.get(field_name))


# ---------------------------------------------------------------------------
# Hard filter evaluation
# ---------------------------------------------------------------------------

def _matches_hard_filter(
    document: Mapping[str, Any],
    field_name: str,
    requested_value: Any,
    config: MetadataFilterConfig,
) -> bool:
    actual_values = _field_values(document, field_name)

    if field_name == "organizations":
        requested_values = _as_list(requested_value)

        if not requested_values:
            return True

        return all(
            any(
                _organization_matches(
                    requested_item,
                    actual_values,
                    config.organization_aliases,
                )
                for actual_values in [actual_values]
            )
            for requested_item in requested_values
        )

    if field_name in {"locations", "dates"}:
        requested_values = _as_list(requested_value)

        if not requested_values:
            return True

        for requested_item in requested_values:
            if not any(
                _scalar_matches(
                    requested_item,
                    actual_item,
                    field_name=field_name,
                    config=config,
                )
                for actual_item in actual_values
            ):
                return False

        return True

    # Scalar fields such as document_type and department.
    return _scalar_matches(
        requested_value,
        actual_values[0] if actual_values else "",
        field_name=field_name,
        config=config,
    )


def _filter_documents(
    documents: Sequence[Mapping[str, Any]],
    filters: Mapping[str, Any],
    config: MetadataFilterConfig,
) -> list[Mapping[str, Any]]:
    candidates = list(documents)

    for field_name, requested_value in filters.items():
        if field_name not in HARD_FILTER_FIELDS:
            continue

        if requested_value in ("", None, [], ()):
            continue

        candidates = [
            document
            for document in candidates
            if _matches_hard_filter(
                document,
                field_name,
                requested_value,
                config,
            )
        ]

        if not candidates:
            break

    return candidates


# ---------------------------------------------------------------------------
# Candidate loading
# ---------------------------------------------------------------------------

def _load_metadata_records(path: str | Path) -> list[dict[str, Any]]:
    metadata_path = Path(path)

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {metadata_path}"
        )

    with metadata_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if isinstance(data, list):
        records = data

    elif isinstance(data, Mapping):
        # Support common wrapper shapes.
        for key in ("documents", "records", "items"):
            if isinstance(data.get(key), list):
                records = data[key]
                break
        else:
            raise ValueError(
                "Metadata JSON must be a list or contain a list under "
                "'documents', 'records', or 'items'."
            )
    else:
        raise ValueError("Metadata JSON must contain document records.")

    valid_records: list[dict[str, Any]] = []

    for record in records:
        if isinstance(record, Mapping) and _document_id(record):
            valid_records.append(dict(record))

    return valid_records


# ---------------------------------------------------------------------------
# Query metadata handling
# ---------------------------------------------------------------------------

def _query_metadata_to_dict(
    metadata: QueryMetadata | Mapping[str, Any],
) -> dict[str, Any]:
    if isinstance(metadata, BaseModel):
        data = metadata.model_dump()
    else:
        data = dict(metadata)

    # Normalize the fields we expect from the query metadata extractor.
    normalized = {
        "document_type": data.get("document_type", ""),
        "organizations": _clean_list(data.get("organizations", [])),
        "locations": _clean_list(data.get("locations", [])),
        "dates": _clean_list(data.get("dates", [])),
        "department": data.get("department", ""),
        "topics": _clean_list(data.get("topics", [])),
    }

    return normalized


def _extract_hard_filters(query_metadata: Mapping[str, Any]) -> dict[str, Any]:
    filters: dict[str, Any] = {}

    for field_name in HARD_FILTER_FIELDS:
        value = query_metadata.get(field_name)

        if value not in ("", None, [], ()):
            filters[field_name] = value

    return filters


def _extract_retrieval_signals(
    query_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    signals: dict[str, Any] = {}

    for field_name in SOFT_RETRIEVAL_FIELDS:
        value = query_metadata.get(field_name)

        if value not in ("", None, [], ()):
            signals[field_name] = value

    return signals


# ---------------------------------------------------------------------------
# Relaxation / fallback
# ---------------------------------------------------------------------------

def _relax_filters(
    documents: Sequence[Mapping[str, Any]],
    filters: Mapping[str, Any],
    config: MetadataFilterConfig,
) -> tuple[list[Mapping[str, Any]], list[str]]:
    """
    Gradually relax hard constraints until candidates are found.

    Example:
        document_type + organization + date
                     ↓ no match
        organization + date
                     ↓ no match
        organization
                     ↓ candidates
    """
    active_filters = dict(filters)
    relaxed: list[str] = []

    exact_candidates = _filter_documents(
        documents,
        active_filters,
        config,
    )

    if exact_candidates:
        return exact_candidates, relaxed

    for field_name in config.relaxation_order:
        if field_name not in active_filters:
            continue

        active_filters.pop(field_name)
        relaxed.append(field_name)

        candidates = _filter_documents(
            documents,
            active_filters,
            config,
        )

        if candidates:
            return candidates, relaxed

    # No hard-filtered candidates at all.
    # Returning every document is intentionally NOT the default because that
    # would silently turn a contradictory explicit constraint into a corpus-wide
    # search. The caller can choose to handle this case differently.
    return [], relaxed


# ---------------------------------------------------------------------------
# Public component
# ---------------------------------------------------------------------------

class MetadataFilter:
    """
    General-purpose metadata candidate filtering component.

    Responsibility:
        Query
          -> query metadata
          -> conservative hard metadata filtering
          -> candidate document IDs
          -> retrieval signals for the downstream retriever

    It intentionally does NOT perform:
        - vector search
        - keyword search
        - reranking
        - chunk retrieval
        - answer generation
    """

    def __init__(
        self,
        config: MetadataFilterConfig | None = None,
    ) -> None:
        self.config = config or MetadataFilterConfig()
        self.documents = _load_metadata_records(
            self.config.metadata_path
        )

    def reload(self) -> None:
        """Reload document metadata from disk."""
        self.documents = _load_metadata_records(
            self.config.metadata_path
        )

    def filter(
        self,
        *,
        query: str | None = None,
        query_metadata: QueryMetadata | Mapping[str, Any] | None = None,
    ) -> FilterResult:
        """
        Filter the corpus using either:

            filter(query="...")
        or:
            filter(query_metadata=...)

        Exactly one must be provided.
        """
        if (query is None) == (query_metadata is None):
            raise ValueError(
                "Provide exactly one of 'query' or 'query_metadata'."
            )

        if query_metadata is None:
            query_metadata = extract_query_metadata(query or "")

        metadata = _query_metadata_to_dict(query_metadata)

        hard_filters = _extract_hard_filters(metadata)
        retrieval_signals = _extract_retrieval_signals(metadata)

        total_documents = len(self.documents)

        # No hard constraints: all valid documents remain candidates.
        if not hard_filters:
            return FilterResult(
                document_ids=[
                    _document_id(document)
                    for document in self.documents
                    if _document_id(document)
                ],
                matched_count=total_documents,
                total_documents=total_documents,
                applied_filters={},
                retrieval_signals=retrieval_signals,
                relaxed_filters=[],
                used_fallback=False,
            )

        candidates, relaxed_filters = _relax_filters(
            self.documents,
            hard_filters,
            self.config,
        )

        used_fallback = bool(relaxed_filters)

        # Important design choice:
        # if no candidates exist even after safe relaxation, return no IDs.
        # The downstream retriever can then choose its own broader fallback.
        candidate_ids = [
            _document_id(document)
            for document in candidates
            if _document_id(document)
        ]

        # Preserve only filters that remained active.
        active_filters = dict(hard_filters)
        for field_name in relaxed_filters:
            active_filters.pop(field_name, None)

        return FilterResult(
            document_ids=candidate_ids,
            matched_count=len(candidate_ids),
            total_documents=total_documents,
            applied_filters=active_filters,
            retrieval_signals=retrieval_signals,
            relaxed_filters=relaxed_filters,
            used_fallback=used_fallback,
        )

    def filter_with_candidates(
        self,
        *,
        query: str | None = None,
        query_metadata: QueryMetadata | Mapping[str, Any] | None = None,
    ) -> tuple[FilterResult, list[dict[str, Any]]]:
        """
        Convenience method returning both the structured result and the
        full candidate document metadata records.
        """
        result = self.filter(
            query=query,
            query_metadata=query_metadata,
        )

        id_set = set(result.document_ids)

        candidates = [
            document
            for document in self.documents
            if _document_id(document) in id_set
        ]

        return result, candidates


# ---------------------------------------------------------------------------
# Functional API
# ---------------------------------------------------------------------------

def filter_documents(
    *,
    query: str | None = None,
    query_metadata: QueryMetadata | Mapping[str, Any] | None = None,
    metadata_path: str | Path = DEFAULT_METADATA_PATH,
    organization_aliases: Mapping[str, set[str]] | None = None,
    document_type_aliases: Mapping[str, str] | None = None,
) -> FilterResult:
    """
    Functional entry point for applications that do not need a long-lived
    MetadataFilter instance.
    """
    config = MetadataFilterConfig(
        metadata_path=metadata_path,
        organization_aliases=organization_aliases or {},
        document_type_aliases=document_type_aliases or {},
    )

    component = MetadataFilter(config)

    return component.filter(
        query=query,
        query_metadata=query_metadata,
    )


# ---------------------------------------------------------------------------
# Example
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    config = MetadataFilterConfig(
        organization_aliases={
            "Tata Power": {
                "Tata Power Delhi Distribution Limited",
            }
        },
        document_type_aliases={
            "research paper": "research_paper",
            "paper": "research_paper",
        },
    )

    metadata_filter = MetadataFilter(config)

    examples = [
        "What did Google researchers propose in the 2017 Transformer paper?",
        "Show me Yash Vasudeva's experience at Tata Power.",
        "Tell me about machine learning.",
        "Which research papers discuss machine translation?",
    ]

    for query in examples:
        print("\n" + "=" * 70)
        print(f"QUERY: {query}")
        print("=" * 70)

        result = metadata_filter.filter(query=query)

        print(result.model_dump_json(indent=2))
