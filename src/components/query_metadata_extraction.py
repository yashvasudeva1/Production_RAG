from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from langsmith import traceable
from pydantic import BaseModel, ConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / ".env"
PROMPT_FILE = PROJECT_ROOT / "prompts" / "query_metadata_extraction.json"

load_dotenv(dotenv_path=ENV_FILE, override=False)

MODEL_NAME = "Qwen/Qwen3-8B"
PROVIDER = "nscale"


class QueryMetadata(BaseModel):
    model_config = ConfigDict(extra="ignore")

    document_type: str
    organizations: list[str]
    locations: list[str]
    dates: list[str]
    department: str
    topics: list[str]


class QueryMetadataExtractor:
    def __init__(
        self,
        model_name: str = MODEL_NAME,
        provider: str = PROVIDER,
        prompt_file: str | Path = PROMPT_FILE,
    ) -> None:
        token = os.getenv("HF_TOKEN")
        if not token:
            raise EnvironmentError(
                f"HF_TOKEN is not set. Expected in {ENV_FILE}"
            )

        self.model_name = model_name
        self.provider = provider
        self.prompt_file = Path(prompt_file)
        self.system_prompt = self._load_prompt()

        self.client = InferenceClient(
            provider=self.provider,
            api_key=token,
        )

    def _load_prompt(self) -> str:
        if not self.prompt_file.exists():
            raise FileNotFoundError(
                f"Prompt file not found: {self.prompt_file}"
            )

        data = json.loads(
            self.prompt_file.read_text(encoding="utf-8")
        )

        if isinstance(data.get("system"), str):
            return data["system"]

        section = data.get("query_metadata_extraction")
        if isinstance(section, dict) and isinstance(section.get("system"), str):
            return section["system"]

        raise ValueError(
            "Prompt JSON must contain 'system' or "
            "'query_metadata_extraction.system'."
        )

    @staticmethod
    def _schema() -> dict[str, Any]:
        schema = QueryMetadata.model_json_schema()
        schema["required"] = list(schema["properties"].keys())
        schema["additionalProperties"] = False

        return {
            "name": "query_metadata",
            "description": "Explicit metadata constraints from a user query.",
            "schema": schema,
            "strict": True,
        }

    @traceable(
        name="query_metadata_extraction",
        run_type="llm",
        tags=["rag", "query-metadata", "huggingface", "qwen3-8b", "nscale"],
    )
    def extract(self, query: str) -> QueryMetadata:
        query = self._validate_query(query)

        base_kwargs = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": self.system_prompt,
                },
                {
                    "role": "user",
                    "content": (
                        "Extract only explicit metadata constraints from this query.\n\n"
                        f"User query:\n{query}\n\n"
                        "Do not answer the query. Return only JSON."
                    ),
                },
            ],
            "temperature": 0.0,
            "max_tokens": 800,
            "extra_body": {
                "chat_template_kwargs": {
                    "enable_thinking": False,
                }
            },
        }

        try:
            response = self.client.chat.completions.create(
                **base_kwargs,
                response_format={
                    "type": "json_schema",
                    "json_schema": self._schema(),
                },
            )
            return self._parse_response(response)

        except Exception:
            response = self.client.chat.completions.create(
                **base_kwargs,
                response_format={
                    "type": "json_object",
                },
            )
            return self._parse_response(response)

    @staticmethod
    def _parse_response(response: Any) -> QueryMetadata:
        if not getattr(response, "choices", None):
            raise RuntimeError("Model returned no choices.")

        content = response.choices[0].message.content

        if isinstance(content, list):
            content = "".join(
                item.get("text", "")
                for item in content
                if isinstance(item, dict)
                and isinstance(item.get("text"), str)
            )

        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("Model returned an empty response.")

        content = content.strip()

        if content.startswith("```"):
            content = re.sub(
                r"^```(?:json)?\s*",
                "",
                content,
                flags=re.IGNORECASE,
            )
            content = re.sub(r"\s*```$", "", content)

        start = content.find("{")
        end = content.rfind("}")
        if start >= 0 and end >= start:
            content = content[start:end + 1]

        payload = json.loads(content)
        result = QueryMetadata.model_validate(payload)

        return QueryMetadata(
            document_type=clean_scalar(result.document_type),
            organizations=clean_list(result.organizations),
            locations=clean_list(result.locations),
            dates=clean_list(result.dates),
            department=clean_scalar(result.department),
            topics=clean_topics(result.topics),
        )

    @staticmethod
    def _validate_query(query: str) -> str:
        if not isinstance(query, str):
            raise TypeError("Query must be a string.")

        query = query.strip()

        if not query:
            raise ValueError("Query cannot be empty.")

        return query


def clean_scalar(value: str) -> str:
    if not isinstance(value, str):
        return ""

    value = value.strip()

    invalid = {
        "",
        "unknown",
        "unspecified",
        "not specified",
        "not mentioned",
        "not stated",
        "n/a",
        "na",
        "none",
        "null",
    }

    return "" if value.casefold() in invalid else value


def clean_list(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()

    for value in values:
        if not isinstance(value, str):
            continue

        value = value.strip()
        if not value:
            continue

        key = value.casefold()
        if key in seen:
            continue

        seen.add(key)
        result.append(value)

    return result


GENERIC_TOPICS = {
    "experience",
    "information",
    "details",
    "overview",
    "documents",
    "document",
    "things",
    "stuff",
    "background",
    "knowledge",
    "content",
    "explain",
    "explanation",
    "summary",
    "summarize",
    "discuss",
    "discussion",
}


def clean_topics(values: list[str]) -> list[str]:
    values = clean_list(values)

    return [
        value
        for value in values
        if value.casefold() not in GENERIC_TOPICS
        and len(value) >= 3
    ]


_default_extractor: QueryMetadataExtractor | None = None


def get_query_metadata_extractor() -> QueryMetadataExtractor:
    global _default_extractor

    if _default_extractor is None:
        _default_extractor = QueryMetadataExtractor()

    return _default_extractor


def extract_query_metadata(query: str) -> QueryMetadata:
    return get_query_metadata_extractor().extract(query)


if __name__ == "__main__":
    extractor = QueryMetadataExtractor()

    test_queries = [
        "What did Google researchers propose in the 2017 Transformer paper?",
        "Show me Yash Vasudeva's experience at Tata Power.",
        "Tell me about machine learning.",
        "Which research papers discuss machine translation?",
    ]

    for query in test_queries:
        print("\n" + "=" * 70)
        print(query)
        print("=" * 70)

        try:
            result = extractor.extract(query)
            print(json.dumps(result.model_dump(), indent=2, ensure_ascii=False))
        except Exception as exc:
            print(f"Error: {exc}")
