import json
import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langsmith import traceable
from pydantic import BaseModel, Field

ROOT_DIR = Path(__file__).resolve().parents[2]
PROMPT_FILE = ROOT_DIR / "prompts" / "query_classification.json"
ENV_FILE = ROOT_DIR / ".env"

load_dotenv(ENV_FILE)


class QueryClassification(BaseModel):
    category: Literal["fact", "detailed", "general"] = Field(
        description="The query category used to select the retrieval strategy."
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score for the classification."
    )


class QueryCategorization:
    def __init__(
        self,
        prompt_file: Path = PROMPT_FILE,
        model_name: str = "gemini-2.5-flash-lite"
    ) -> None:
        if not os.getenv("GEMINI_API_KEY"):
            raise EnvironmentError("GEMINI_API_KEY is not set.")

        self.prompt_file = Path(prompt_file)

        if not self.prompt_file.exists():
            raise FileNotFoundError(
                f"Prompt file not found: {self.prompt_file}"
            )

        self.system_prompt = self._load_prompt()

        self.model = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0,
            max_retries=2
        ).with_structured_output(QueryClassification)

    def _load_prompt(self) -> str:
        try:
            data = json.loads(
                self.prompt_file.read_text(encoding="utf-8")
            )
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSON in {self.prompt_file}: {exc}"
            ) from exc

        if isinstance(data, dict):
            if isinstance(data.get("system"), str):
                return data["system"]

            query_classifier = data.get("query_classifier")
            if isinstance(query_classifier, dict):
                system_prompt = query_classifier.get("system")
                if isinstance(system_prompt, str):
                    return system_prompt

        raise ValueError(
            "query_classification.json must contain either "
            "'system' or 'query_classifier.system'."
        )

    @traceable(
        name="query_categorization",
        run_type="chain",
        tags=["rag", "query-classification"]
    )
    def classify(self, query: str) -> QueryClassification:
        if not isinstance(query, str):
            raise TypeError("Query must be a string.")

        query = query.strip()

        if not query:
            raise ValueError("Query cannot be empty.")

        result = self.model.invoke(
            [
                ("system", self.system_prompt),
                ("human", query)
            ],
            config={
                "metadata": {
                    "component": "query_categorization",
                    "model": "gemini-2.5-flash-lite"
                },
                "tags": [
                    "rag",
                    "query-classification",
                    "gemini-2.5-flash-lite"
                ]
            }
        )

        if not isinstance(result, QueryClassification):
            raise TypeError(
                f"Unexpected output type: {type(result).__name__}"
            )

        return result


_classifier = None


def get_classifier() -> QueryCategorization:
    global _classifier

    if _classifier is None:
        _classifier = QueryCategorization()

    return _classifier


def classify_query(query: str) -> QueryClassification:
    return get_classifier().classify(query)


if __name__ == "__main__":
    classifier = QueryCategorization()

    test_queries = [
        "What was the company's revenue in 2025?",
        "Explain the complete employee onboarding process.",
        "Tell me about the company's AI strategy."
    ]

    for query in test_queries:
        try:
            result = classifier.classify(query)

            print(f"Query: {query}")
            print(f"Category: {result.category}")
            print(f"Confidence: {result.confidence:.2f}")
            print("-" * 60)

        except Exception as exc:
            print(f"Error: {exc}")
