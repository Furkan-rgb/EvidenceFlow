"""Bounded structured-output invocation shared by task capabilities."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping, Sequence

from langchain_core.exceptions import OutputParserException
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage
from pydantic import BaseModel, ValidationError

from app.ai.usage import capture_usage_metadata, clear_usage_metadata
from app.errors import InvalidStructuredOutputError, ModelUnavailableError


async def invoke_structured[StructuredT: BaseModel](
    model: BaseChatModel,
    schema: type[StructuredT],
    messages: Sequence[BaseMessage],
    *,
    capability: str,
    validator: Callable[[StructuredT], StructuredT] | None = None,
) -> StructuredT:
    """Invoke JSON-schema output once, with exactly one bounded repair attempt."""

    structured_model = model.with_structured_output(
        schema, method="json_schema", include_raw=True
    )
    current_messages = list(messages)
    last_error: Exception | None = None
    for attempt in range(2):
        try:
            clear_usage_metadata()
            response = await structured_model.ainvoke(current_messages)
            if isinstance(response, Mapping) and "parsed" in response:
                capture_usage_metadata(response.get("raw"))
                parsing_error = response.get("parsing_error")
                if parsing_error is not None:
                    raise ValueError(f"structured-output parsing failed: {parsing_error}")
                raw = response.get("parsed")
            else:
                raw = response
            result = schema.model_validate(raw)
            return validator(result) if validator else result
        except asyncio.CancelledError:
            raise
        except (OutputParserException, ValidationError, TypeError, ValueError, KeyError) as exc:
            last_error = exc
            if attempt == 0:
                current_messages.append(
                    HumanMessage(
                        content=(
                            "Your previous response did not satisfy the required JSON schema or "
                            "EvidenceFlow invariants. Return one corrected JSON object only. "
                            f"Validation summary: {str(exc)[:1000]}"
                        )
                    )
                )
                continue
        except Exception as exc:
            raise ModelUnavailableError(
                f"The model call for {capability} failed.",
                details={"capability": capability},
            ) from exc

    raise InvalidStructuredOutputError(
        f"The model returned invalid structured output for {capability} after one repair attempt.",
        details={
            "capability": capability,
            "validation_error": type(last_error).__name__ if last_error else "unknown",
        },
    ) from last_error
