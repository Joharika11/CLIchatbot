
import os
import time
import random

from dotenv import load_dotenv
from openai import (
    OpenAI,
    APIConnectionError,
    APITimeoutError,
    RateLimitError,
    InternalServerError,
    APIStatusError,
    APIError,
)

# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

load_dotenv()

MAX_RETRIES = 3
BASE_DELAY = 1.0
MAX_DELAY = 20.0
REQUEST_TIMEOUT = 30.0
MAX_OUTPUT_TOKENS = 500

SYSTEM_PROMPT = (
    "You are a helpful assistant. "
    "Give clear, accurate, and concise answers."
)

# Retry only failures that may be temporary.
RETRYABLE_ERRORS = (
    RateLimitError,
    APITimeoutError,
    APIConnectionError,
    InternalServerError,
)

# --------------------------------------------------
# 2. PROVIDER CONFIGURATION
# --------------------------------------------------

def create_providers():
    """Create clients for the primary and fallback providers."""

    gemini_key = os.getenv("GEMINI_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    if not gemini_key:
        raise ValueError("GEMINI_API_KEY is missing from .env")

    if not openai_key:
        raise ValueError("OPENAI_API_KEY is missing from .env")

    return [
        {
            "name": "gemini",
            "model": "gemini-2.5-flash",
            "client": OpenAI(
                api_key=gemini_key,
                base_url=(
                    "https://generativelanguage.googleapis.com/"
                    "v1beta/openai/"
                ),
                max_retries=0,
                timeout=REQUEST_TIMEOUT,
            ),
        },
        {
            "name": "openai",
            "model": "gpt-4o-mini",
            "client": OpenAI(
                api_key=openai_key,
                max_retries=0,
                timeout=REQUEST_TIMEOUT,
            ),
        },
    ]


PROVIDERS = create_providers()

# --------------------------------------------------
# 3. RETRY WITH EXPONENTIAL BACKOFF
# --------------------------------------------------

def call_with_retries(provider, messages):
    """
    Stream a response from one provider.

    Retry temporary API failures. If retries are exhausted,
    raise the error so the caller can try another provider.
    """

    for attempt in range(MAX_RETRIES + 1):
        try:
            return provider["client"].chat.completions.create(
                model=provider["model"],
                messages=messages,
                temperature=0.7,
                max_tokens=MAX_OUTPUT_TOKENS,
                stream=True,
                stream_options={"include_usage": True},
            )

        except RETRYABLE_ERRORS as error:
            if attempt == MAX_RETRIES:
                raise

            delay = min(
                BASE_DELAY * (2 ** attempt),
                MAX_DELAY,
            )

            # Add jitter to spread out simultaneous retries.
            delay += random.uniform(0, delay * 0.25)

            print(
                f"\n[{provider['name']}] "
                f"{type(error).__name__}. "
                f"Retry {attempt + 1}/{MAX_RETRIES} "
                f"in {delay:.1f}s..."
            )

            time.sleep(delay)

# --------------------------------------------------
# 4. PROVIDER FALLBACK + STREAMING
# --------------------------------------------------

def chat(messages):
    """
    Try providers in priority order.

    A failed stream may already have printed partial text.
    In that case, restart the next provider on a new line.
    """

    last_error = None

    for provider in PROVIDERS:
        print(f"\nTrying provider: {provider['name']}")

        try:
            stream = call_with_retries(provider, messages)

            answer_parts = []
            usage = None

            print("\nAssistant: ", end="", flush=True)

            # A stream yields response chunks as they arrive.
            for chunk in stream:
                if not chunk.choices:
                    # Usage-only chunks can have no choices.
                    if chunk.usage is not None:
                        usage = chunk.usage
                    continue

                delta = chunk.choices[0].delta
                content = delta.content

                if content:
                    print(content, end="", flush=True)
                    answer_parts.append(content)

                # The final chunk may contain token usage.
                if chunk.usage is not None:
                    usage = chunk.usage

            # Only report success after the stream completes.
            answer = "".join(answer_parts)

            print(f"\n\nAnswered by: {provider['name']}")

            if usage is not None:
                print("\n--- Token Usage ---")
                print(f"Prompt tokens:     {usage.prompt_tokens}")
                print(
                    f"Completion tokens: "
                    f"{usage.completion_tokens}"
                )
                print(f"Total tokens:      {usage.total_tokens}")
            else:
                print(
                    "\nToken usage unavailable for this response."
                )

            return answer, provider["name"]

        except Exception as error:
            last_error = error

            print(
                f"\n[{provider['name']}] "
                f"Provider failed: {type(error).__name__}: "
                f"{error}"
            )

            print("Switching to the next provider...")

    raise RuntimeError(
        "All configured providers failed."
    ) from last_error

# --------------------------------------------------
# 5. INTERACTIVE CLI LOOP
# --------------------------------------------------

def main():
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        }
    ]

    print("=" * 48)
    print("          CLI CHATBOT WITH FALLBACK")
    print("=" * 48)
    print("Primary: Gemini")
    print("Fallback: OpenAI")
    print("Type 'exit' or 'quit' to end the conversation.")
    print("Type 'clear' to reset conversation history.")

    while True:
        try:
            user_input = input("\nYou: ").strip()

            if not user_input:
                print("Please enter a message.")
                continue

            if user_input.lower() in {"exit", "quit"}:
                print("Goodbye!")
                break

            if user_input.lower() == "clear":
                messages = [
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT,
                    }
                ]
                print("Conversation history cleared.")
                continue

            # Add the new message to the conversation context.
            messages.append(
                {
                    "role": "user",
                    "content": user_input,
                }
            )

            try:
                answer, used_provider = chat(messages)

                # Store the answer only after a complete response.
                messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                    }
                )

            except Exception as error:
                # The user message stays in history, but no
                # incomplete assistant response is added.
                print(f"\nChatbot error: {error}")
                print("Please try again.")

        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break


if __name__ == "__main__":
    main()
