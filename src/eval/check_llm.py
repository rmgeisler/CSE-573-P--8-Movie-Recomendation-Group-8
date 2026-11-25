import os
import time
from concurrent.futures import ThreadPoolExecutor
from getpass import getpass

from openai import OpenAI

BASE_URL = "https://openai.rc.asu.edu/v1"
CHAT_MODEL = "qwen35-27b"
EMBED_MODEL = "qwen3-embedding-4b"


def chat(client, content, max_tokens=50):
    return client.chat.completions.create(
        model=CHAT_MODEL,
        reasoning_effort="none",
        messages=[
            {"role": "system", "content": "Return valid JSON only."},
            {"role": "user", "content": content},
        ],
        temperature=0.1,
        max_tokens=max_tokens,
    )


def timed(label, fn):
    t0 = time.time()
    try:
        out = fn()
        print(f"  OK   {label}  ({time.time() - t0:.1f}s)")
        return out
    except Exception as e:
        print(f"  FAIL {label}  ({time.time() - t0:.1f}s): {type(e).__name__}: {str(e)[:300]}")
        return None


def main():
    key = os.environ.get("ASU_API_KEY") or getpass("Enter your ASU API key: ")
    client = OpenAI(api_key=key, base_url=BASE_URL, timeout=300.0, max_retries=1)

    print("\n1. models listed by the server")
    models = timed("models.list", lambda: client.models.list())
    if models is not None:
        print("    " + ", ".join(sorted(m.id for m in models.data)))

    print("\n2. one chat call")
    r = timed("short chat", lambda: chat(client, 'Return {"ranked_ids": [2, 0, 1]} exactly.'))
    if r is not None:
        print("    reply:", (r.choices[0].message.content or "").strip()[:200])

    print("\n3. one embedding call (32 short texts)")
    e = timed("embeddings x32", lambda: client.embeddings.create(model=EMBED_MODEL, input=["a movie about space"] * 32))
    if e is not None:
        print("    embedding size:", len(e.data[0].embedding))

    print("\n4. longer and longer prompts (approx tokens = characters / 4)")
    filler = "The quick brown fox jumps over the lazy dog. "
    for approx_tokens in (8_000, 25_000, 50_000, 90_000):
        text = (filler * (approx_tokens * 4 // len(filler)))
        r = timed(f"prompt ~{approx_tokens:,} tokens",
                  lambda t=text: chat(client, t + '\nIgnore the text above. Return {"ok": true}'))
        if r is None:
            print("    stopping here: the server did not accept this length")
            break
        print(f"    server counted prompt_tokens = {getattr(r.usage, 'prompt_tokens', 'n/a')}")

    print("\n5. four chat calls at the same time")
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=4) as ex:
        results = list(ex.map(lambda i: timed(f"parallel call {i}", lambda: chat(client, 'Return {"ok": true}')), range(4)))
    print(f"  total wall time for 4 parallel calls: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
