from typing import Mapping, Optional, Set, Coroutine, Tuple

from asyncio import FIRST_COMPLETED, CancelledError, Queue
from asyncio import wait, gather, create_task


def getattr_dot(obj: any, index: str):
    parts = index.split('.')
    for part in parts:
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj


async def cancel_all(tasks: set):
    tasks.discard(None)
    for task in tasks:
        task.cancel()

    try:
        await gather(*tasks)
    except CancelledError:
        pass


async def race(tasks: Set[Coroutine]):
    # Coroutines must be wrapped by asyncio Tasks:
    #
    # "The explicit passing of coroutine objects to asyncio.wait() is deprecated
    # since Python 3.8, and scheduled for removal in Python 3.11."
    tasks = map(create_task, tasks)
    done, pending = await wait(tasks, return_when=FIRST_COMPLETED)

    await cancel_all(pending)
    results = await gather(*done)

    return results[0]


async def close_queue(queue: Queue):
    while not queue.empty():
        queue.get_nowait()
        queue.task_done()

    return await queue.join()


def _get_limit(headers: Mapping[str, str], header: str) -> Optional[int]:
    value = headers.get(header)
    if value is None:
        return None

    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def get_limits(headers: Mapping[str, str]) -> Tuple[Optional[int], Optional[int]]:
    return (
        _get_limit(headers, "x-ratelimit-limit-requests"),
        _get_limit(headers, "x-ratelimit-limit-tokens"),
    )


EXAMPLE = """
    # Pass API key, or set env var `OPENAI_API_KEY`
    batch = Batch(api_key="your_openai_api_key")

    # Add chat completion requests
    for i in range(5):
        await batch.add(
            "chat.completions.create", 
            model="gpt-3.5-turbo", 
            messages=[{"role": "user", "content": "say hello"}]
        )

    # Run the batch processing for chat completions
    chat = await batch.run()

    # Add embedding requests
    for i in range(5):
        await batch.add(
            "embeddings.create", 
            model="text-embedding-3-large", 
            input="hello world"
        )

    # Run the batch processing for embeddings
    embeddings = await batch.run()
"""
