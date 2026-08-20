from pathlib import Path
from types import SimpleNamespace

from oaib import Auto
from oaib.utils import get_limits


class FakeTranscription:
    def __init__(self, text, usage):
        self.text = text
        self.usage = usage

    def model_dump(self):
        return {"text": self.text}


class FakeRawResponse:
    def __init__(self, headers, parsed):
        self.headers = headers
        self.parsed = parsed

    def parse(self):
        return self.parsed


class FakeTranscriptions:
    def __init__(self, responses):
        self.responses = iter(responses)

    async def create(self, **kwargs):
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response


class FakeClient:
    def __init__(self, responses):
        transcriptions = FakeTranscriptions(responses)
        self.with_raw_response = SimpleNamespace(
            audio=SimpleNamespace(transcriptions=transcriptions)
        )


def audio_response(text, headers=None):
    usage = SimpleNamespace(type="duration", seconds=1.0)
    transcription = FakeTranscription(text, usage)
    response_headers = headers or {"x-ratelimit-limit-requests": "600"}
    return FakeRawResponse(response_headers, transcription)


def test_get_limits_allows_missing_headers():
    headers = {"x-ratelimit-limit-requests": "120"}

    assert get_limits(headers) == (120, None)
    assert get_limits({}) == (None, None)
    assert get_limits({"x-ratelimit-limit-requests": "invalid"}) == (None, None)


async def test_auto_processes_transcription_without_tpm_header():
    batch = Auto(api_key="test-key", workers=2, loglevel=0)
    batch.client = FakeClient([
        audio_response("first"),
        audio_response("second"),
        audio_response("third"),
    ])

    for index in range(3):
        await batch.add(
            "audio.transcriptions.create",
            metadata={"id": index},
            file=b"audio",
            model="gpt-4o-mini-transcribe-2025-12-15",
        )

    results = await batch.run()

    assert not Path("oaib.txt").exists()
    assert len(results) == 3
    assert batch.rpm == 600
    assert batch.tpm is None
    assert results["error"].isna().all()
    assert results["result"].map(lambda result: result["text"]).tolist() == [
        "first",
        "second",
        "third",
    ]


async def test_auto_keeps_tpm_for_other_models():
    headers = {
        "x-ratelimit-limit-requests": "600",
        "x-ratelimit-limit-tokens": "1000",
    }
    batch = Auto(api_key="test-key", workers=1, loglevel=0)
    batch.client = FakeClient([
        audio_response("first", headers),
        audio_response("second", headers),
    ])

    for index in range(2):
        await batch.add(
            "audio.transcriptions.create",
            metadata={"id": index},
            file=b"audio",
            model="gpt-4o-transcribe",
        )

    await batch.run()

    assert batch.rpm == 600
    assert batch.tpm == 1000


async def test_batch_returns_transcription_failure_row():
    batch = Auto(api_key="test-key", workers=1, loglevel=0)
    batch.client = FakeClient([
        audio_response("success"),
        RuntimeError("transcription failed"),
    ])

    for index in range(2):
        await batch.add(
            "audio.transcriptions.create",
            metadata={"id": index},
            file=b"audio",
            model="gpt-4o-mini-transcribe-2025-12-15",
        )

    results = await batch.run()
    failure = results[results["error"].notna()].iloc[0]

    assert len(results) == 2
    assert failure["id"] == 1
    assert failure["result"] is None
    assert failure["error"] == "RuntimeError: transcription failed"
