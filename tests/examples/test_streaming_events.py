import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from llama_index.core.workflow import (
    Context,
    StartEvent,
    StopEvent,
)

# Assuming llama_demo is importable, adjust if needed
from llama_demo.examples.streaming_events import (
    FirstEvent,
    MyWorkflow,
    ProgressEvent,
    SecondEvent,
)


# Mock response class for OpenAI stream
class MockResponseChunk:
    def __init__(self, delta: str):
        self.delta = delta

    def __str__(self) -> str:
        # Mock the final response aggregation if needed
        return self.delta


# Mock async generator for OpenAI stream
async def mock_openai_stream(*args, **kwargs):
    yield MockResponseChunk(delta="Call me Ishmael.")
    await asyncio.sleep(0.01)  # Simulate async delay
    yield MockResponseChunk(delta=" Some years ago...")
    # Simulate the final response object having the full text (depends on how response is used)
    # In the original code, str(response) is used, so the last yielded object needs __str__
    yield MockResponseChunk(delta="Call me Ishmael. Some years ago...")


@pytest.mark.asyncio
async def test_my_workflow_run():
    """Tests the full run of MyWorkflow, checking events and final result."""

    # Mock the OpenAI class and its methods
    mock_llm_instance = AsyncMock()
    mock_llm_instance.astream_complete.return_value = mock_openai_stream()

    # Patch the OpenAI class within the streaming_events module
    with patch(
        "llama_demo.examples.streaming_events.OpenAI", return_value=mock_llm_instance
    ) as mock_openai_class:
        workflow = MyWorkflow(timeout=10)  # Shorter timeout for tests
        handler = workflow.run(first_input="Test start")

        collected_events = []
        async for event in handler.stream_events():
            collected_events.append(event)
            print(f"Collected event: {event}")  # Debugging print

        final_result = await handler

        # --- Assertions ---

        # Check number of events
        # Progress(1), Progress(openai_1), Progress(openai_2), Progress(openai_3), Progress(3), Stop
        assert len(collected_events) == 6

        # Check event types and order
        assert isinstance(collected_events[0], ProgressEvent)
        assert isinstance(collected_events[1], ProgressEvent)
        assert isinstance(collected_events[2], ProgressEvent)
        assert isinstance(collected_events[3], ProgressEvent)
        assert isinstance(collected_events[4], ProgressEvent)
        assert isinstance(collected_events[5], StopEvent)

        # Check specific event content
        assert collected_events[0].msg == "Step one is happening"
        assert collected_events[1].msg == "Call me Ishmael."
        assert collected_events[2].msg == " Some years ago..."
        # This comes from the 3rd yield in the mock generator
        assert collected_events[3].msg == "Call me Ishmael. Some years ago..."
        assert collected_events[4].msg == "Step three is happening"
        assert collected_events[5].result == "Workflow complete."

        # Check final result
        assert final_result == "Workflow complete."

        # Verify OpenAI was called
        mock_openai_class.assert_called_once_with(model="gpt-4.1-nano-2025-04-14")
        mock_llm_instance.astream_complete.assert_awaited_once_with(
            "Please give me the first 3 paragraphs of Moby Dick, a book in the public domain."
        )


@pytest.mark.asyncio
async def test_step_one():
    """Tests step_one directly."""
    workflow = MyWorkflow()
    ctx_mock = MagicMock(spec=Context)
    start_event = StartEvent(input="start")
    result = await workflow.step_one(ctx_mock, start_event)

    assert isinstance(result, FirstEvent)
    assert result.first_output == "First step complete."
    ctx_mock.write_event_to_stream.assert_called_once_with(
        ProgressEvent(msg="Step one is happening")
    )


@pytest.mark.asyncio
async def test_step_three():
    """Tests step_three directly."""
    workflow = MyWorkflow()
    ctx_mock = MagicMock(spec=Context)
    second_event = SecondEvent(second_output="done", response="full text")
    result = await workflow.step_three(ctx_mock, second_event)

    assert isinstance(result, StopEvent)
    assert result.result == "Workflow complete."
    ctx_mock.write_event_to_stream.assert_called_once_with(
        ProgressEvent(msg="Step three is happening")
    )


# Note: Testing step_two directly is complex due to mocking the async generator
# and the context interaction. Testing it via the full workflow run is more practical.

# Add __init__.py files if necessary for test discovery
# For example, create empty files:
# tests/__init__.py
# tests/examples/__init__.py
