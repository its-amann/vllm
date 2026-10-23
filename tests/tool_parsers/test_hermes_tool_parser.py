# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest

from tests.tool_parsers.utils import (
    get_function_delta,
    split_string_into_token_deltas,
)
from vllm.entrypoints.generate.base.protocol import DeltaMessage
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.abstract_tool_parser import ToolParser
from vllm.tool_parsers.granite4_tool_parser import Granite4ToolParser
from vllm.tool_parsers.hermes_tool_parser import Hermes2ProToolParser

CONFIGS: dict[str, dict[str, type[ToolParser]]] = {
    "llama": {
        "tool_parser": Hermes2ProToolParser,
    },
    "granite4": {
        "tool_parser": Granite4ToolParser,
    },
}


@pytest.fixture
def qwen_tokenizer() -> TokenizerLike:
    from vllm.tokenizers import get_tokenizer

    return get_tokenizer("Qwen/Qwen3-32B")


@pytest.fixture(params=CONFIGS.keys())
def hermes_parser(request, qwen_tokenizer: TokenizerLike) -> ToolParser:
    config = CONFIGS[request.param]
    return config["tool_parser"](qwen_tokenizer)


@pytest.fixture
def any_chat_request() -> ChatCompletionRequest:
    return ChatCompletionRequest(
        seed=42,
        model="Qwen/Qwen3-32B",
        messages=[],
    )


def test_hermes_parser_streaming_just_forward_text(
    qwen_tokenizer: TokenizerLike,
    hermes_parser: ToolParser,
    any_chat_request: ChatCompletionRequest,
) -> None:
    text = """This is some prior text that has nothing to do with tool calling."""
    tokens = qwen_tokenizer.encode(text)
    previous_text = ""
    delta_messages = []
    for token in tokens:
        delta_text = qwen_tokenizer.decode([token])
        current_text = previous_text + delta_text
        delta = hermes_parser.extract_tool_calls_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=delta_text,
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[],
            request=any_chat_request,
        )
        previous_text = current_text
        delta_messages.append(delta)

    contents: list[str] = []
    for delta in delta_messages:
        assert delta is not None
        assert not delta.tool_calls
        assert delta.content is not None
        contents.append(delta.content)

    print(delta_messages)
    assert "".join(contents) == text


def test_hermes_parser_streaming_failure_case_bug_19056(
    qwen_tokenizer: TokenizerLike,
    hermes_parser: ToolParser,
    any_chat_request: ChatCompletionRequest,
) -> None:
    text = """<tool_call>
{"name": "final_answer", "arguments": {"trigger": true}}
</tool_call>"""
    tokens = qwen_tokenizer.encode(text)
    previous_text = ""
    delta_messages = []
    for token in tokens:
        text = qwen_tokenizer.decode([token])
        current_text = previous_text + text
        delta = hermes_parser.extract_tool_calls_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=text,
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[],
            request=any_chat_request,
        )
        previous_text = current_text
        if delta is not None:
            delta_messages.append(delta)

    assert delta_messages[0].tool_calls[0].function is not None
    assert delta_messages[0].tool_calls[0].function.name == "final_answer"
    tool_call_args = "".join(
        get_function_delta(delta.tool_calls[0]).arguments or ""
        for delta in delta_messages
    )
    assert tool_call_args == '{"trigger": true}'


def test_hermes_parser_streaming(
    qwen_tokenizer: TokenizerLike,
    hermes_parser: ToolParser,
    any_chat_request: ChatCompletionRequest,
) -> None:
    text = '<tool_call>\
{"name": "get_current_temperature",\
"arguments": {"location":\
"San Francisco, California, United States", "unit": "celsius"}}\
</tool_call>'

    tokens = qwen_tokenizer.encode(text)
    previous_text = ""
    delta_messages = []
    for token in tokens:
        text = qwen_tokenizer.decode([token])
        current_text = previous_text + text
        delta = hermes_parser.extract_tool_calls_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=text,
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[],
            request=any_chat_request,
        )
        previous_text = current_text
        if delta is not None:
            delta_messages.append(delta)
    print(delta_messages)
    assert delta_messages[0].tool_calls[0].function is not None
    assert delta_messages[0].tool_calls[0].function.name == "get_current_temperature"
    # load to normalize whitespace
    tool_call_args = json.loads(
        "".join(
            get_function_delta(delta.tool_calls[0]).arguments or ""
            for delta in delta_messages
        )
    )
    assert tool_call_args == {
        "location": "San Francisco, California, United States",
        "unit": "celsius",
    }


def _simulate_streaming(
    tokenizer: TokenizerLike,
    parser: ToolParser,
    request: ChatCompletionRequest,
    text: str,
    stream_interval: int = 1,
) -> list:
    """Simulate streaming with a given stream_interval.

    Tokens are batched into chunks of `stream_interval` tokens,
    mimicking how the output processor delivers them.
    Returns a list of non-None DeltaMessages.
    """
    tokens = tokenizer.encode(text)
    previous_text = ""
    delta_messages = []
    for i in range(0, len(tokens), stream_interval):
        chunk_ids = tokens[i : i + stream_interval]
        delta_text = tokenizer.decode(chunk_ids)
        current_text = previous_text + delta_text
        delta = parser.extract_tool_calls_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=delta_text,
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=chunk_ids,
            request=request,
        )
        previous_text = current_text
        if delta is not None:
            delta_messages.append(delta)
    return delta_messages


@pytest.mark.parametrize("stream_interval", [2, 3, 5, 8])
def test_hermes_streaming_tool_call_with_stream_interval(
    qwen_tokenizer: TokenizerLike,
    any_chat_request: ChatCompletionRequest,
    stream_interval: int,
) -> None:
    """Tool call streaming must produce correct name + args at any interval."""
    text = (
        '<tool_call>{"name": "get_current_temperature", '
        '"arguments": {"location": "San Francisco", "unit": "celsius"}}'
        "</tool_call>"
    )
    parser = Hermes2ProToolParser(qwen_tokenizer)
    deltas = _simulate_streaming(
        qwen_tokenizer, parser, any_chat_request, text, stream_interval
    )

    # Flatten all DeltaToolCalls across all deltas.
    tool_deltas = [tc for d in deltas if d.tool_calls for tc in d.tool_calls]
    assert tool_deltas, "Expected at least one tool call delta"
    assert tool_deltas[0].function.name == "get_current_temperature"

    # Concatenated arguments must be valid JSON matching the original.
    args_str = "".join(tc.function.arguments or "" for tc in tool_deltas)
    assert json.loads(args_str) == {
        "location": "San Francisco",
        "unit": "celsius",
    }


@pytest.mark.parametrize("stream_interval", [2, 3, 5, 8])
def test_hermes_streaming_content_then_tool_call_with_stream_interval(
    qwen_tokenizer: TokenizerLike,
    any_chat_request: ChatCompletionRequest,
    stream_interval: int,
) -> None:
    """Content before a tool call must be fully streamed, then tool call."""
    text = (
        "Sure, let me check the weather."
        '<tool_call>{"name": "get_weather", '
        '"arguments": {"city": "NYC"}}</tool_call>'
    )
    parser = Hermes2ProToolParser(qwen_tokenizer)
    deltas = _simulate_streaming(
        qwen_tokenizer, parser, any_chat_request, text, stream_interval
    )

    content_deltas = [d for d in deltas if d.content]
    tool_deltas = [d for d in deltas if d.tool_calls]

    # Content must reconstruct the prefix.
    content_str = "".join(d.content for d in content_deltas)
    assert content_str == "Sure, let me check the weather."

    # Tool call must be correct.
    tool_calls = [tc for d in tool_deltas for tc in d.tool_calls]
    assert tool_calls[0].function.name == "get_weather"
    args_str = "".join(tc.function.arguments or "" for tc in tool_calls)
    assert json.loads(args_str) == {"city": "NYC"}


@pytest.mark.parametrize("stream_interval", [1, 2, 4])
def test_hermes_streaming_multiple_tool_calls_with_stream_interval(
    qwen_tokenizer: TokenizerLike,
    any_chat_request: ChatCompletionRequest,
    stream_interval: int,
) -> None:
    """Multiple sequential tool calls must each be streamed correctly."""
    text = (
        '<tool_call>{"name": "search", "arguments": {"q": "cats"}}</tool_call>'
        '<tool_call>{"name": "search", "arguments": {"q": "dogs"}}</tool_call>'
    )
    parser = Hermes2ProToolParser(qwen_tokenizer)
    deltas = _simulate_streaming(
        qwen_tokenizer, parser, any_chat_request, text, stream_interval
    )

    # Flatten all DeltaToolCalls across all deltas.
    all_tool_calls = [tc for d in deltas if d.tool_calls for tc in d.tool_calls]

    # Separate by tool index.
    tool0 = [tc for tc in all_tool_calls if tc.index == 0]
    tool1 = [tc for tc in all_tool_calls if tc.index == 1]

    assert tool0[0].function.name == "search"
    args0 = "".join(tc.function.arguments or "" for tc in tool0)
    assert json.loads(args0) == {"q": "cats"}

    assert tool1[0].function.name == "search"
    args1 = "".join(tc.function.arguments or "" for tc in tool1)
    assert json.loads(args1) == {"q": "dogs"}


@pytest.mark.parametrize("stream_interval", [2, 5])
def test_hermes_streaming_boolean_args_with_stream_interval(
    qwen_tokenizer: TokenizerLike,
    any_chat_request: ChatCompletionRequest,
    stream_interval: int,
) -> None:
    """Regression test for bug #19056 with stream_interval > 1."""
    text = (
        "<tool_call>\n"
        '{"name": "final_answer", "arguments": {"trigger": true}}\n'
        "</tool_call>"
    )
    parser = Hermes2ProToolParser(qwen_tokenizer)
    deltas = _simulate_streaming(
        qwen_tokenizer, parser, any_chat_request, text, stream_interval
    )

    tool_calls = [tc for d in deltas if d.tool_calls for tc in d.tool_calls]
    assert tool_calls[0].function.name == "final_answer"
    args_str = "".join(tc.function.arguments or "" for tc in tool_calls)
    assert json.loads(args_str) == {"trigger": True}


@pytest.mark.parametrize("stream_interval", [2, 3, 5])
def test_hermes_streaming_just_forward_text_with_stream_interval(
    qwen_tokenizer: TokenizerLike,
    any_chat_request: ChatCompletionRequest,
    stream_interval: int,
) -> None:
    """Plain text with no tool calls must be fully forwarded."""
    text = "This is plain text with no tool calling involved."
    parser = Hermes2ProToolParser(qwen_tokenizer)
    deltas = _simulate_streaming(
        qwen_tokenizer, parser, any_chat_request, text, stream_interval
    )

    for d in deltas:
        assert not d.tool_calls
    assert "".join(d.content for d in deltas) == text


def test_hermes_parser_non_streaming_no_tool_call(
    hermes_parser: ToolParser,
    any_chat_request: ChatCompletionRequest,
) -> None:
    text = """This is not a tool call."""
    tool_call = hermes_parser.extract_tool_calls(
        model_output=text,
        request=any_chat_request,
    )

    assert tool_call is not None
    assert not tool_call.tools_called


def test_hermes_parser_non_streaming_tool_call_between_tags(
    hermes_parser: ToolParser,
    any_chat_request: ChatCompletionRequest,
) -> None:
    text = """<tool_call>
{"name": "final_answer", "arguments": {"trigger": true}}
</tool_call>"""
    tool_call = hermes_parser.extract_tool_calls(
        model_output=text,
        request=any_chat_request,
    )

    assert tool_call is not None
    assert tool_call.tools_called
    assert tool_call.tool_calls[0].function.name == "final_answer"
    assert tool_call.tool_calls[0].function.arguments == '{"trigger": true}'


def test_hermes_parser_non_streaming_tool_call_until_eos(
    hermes_parser: ToolParser,
    any_chat_request: ChatCompletionRequest,
) -> None:
    if isinstance(hermes_parser, Granite4ToolParser):
        pytest.skip(reason="The Granite4 tool parser enforces a complete response")

    text = """<tool_call>
{"name": "final_answer", "arguments": {"trigger": true}}"""
    tool_call = hermes_parser.extract_tool_calls(
        model_output=text,
        request=any_chat_request,
    )

    assert tool_call is not None
    assert tool_call.tools_called
    assert tool_call.tool_calls[0].function.name == "final_answer"
    assert tool_call.tool_calls[0].function.arguments == '{"trigger": true}'


def test_hermes_parser_non_streaming_tool_call_invalid_json(
    hermes_parser: ToolParser,
    any_chat_request: ChatCompletionRequest,
) -> None:
    # Missing closing brace to trigger exception
    text = """<tool_call>
{"name": "final_answer", "arguments": {"trigger": true}"""
    tool_call = hermes_parser.extract_tool_calls(
        model_output=text,
        request=any_chat_request,
    )

    assert tool_call is not None
    assert not tool_call.tools_called


def test_hermes_streaming_content_and_tool_call_in_single_chunk(
    qwen_tokenizer: TokenizerLike,
    any_chat_request: ChatCompletionRequest,
) -> None:
    """Content + complete tool call in one chunk must both be emitted."""
    text = 'Hi!<tool_call>{"name": "f", "arguments": {"x": 1}}</tool_call>'
    # Use a stream_interval large enough to guarantee a single chunk.
    parser = Hermes2ProToolParser(qwen_tokenizer)
    deltas = _simulate_streaming(
        qwen_tokenizer,
        parser,
        any_chat_request,
        text,
        stream_interval=9999,
    )

    content_parts = [d.content for d in deltas if d.content]
    tool_parts = [tc for d in deltas if d.tool_calls for tc in d.tool_calls]

    assert "".join(content_parts) == "Hi!"
    assert tool_parts[0].function.name == "f"
    args_str = "".join(tc.function.arguments or "" for tc in tool_parts)
    assert json.loads(args_str) == {"x": 1}


# A completed call, then a `<tool_call>` region that cannot be a call.
# The call is kept. The bad region and the text after it are content.
# Text between valid calls, and text after the last valid call when no
# bad region follows, stays omitted (`" Second. "`, and `" Third."` in
# the two-call case).
_WEATHER_CALL = (
    '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>'
)
_PARIS = {"city": "Paris"}

_TRAILING_CASES = [
    (
        "unclosed_tag",
        f"First. {_WEATHER_CALL} Second. <tool_call>",
        "First. <tool_call>",
        [("get_weather", _PARIS)],
    ),
    (
        "unclosed_prose",
        f"First. {_WEATHER_CALL} Second. <tool_call> opens a call. Done.",
        "First. <tool_call> opens a call. Done.",
        [("get_weather", _PARIS)],
    ),
    (
        "empty_closed_tag",
        f"First. {_WEATHER_CALL} Second. <tool_call></tool_call> Third.",
        "First. <tool_call></tool_call> Third.",
        [("get_weather", _PARIS)],
    ),
    (
        "non_json_closed_tag",
        f"First. {_WEATHER_CALL} Second. <tool_call>not json</tool_call> Third.",
        "First. <tool_call>not json</tool_call> Third.",
        [("get_weather", _PARIS)],
    ),
    (
        "trailing_text_only",
        f"First. {_WEATHER_CALL} Second.",
        "First. ",
        [("get_weather", _PARIS)],
    ),
    (
        "text_between_two_valid_calls",
        f"First. {_WEATHER_CALL} Second. {_WEATHER_CALL} Third.",
        "First. ",
        [("get_weather", _PARIS), ("get_weather", _PARIS)],
    ),
    (
        "unclosed_complete_json",
        "First. "
        + _WEATHER_CALL
        + ' Second. <tool_call>\n{"name": "get_time", "arguments": {"tz": "UTC"}}',
        "First. ",
        [("get_weather", _PARIS), ("get_time", {"tz": "UTC"})],
    ),
]


def _chunk_text(tokenizer: TokenizerLike, text: str, mode: str) -> list[str]:
    if mode == "char":
        return list(text)
    if mode == "chunk5":
        return [text[i : i + 5] for i in range(0, len(text), 5)]
    assert mode == "token"
    encoded = tokenizer.encode(text, add_special_tokens=False)
    assert tokenizer.decode(encoded) == text
    return split_string_into_token_deltas(tokenizer, text)


def _record_delta(
    message: DeltaMessage | None,
    content_parts: list[str],
    names: dict[int, str],
    arguments: dict[int, str],
) -> None:
    if message is None:
        return
    if message.content:
        content_parts.append(message.content)
    for tool_call in message.tool_calls or []:
        function = get_function_delta(tool_call)
        if function.name:
            names[tool_call.index] = function.name
        if function.arguments:
            arguments[tool_call.index] = (
                arguments.get(tool_call.index, "") + function.arguments
            )


def _stream_hermes(
    parser: Hermes2ProToolParser,
    request: ChatCompletionRequest,
    chunks: list[str],
    *,
    finish: bool,
) -> tuple[str, list[tuple[str, dict]]]:
    previous = ""
    content_parts: list[str] = []
    names: dict[int, str] = {}
    arguments: dict[int, str] = {}
    for delta in chunks:
        current = previous + delta
        message = parser.extract_tool_calls_streaming(
            previous_text=previous,
            current_text=current,
            delta_text=delta,
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[],
            request=request,
        )
        previous = current
        _record_delta(message, content_parts, names, arguments)
    if finish:
        _record_delta(parser.finish_streaming(), content_parts, names, arguments)
    calls = [
        (names[index], json.loads(arguments[index]))
        for index in sorted(set(names) | set(arguments))
    ]
    return "".join(content_parts), calls


@pytest.mark.parametrize(
    ("case_id", "text", "expected_content", "expected_calls"),
    _TRAILING_CASES,
    ids=[case[0] for case in _TRAILING_CASES],
)
def test_hermes_nonstreaming_trailing_region(
    qwen_tokenizer: TokenizerLike,
    any_chat_request: ChatCompletionRequest,
    case_id: str,
    text: str,
    expected_content: str,
    expected_calls: list[tuple[str, dict]],
) -> None:
    """Non-streaming keeps valid calls and surfaces a bad trailing region."""
    result = Hermes2ProToolParser(qwen_tokenizer).extract_tool_calls(
        model_output=text,
        request=any_chat_request,
    )

    assert result.tools_called, case_id
    assert result.content == expected_content, case_id
    assert len(result.tool_calls) == len(expected_calls)
    for tool_call, (name, args) in zip(result.tool_calls, expected_calls):
        assert tool_call.function.name == name
        assert json.loads(tool_call.function.arguments) == args


@pytest.mark.parametrize("mode", ["char", "chunk5", "token"])
@pytest.mark.parametrize(
    ("case_id", "text", "expected_content", "expected_calls"),
    _TRAILING_CASES,
    ids=[case[0] for case in _TRAILING_CASES],
)
def test_hermes_streaming_trailing_region_matches_nonstreaming(
    qwen_tokenizer: TokenizerLike,
    any_chat_request: ChatCompletionRequest,
    mode: str,
    case_id: str,
    text: str,
    expected_content: str,
    expected_calls: list[tuple[str, dict]],
) -> None:
    """Streaming agrees with non-streaming, including at end of stream."""
    parser = Hermes2ProToolParser(qwen_tokenizer)
    content, calls = _stream_hermes(
        parser,
        any_chat_request,
        _chunk_text(qwen_tokenizer, text, mode),
        finish=True,
    )

    assert content == expected_content
    assert calls == expected_calls

    if case_id == "unclosed_tag":
        held = Hermes2ProToolParser(qwen_tokenizer)
        held_content, held_calls = _stream_hermes(
            held,
            any_chat_request,
            _chunk_text(qwen_tokenizer, text, mode),
            finish=False,
        )
        # The empty unclosed tag is still a possible call until the stream ends.
        assert held_content == "First. "
        assert held_calls == [("get_weather", _PARIS)]
        flushed = held.finish_streaming()
        assert flushed is not None
        assert flushed.content == "<tool_call>"


def test_hermes_nonstreaming_invalid_json_keeps_raw_text(
    qwen_tokenizer: TokenizerLike,
    any_chat_request: ChatCompletionRequest,
) -> None:
    """One incomplete call still falls back to the raw text."""
    text = '<tool_call>\n{"name": "final_answer", "arguments": {"trigger": true}'
    result = Hermes2ProToolParser(qwen_tokenizer).extract_tool_calls(
        model_output=text,
        request=any_chat_request,
    )

    assert not result.tools_called
    assert result.tool_calls == []
    assert result.content == text
