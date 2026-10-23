# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from collections.abc import Sequence

import regex as re

from vllm.entrypoints.generate.base.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.logger import init_logger
from vllm.renderers.chat_utils import make_tool_call_id
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.abstract_tool_parser import (
    Tool,
    ToolParser,
)
from vllm.tool_parsers.utils import is_complete_json, partial_tag_overlap
from vllm.utils.mistral import is_mistral_tokenizer

logger = init_logger(__name__)


class Hermes2ProToolParser(ToolParser):
    structural_tag_model = "hermes"
    tool_call_start_token: str = "<tool_call>"
    tool_call_end_token: str = "</tool_call>"
    tool_call_regex = re.compile(
        r"<tool_call>(.*?)</tool_call>|<tool_call>(.*)", re.DOTALL
    )
    scratch_pad_regex = re.compile(r"<scratch_pad>(.*?)</scratch_pad>", re.DOTALL)

    def __init__(self, tokenizer: TokenizerLike, tools: list[Tool] | None = None):
        super().__init__(tokenizer, tools)

        if is_mistral_tokenizer(tokenizer):
            logger.error("Detected Mistral tokenizer when using a Hermes model")
            self.model_tokenizer = tokenizer.tokenizer

        if not self.model_tokenizer:
            raise ValueError(
                "The model tokenizer must be passed to the ToolParser "
                "constructor during construction."
            )

        # Streaming state: what has been sent to the client.
        self._sent_content_idx: int = 0
        # Accumulated output, so end-of-stream can flush a trailing region
        # that never became a tool call.
        self._current_text: str = ""
        # How far into `_current_text` trailing prose has already been sent.
        self._trailing_prose_idx: int = 0

    def adjust_request(
        self, request: ChatCompletionRequest | ResponsesRequest
    ) -> ChatCompletionRequest | ResponsesRequest:
        request = super().adjust_request(request)
        if request.tools and request.tool_choice != "none":
            # do not skip special tokens because the tool_call tokens are
            # marked "special" in some models. Since they are skipped
            # prior to the call to the tool parser, it breaks tool calling.
            request.skip_special_tokens = False
        return request

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        # sanity check; avoid unnecessary processing
        if self.tool_call_start_token not in model_output:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

        tool_calls: list[ToolCall] = []
        # A later region that is not a call must not drop earlier calls.
        # Text between valid calls stays out of `content` (prefix only).
        # The unparsable region and everything after it are content.
        prose_from: int | None = None
        try:
            for match in self.tool_call_regex.finditer(model_output):
                raw = match.group(1)
                if raw is None:
                    raw = match.group(2) or ""
                try:
                    tool_calls.append(self._tool_call_from_json(raw))
                except Exception:
                    if not tool_calls:
                        raise
                    prose_from = match.start()
                    break
        except Exception:
            logger.exception("Error in extracting tool call from response.")
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

        if not tool_calls:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

        content = model_output[: model_output.find(self.tool_call_start_token)]
        if prose_from is not None:
            content += model_output[prose_from:]
        return ExtractedToolCallInformation(
            tools_called=True,
            tool_calls=tool_calls,
            content=content if content else None,
        )

    def _extract_content(self, current_text: str) -> str | None:
        """Return unsent non-tool-call text, or None.

        Holds back any suffix that could be a partial <tool_call> tag.
        """
        if self.tool_call_start_token not in current_text:
            overlap_length = partial_tag_overlap(
                current_text, self.tool_call_start_token
            )
            sendable_idx = len(current_text) - overlap_length
        else:
            sendable_idx = current_text.index(self.tool_call_start_token)

        if sendable_idx > self._sent_content_idx:
            content = current_text[self._sent_content_idx : sendable_idx]
            self._sent_content_idx = sendable_idx
            return content
        return None

    def _tool_call_from_json(self, raw: str) -> ToolCall:
        """Build one tool call, or raise if `raw` is not a call."""
        function_call = json.loads(raw)
        return ToolCall(
            type="function",
            function=FunctionCall(
                name=function_call["name"],
                # function call args are JSON but as a string
                arguments=json.dumps(function_call["arguments"], ensure_ascii=False),
            ),
        )

    def _tool_call_regions(self, text: str) -> list[tuple[int, str, bool]]:
        """Return (start, body, closed) for each `<tool_call>` region.

        `body` excludes a partial end tag. `start` indexes `<tool_call>`.
        """
        regions: list[tuple[int, str, bool]] = []
        pos = 0
        start_tok = self.tool_call_start_token
        end_tok = self.tool_call_end_token
        while True:
            start = text.find(start_tok, pos)
            if start == -1:
                break
            body_start = start + len(start_tok)
            end = text.find(end_tok, body_start)
            if end != -1:
                regions.append((start, text[body_start:end], True))
                pos = end + len(end_tok)
                continue
            raw = text[body_start:]
            overlap = partial_tag_overlap(raw, end_tok)
            if overlap:
                raw = raw[:-overlap]
            regions.append((start, raw, False))
            break
        return regions

    def _region_json(self, body: str, closed: bool) -> tuple[str, bool]:
        """Return (json_text, is_complete) for one region body."""
        tc_json = body.strip()
        if closed:
            return tc_json, True
        # Valid JSON without a closing tag is a complete body; the end
        # tag just has not arrived yet.
        return tc_json, is_complete_json(tc_json) if tc_json else False

    def _region_role(self, body: str, closed: bool, *, final: bool) -> str:
        """Classify a region as `call`, `prose`, or `pending`.

        `pending` may still become a call. A region that does not open
        with a JSON object, or that is finished without a tool name, is
        `prose`.
        """
        stripped = body.strip()
        if not stripped:
            return "prose" if closed or final else "pending"
        if not stripped.startswith("{"):
            return "prose"
        if self._extract_tool_name(stripped):
            return "call"
        return "prose" if closed or final else "pending"

    def _unsent_trailing_prose(self, text: str, *, final: bool) -> str | None:
        """Return a trailing region that cannot be a call, if not yet sent.

        Text between valid calls is not included: non-streaming keeps only
        the prefix before the first call, plus an unparsable tail. A
        leading stray tag is left untouched so that case stays separate.
        """
        saw_call = False
        for start, body, closed in self._tool_call_regions(text):
            role = self._region_role(body, closed, final=final)
            if role == "call":
                saw_call = True
                continue
            if role != "prose" or not saw_call:
                return None
            emit_from = max(start, self._trailing_prose_idx)
            if emit_from >= len(text):
                return None
            self._trailing_prose_idx = len(text)
            return text[emit_from:]
        return None

    def finish_streaming(self) -> DeltaMessage | None:
        """Flush a trailing `<tool_call>` that never became a call.

        Called at the end of a stream. A region that already has a tool
        name stays a tool call. Only a later region that cannot be one is
        returned as content.
        """
        content = self._unsent_trailing_prose(self._current_text, final=True)
        return DeltaMessage(content=content) if content else None

    @staticmethod
    def _extract_tool_name(tc_json: str) -> str | None:
        """Extract tool name, or None if the name isn't complete yet."""
        match = re.search(r'"name"\s*:\s*"([^"]+)"', tc_json)
        return match.group(1) if match else None

    @staticmethod
    def _extract_tool_args(tc_json: str, is_complete: bool) -> str | None:
        """Extract tool arguments from the tool call JSON.

        Given {"name": "f", "arguments": {"x": 1}}, returns '{"x": 1}'.
        When is_complete, strips the trailing '}' that closes the outer
        object (not the arguments). For partial JSON, returns as-is.
        """
        match = re.search(r'"arguments"\s*:\s*', tc_json)
        if not match:
            return None
        raw = tc_json[match.end() :]
        if is_complete:
            raw = raw.rstrip()
            if raw.endswith("}"):
                raw = raw[:-1].rstrip()
        return raw

    def _compute_args_diff(
        self, index: int, tc_json: str, is_complete: bool
    ) -> str | None:
        """Return new argument text not yet sent for tool `index`, or None."""
        args = self._extract_tool_args(tc_json, is_complete)
        if args is None or len(args) <= len(self.streamed_args_for_tool[index]):
            return None
        diff = args[len(self.streamed_args_for_tool[index]) :]
        self.streamed_args_for_tool[index] = args
        self.prev_tool_call_arr[index]["arguments"] = args
        return diff

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> DeltaMessage | None:
        """Incrementally stream tool call deltas from accumulated output.

        On each invocation, re-parses the full ``current_text`` to find
        ``<tool_call>`` regions, then diffs against previously sent state
        to emit only new content, tool names, or argument fragments.

        Returns a ``DeltaMessage`` containing either plain content (for
        text preceding any tool call) or one or more ``DeltaToolCall``
        entries, or ``None`` if there is nothing new to send yet."""
        try:
            self._current_text = current_text
            # Extract any content before tool calls.
            content = self._extract_content(current_text)
            regions = self._tool_call_regions(current_text)
            tool_call_jsons = [
                self._region_json(body, closed) for _start, body, closed in regions
            ]
            tool_call_deltas: list[DeltaToolCall] = []
            saw_call = False

            for i, (tc_json, is_complete) in enumerate(tool_call_jsons):
                _start, body, closed = regions[i]
                role = self._region_role(body, closed, final=False)
                # A later region that cannot be a call is content. Stop
                # before parsing a name out of it.
                if saw_call and role == "prose":
                    break
                if role == "pending":
                    # Can't skip to tool i+1 if i isn't ready.
                    break

                if i >= len(self.prev_tool_call_arr):
                    self.prev_tool_call_arr.append({})
                    self.streamed_args_for_tool.append("")

                # Stream back tool name.
                if "name" not in self.prev_tool_call_arr[i]:
                    name = self._extract_tool_name(tc_json)
                    if not name:
                        # Can't skip to tool i+1 if i isn't ready
                        break
                    self.prev_tool_call_arr[i]["name"] = name
                    tool_call_deltas.append(
                        DeltaToolCall(
                            index=i,
                            type="function",
                            id=make_tool_call_id(),
                            function=DeltaFunctionCall(name=name).model_dump(
                                exclude_none=True
                            ),
                        )
                    )

                if "name" in self.prev_tool_call_arr[i]:
                    saw_call = True

                # Stream back new tool args by diffing against what was sent.
                args_diff = self._compute_args_diff(i, tc_json, is_complete)
                if args_diff:
                    tool_call_deltas.append(
                        DeltaToolCall(
                            index=i,
                            function=DeltaFunctionCall(arguments=args_diff).model_dump(
                                exclude_none=True
                            ),
                        )
                    )

            trailing = self._unsent_trailing_prose(current_text, final=False)
            if trailing:
                content = f"{content}{trailing}" if content else trailing

            if content or tool_call_deltas:
                return DeltaMessage(
                    content=content,
                    tool_calls=tool_call_deltas,
                )

            return None

        except Exception:
            logger.exception("Error trying to handle streaming tool call.")
            return None
