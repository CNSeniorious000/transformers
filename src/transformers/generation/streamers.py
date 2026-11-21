# Copyright 2023 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import asyncio
import codecs
import time
import unicodedata
from functools import cache
from queue import Queue
from typing import TYPE_CHECKING, Any, cast


if TYPE_CHECKING:
    from tokenizers import Tokenizer

    from ..tokenization_utils_base import PreTrainedTokenizerBase


@cache
def _byte_level_alphabet() -> dict[str, int]:
    from ..convert_slow_tokenizer import bytes_to_unicode

    return {char: byte for byte, char in bytes_to_unicode().items()}


def _get_raw_byte_level_backend(tokenizer: PreTrainedTokenizerBase, decode_kwargs: dict[str, Any]) -> Tokenizer | None:
    from ..utils import is_tokenizers_available

    if not is_tokenizers_available():
        return None

    from tokenizers.decoders import ByteLevel

    from ..tokenization_utils_base import PreTrainedTokenizerBase
    from ..tokenization_utils_tokenizers import TokenizersBackend

    # Other decoder stages, custom decode methods and text cleanup can change the decoded suffix.
    if not isinstance(tokenizer, TokenizersBackend):
        return None
    if (
        getattr(tokenizer.decode, "__func__", None) is not PreTrainedTokenizerBase.decode
        or getattr(tokenizer._decode, "__func__", None) is not TokenizersBackend._decode
        or decode_kwargs.keys() - {"clean_up_tokenization_spaces", "skip_special_tokens"}
    ):
        return None
    backend = tokenizer.backend_tokenizer
    if type(backend.decoder) is not ByteLevel:
        return None
    cleanup = decode_kwargs.get("clean_up_tokenization_spaces")
    if cleanup is None:
        cleanup = tokenizer.clean_up_tokenization_spaces
    if cleanup and (
        type(backend.model).__name__ != "BPE"
        or tokenizer.clean_up_tokenization_spaces_for_bpe_even_though_it_will_corrupt_output
    ):
        return None
    return backend


def _byte_level_has_partial_utf8(backend: Tokenizer, token_ids: list[int], skip_special_tokens: bool) -> bool:
    # A UTF-8 code point has at most three trailing bytes waiting for completion.
    # Starting in the middle of an earlier code point does not affect this suffix:
    # continuation bytes cannot start a new code point.
    alphabet = _byte_level_alphabet()
    tail = b""
    for token_id in reversed(token_ids):
        token = backend.id_to_token(token_id)
        # Use the backend's own special-token filter, not the wrapper's list of special IDs.
        if not token or (skip_special_tokens and not backend.decode([token_id], skip_special_tokens=True)):
            continue
        try:
            token_bytes = bytes(alphabet[char] for char in token)
        except KeyError:
            # ByteLevel leaves the entire token untouched if any character is outside its alphabet.
            token_bytes = token.encode("utf-8")
        tail = (token_bytes + tail)[-3:]
        if len(tail) == 3:
            break

    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    decoder.decode(tail, final=False)
    pending = decoder.getstate()[0]
    # Some Python implementations can also buffer an invalid surrogate prefix, which is
    # already irrecoverable. Only an unfinished code point produces one replacement at EOF.
    return pending.decode("utf-8", errors="replace") == "\ufffd"


class BaseStreamer:
    """
    Base class from which `.generate()` streamers should inherit.
    """

    def put(self, value):
        """Function that is called by `.generate()` to push new tokens"""
        raise NotImplementedError()

    def end(self):
        """Function that is called by `.generate()` to signal the end of generation"""
        raise NotImplementedError()


class TextStreamer(BaseStreamer):
    """
    Simple text streamer that prints the token(s) to stdout as soon as they are decoded.

    Parameters:
        tokenizer (`AutoTokenizer`):
            The tokenizer used to decode the tokens.
        skip_prompt (`bool`, *optional*, defaults to `False`):
            Whether to skip the prompt to `.generate()` or not. Useful e.g. for chatbots.
        decode_kwargs (`dict`, *optional*):
            Additional keyword arguments to pass to the tokenizer's `decode` method.

    Examples:

        ```python
        >>> from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

        >>> tok = AutoTokenizer.from_pretrained("openai-community/gpt2")
        >>> model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
        >>> inputs = tok(["An increasing sequence: one,"], return_tensors="pt")
        >>> streamer = TextStreamer(tok)

        >>> # Despite returning the usual output, the streamer will also print the generated text to stdout.
        >>> _ = model.generate(**inputs, streamer=streamer, max_new_tokens=20)
        An increasing sequence: one, two, three, four, five, six, seven, eight, nine, ten, eleven,
        ```
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase, skip_prompt: bool = False, **decode_kwargs: Any):
        self.tokenizer = tokenizer
        self.skip_prompt = skip_prompt
        self.decode_kwargs = decode_kwargs

        # variables used in the streaming process
        self.token_cache: list[int] = []
        self.print_len = 0
        self.next_tokens_are_prompt = True

    def put(self, value):
        """
        Receives tokens, decodes them, and prints them to stdout as soon as they are decoded.
        """
        if len(value.shape) > 1 and value.shape[0] > 1:
            raise ValueError("TextStreamer only supports batch size 1")
        elif len(value.shape) > 1:
            value = value[0]

        if self.skip_prompt and self.next_tokens_are_prompt:
            self.next_tokens_are_prompt = False
            return

        # Add the new token to the cache and decodes the entire thing.
        self.token_cache.extend(value.tolist())
        text = cast(str, self.tokenizer.decode(self.token_cache, **self.decode_kwargs))

        # Only raw ByteLevel decoding lets us distinguish an unfinished UTF-8 suffix from a
        # real replacement character. Keep using the tokenizer's decode for the text itself.
        partial_utf8 = False
        known_utf8_boundaries = False
        if "\ufffd" in text[self.print_len :]:
            backend = _get_raw_byte_level_backend(self.tokenizer, self.decode_kwargs)
            if backend is not None:
                known_utf8_boundaries = True
                if text.endswith("\ufffd") and _byte_level_has_partial_utf8(
                    backend, self.token_cache, self.decode_kwargs.get("skip_special_tokens", False)
                ):
                    partial_utf8 = True
                    text = text[:-1]

        # After the symbol for a new line, we flush the cache but keep the last token as decoding context to preserve
        # leading spaces on the next line. A newline in the stable prefix must not discard pending UTF-8 bytes.
        if text.endswith("\n") and not partial_utf8:
            printable_text = text[self.print_len :]
            self.token_cache = self.token_cache[-1:]
            self.print_len = len(self.tokenizer.decode(self.token_cache[-1], **self.decode_kwargs))
        # If the stable prefix ends with a newline or CJK character, print it in full.
        elif text.endswith("\n") or (len(text) > 0 and self._is_chinese_char(ord(text[-1]))):
            printable_text = text[self.print_len :]
            self.print_len += len(printable_text)
        # Letters (with their combining marks), digits and spaces are printed right away, so text without spaces (long
        # words, Thai) streams too. Symbols wait for the next space: they may start some formatting (`\n`, `<br>`).
        # A replacement character is only safe to print here after checking the UTF-8 boundary.
        elif all(
            char.isspace() or unicodedata.category(char)[0] in "LMN" or (known_utf8_boundaries and char == "\ufffd")
            for char in text[self.print_len :]
        ):
            printable_text = text[self.print_len :]
            self.print_len += len(printable_text)
        # Otherwise, prints until the last space char (simple heuristic to avoid printing incomplete words,
        # which may change with the subsequent token -- there are probably smarter ways to do this!)
        else:
            printable_text = text[self.print_len : text.rfind(" ") + 1]
            self.print_len += len(printable_text)

        self.on_finalized_text(printable_text)

    def end(self):
        """Flushes any remaining cache and prints a newline to stdout."""
        # Flush the cache, if it exists
        if len(self.token_cache) > 0:
            text = cast(str, self.tokenizer.decode(self.token_cache, **self.decode_kwargs))
            printable_text = text[self.print_len :]
            self.token_cache = []
            self.print_len = 0
        else:
            printable_text = ""

        self.next_tokens_are_prompt = True
        self.on_finalized_text(printable_text, stream_end=True)

    def on_finalized_text(self, text: str, stream_end: bool = False):
        """Prints the new text to stdout. If the stream is ending, also prints a newline."""
        print(text, flush=True, end="" if not stream_end else None)

    def _is_chinese_char(self, cp):
        """Checks whether CP is the codepoint of a CJK character."""
        # This defines a "chinese character" as anything in the CJK Unicode block:
        #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        #
        # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
        # despite its name. The modern Korean Hangul alphabet is a different block,
        # as is Japanese Hiragana and Katakana. Those alphabets are used to write
        # space-separated words, so they are not treated specially and handled
        # like the all of the other languages.
        if (
            (cp >= 0x4E00 and cp <= 0x9FFF)
            or (cp >= 0x3400 and cp <= 0x4DBF)
            or (cp >= 0x20000 and cp <= 0x2A6DF)
            or (cp >= 0x2A700 and cp <= 0x2B73F)
            or (cp >= 0x2B740 and cp <= 0x2B81F)
            or (cp >= 0x2B820 and cp <= 0x2CEAF)
            or (cp >= 0xF900 and cp <= 0xFAFF)
            or (cp >= 0x2F800 and cp <= 0x2FA1F)
        ):
            return True

        return False


class TextIteratorStreamer(TextStreamer):
    """
    Streamer that stores print-ready text in a queue, to be used by a downstream application as an iterator. This is
    useful for applications that benefit from accessing the generated text in a non-blocking way (e.g. in an interactive
    Gradio demo).

    Parameters:
        tokenizer (`AutoTokenizer`):
            The tokenizer used to decode the tokens.
        skip_prompt (`bool`, *optional*, defaults to `False`):
            Whether to skip the prompt to `.generate()` or not. Useful e.g. for chatbots.
        timeout (`float`, *optional*):
            The timeout for the text queue. If `None`, the queue will block indefinitely. Useful to handle exceptions
            in `.generate()`, when it is called in a separate thread.
        decode_kwargs (`dict`, *optional*):
            Additional keyword arguments to pass to the tokenizer's `decode` method.

    Examples:

        ```python
        >>> from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
        >>> from threading import Thread

        >>> tok = AutoTokenizer.from_pretrained("openai-community/gpt2")
        >>> model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
        >>> inputs = tok(["An increasing sequence: one,"], return_tensors="pt")
        >>> streamer = TextIteratorStreamer(tok)

        >>> # Run the generation in a separate thread, so that we can fetch the generated text in a non-blocking way.
        >>> generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=20)
        >>> thread = Thread(target=model.generate, kwargs=generation_kwargs)
        >>> thread.start()
        >>> generated_text = ""
        >>> for new_text in streamer:
        ...     generated_text += new_text
        >>> generated_text
        'An increasing sequence: one, two, three, four, five, six, seven, eight, nine, ten, eleven,'
        ```
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        skip_prompt: bool = False,
        timeout: float | None = None,
        **decode_kwargs: Any,
    ):
        super().__init__(tokenizer, skip_prompt, **decode_kwargs)
        self.text_queue = Queue()
        self.stop_signal = None
        self.timeout = timeout

    def on_finalized_text(self, text: str, stream_end: bool = False):
        """Put the new text in the queue. If the stream is ending, also put a stop signal in the queue."""
        self.text_queue.put(text, timeout=self.timeout)
        if stream_end:
            self.text_queue.put(self.stop_signal, timeout=self.timeout)

    def __iter__(self):
        return self

    def __next__(self):
        value = self.text_queue.get(timeout=self.timeout)
        if value == self.stop_signal:
            raise StopIteration()
        else:
            return value


class AsyncTextIteratorStreamer(TextStreamer):
    """
    Streamer that stores print-ready text in a queue, to be used by a downstream application as an async iterator.
    This is useful for applications that benefit from accessing the generated text asynchronously (e.g. in an
    interactive Gradio demo).

    Parameters:
        tokenizer (`AutoTokenizer`):
            The tokenizer used to decode the tokens.
        skip_prompt (`bool`, *optional*, defaults to `False`):
            Whether to skip the prompt to `.generate()` or not. Useful e.g. for chatbots.
        timeout (`float`, *optional*):
            The timeout for the text queue. If `None`, the queue will block indefinitely. Useful to handle exceptions
            in `.generate()`, when it is called in a separate thread.
        decode_kwargs (`dict`, *optional*):
            Additional keyword arguments to pass to the tokenizer's `decode` method.

    Raises:
        TimeoutError: If token generation time exceeds timeout value.

    Examples:

        ```python
        >>> from transformers import AutoModelForCausalLM, AutoTokenizer, AsyncTextIteratorStreamer
        >>> from threading import Thread
        >>> import asyncio

        >>> tok = AutoTokenizer.from_pretrained("openai-community/gpt2")
        >>> model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
        >>> inputs = tok(["An increasing sequence: one,"], return_tensors="pt")

        >>> # Run the generation in a separate thread, so that we can fetch the generated text in a non-blocking way.
        >>> async def main():
        ...     # Important: AsyncTextIteratorStreamer must be initialized inside a coroutine!
        ...     streamer = AsyncTextIteratorStreamer(tok)
        ...     generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=20)
        ...     thread = Thread(target=model.generate, kwargs=generation_kwargs)
        ...     thread.start()
        ...     generated_text = ""
        ...     async for new_text in streamer:
        ...         generated_text += new_text
        >>>     print(generated_text)
        >>> asyncio.run(main())
        An increasing sequence: one, two, three, four, five, six, seven, eight, nine, ten, eleven,
        ```
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        skip_prompt: bool = False,
        timeout: float | None = None,
        **decode_kwargs: Any,
    ):
        super().__init__(tokenizer, skip_prompt, **decode_kwargs)
        self.text_queue = asyncio.Queue()
        self.stop_signal = None
        self.timeout = timeout
        self.loop = asyncio.get_running_loop()

    def on_finalized_text(self, text: str, stream_end: bool = False):
        """Put the new text in the queue. If the stream is ending, also put a stop signal in the queue."""
        self.loop.call_soon_threadsafe(self.text_queue.put_nowait, text)
        if stream_end:
            self.loop.call_soon_threadsafe(self.text_queue.put_nowait, self.stop_signal)

    def __aiter__(self):
        return self

    async def __anext__(self):
        async with asyncio.timeout(self.timeout):
            value = await self.text_queue.get()
        if value == self.stop_signal:
            raise StopAsyncIteration()
        else:
            return value


class TextDiffusionStreamer(TextStreamer):
    """
    Streamer that prints text diffusion outputs. Intermediate diffusion steps (drafts) are temporary
    and overwritten by subsequent drafts, and removed when confirmed text is printed.

    <Tip warning={true}>

    If you're running on an environment like tmux, the draft text may fail to overwrite itself.

    </Tip>


    Parameters:
        tokenizer (`AutoTokenizer`):
            The tokenized used to decode the tokens.
        skip_prompt (`bool`, *optional*, defaults to `False`):
            Whether to skip the prompt to `.generate()` or not. Useful e.g. for chatbots.
        sleep_time (`float`, *optional*):
            Time to sleep between diffusion drafts, which may be helpful to visualize intermediate outputs.
        decode_kwargs (`dict`, *optional*):
            Additional keyword arguments to pass to the tokenizer's `decode` method.

    Examples:

        ```python
        >>> from transformers import DiffusionGemmaForBlockDiffusion, AutoProcessor, TextDiffusionStreamer

        >>> model = DiffusionGemmaForBlockDiffusion.from_pretrained(
        ...     "google/diffusiongemma-26B-A4B-it", device_map="auto",
        ... )
        >>> processor = AutoProcessor.from_pretrained("google/diffusiongemma-26B-A4B-it")

        >>> chat = [{"role": "user", "content": "Why is the sky blue?"},]
        >>> input_ids = processor.apply_chat_template(
        ...     chat, tokenize=True, return_tensors="pt", add_generation_prompt=True
        ... )
        >>> streamer = TextDiffusionStreamer(tokenizer=processor.tokenizer)
        >>> model.generate(input_ids.to(model.device), max_new_tokens=512, streamer=streamer)
        ```
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        skip_prompt: bool = False,
        sleep_time: float | None = None,
        **decode_kwargs: Any,
    ):
        super().__init__(tokenizer, skip_prompt, **decode_kwargs)
        self._has_draft = False
        # `_takes_logits`: Overwrite this attribute if you want your new Streamer class to take the draft
        # logits as an input to `put_draft`. On diffusion models, `logits` can be a very large tensor, so
        # we recommend setting it to `False` by default.
        self._takes_logits = False
        self.sleep_time = sleep_time

    def _clear_draft(self):
        if self._has_draft:
            # Restore cursor and clear to end of screen
            print("\0338\033[J", end="", flush=True)
            self._has_draft = False

    def put_draft(self, value, **kwargs):
        """
        Receives the full sequence of draft tokens, decodes them, and prints them in yellow.
        Overwrites previous draft.
        """
        self._clear_draft()

        if len(value.shape) > 1 and value.shape[0] > 1:
            raise ValueError("TextDiffusionStreamer only supports batch size 1")
        elif len(value.shape) > 1:
            value = value[0]

        text = self.tokenizer.decode(value, **self.decode_kwargs)

        # Save cursor position
        print("\0337", end="", flush=True)
        # Print draft in yellow
        print(f"\033[33m{text}\033[0m", end="", flush=True)
        self._has_draft = True
        if self.sleep_time is not None:
            time.sleep(self.sleep_time)

    def put(self, value):
        """Receives confirmed tokens, clears draft, and prints them permanently."""
        self._clear_draft()
        super().put(value)

    def end(self):
        """Flushes any remaining cache and prints a newline."""
        self._clear_draft()
        super().end()
