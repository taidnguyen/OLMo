"""
Use this to prepare a numpy memory-mapped language modeling dataset from raw *.json.gz
dataset files, such as those from c4. Each file is expected to be a gzipped JSON lines
file, which each JSON line has a field named "text" that is a string representing a single
document from the dataset.

To test out this script, run:

```bash
python scripts/prepare_memmap_dataset.py test_fixtures/*.json.gz -o /tmp/out.npy
```
"""

import concurrent.futures
import functools
import logging
import multiprocessing as mp
import os
from concurrent.futures import Future
from contextlib import ExitStack
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Generator, List, Optional, Sequence, Tuple, TypeVar

import click
import msgspec
import numpy as np
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TaskProgressColumn,
    TimeElapsedColumn,
)
from smashed.utils.io_utils import (
    MultiPath,
    decompress_stream,
    open_file_for_write,
    recursively_list_files,
    stream_file_for_read,
)

from olmo import Tokenizer
from olmo.util import prepare_cli_environment

log = logging.getLogger(__name__)

T = TypeVar("T", bound=Sequence)


def get_progress() -> Progress:
    return Progress(
        "[progress.description]{task.description}",
        MofNCompleteColumn(),
        "files",
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
    )


class InputDocumentSpec(msgspec.Struct):
    # almost 5x faster than built-in json decoding in my tests;
    # can work with approximate spec (i.e., ignore missing fields)
    text: str


def tokenize_file(tokenizer: Tokenizer, path: str, batch_size: int = 1_000) -> Generator[List[int], None, None]:
    decoder = msgspec.json.Decoder(InputDocumentSpec)

    with ExitStack() as stack:
        input_file = stack.enter_context(stream_file_for_read(path, mode="rb"))
        input_stream = stack.enter_context(decompress_stream(input_file, mode="rt"))

        for line in input_stream:
            row = decoder.decode(line)
            yield tokenizer.encode(row.text, add_special_tokens=True)


class MemmapFile:
    """Context manager responsible for writing, resizing, and closing / uploading a memmap file."""

    DEFAULT_2G_MAX_TOKENS = 2 * 1024 * 1024 * 1024  # 2B tokens

    def __init__(
        self,
        path: str,
        dtype: np.dtype,
        max_tokens: int = DEFAULT_2G_MAX_TOKENS,
    ):
        self.path = MultiPath.parse(path)
        self.dtype = dtype
        self.max_tokens = max_tokens

        self._local_path: Optional[Path] = None
        self._written_tokens = 0
        self._memmap: Optional[np.memmap] = None

    def __len__(self) -> int:
        return self._written_tokens

    def write(self, values: List[int], flush: bool = False) -> Optional[List[int]]:
        if self._memmap is None:
            raise RuntimeError("MemmapFile is not open")

        if (len(values) + self._written_tokens) >= self.max_tokens:
            values = values[: self.max_tokens - self._written_tokens]
            rest = values[self.max_tokens - self._written_tokens :]
        else:
            rest = None

        self._memmap[self._written_tokens : self._written_tokens + len(values)] = values
        self._written_tokens += len(values)

        if flush:
            self._memmap.flush()

        return rest

    def __enter__(self) -> "MemmapFile":
        assert self._memmap is None, "MemmapFile is already open"

        if self.path.is_local:
            self._local_path = self.path.as_path
            # make sure the directory exists
            self._local_path.parent.mkdir(parents=True, exist_ok=True)  # type: ignore
        else:
            with NamedTemporaryFile(delete=False) as f:
                # if the destination for the memmap is not local, we need to write to a temporary file first
                self._local_path = Path(f.name)

        self._memmap = np.memmap(mode="w+", filename=self._local_path, dtype=self.dtype, shape=(self.max_tokens,))
        log.info(f"Created memmap file at {self._local_path} of size {self._memmap.nbytes:,} bytes")

        return self

    def __exit__(self, *_):
        return self.close()

    def close(self):
        assert self._local_path is not None, "MemmapFile is not open"
        assert self._memmap is not None, "MemmapFile is not open"

        # write the memmap to the destination
        self._memmap.flush()

        # we resize the memmap to the number of tokens actually written
        if self._written_tokens < self.max_tokens:
            del self._memmap
            os.rename(self._local_path, (temp_path := self._local_path.with_suffix(".tmp")))

            new_memmap = np.memmap(
                mode="w+", filename=self._local_path, dtype=self.dtype, shape=(self._written_tokens,)
            )
            old_memmap = np.memmap(mode="r", filename=temp_path, dtype=self.dtype, shape=(self.max_tokens,))
            new_memmap[:] = old_memmap[: self._written_tokens]
            new_memmap.flush()
            log.info(f"Resized memmap file from {old_memmap.nbytes:,} to {new_memmap.nbytes:,} bytes")
            os.remove(temp_path)

        if not self.path.is_local:
            with ExitStack() as stack:
                f = stack.enter_context(stream_file_for_read(self._local_path, "rb"))
                g = stack.enter_context(open_file_for_write(self.path, mode="wb"))
                g.write(f.read())
            log.info(f"Written memmap file to {self.path.as_str}")

            # delete the temporary file
            os.remove(self._local_path)

        self._local_path = self._memmap = None


def fill_memmap(
    tokenizer_id: str,
    path: str,
    memmap_path: str,
    dtype: np.dtype,
    max_tokens: int = 2 * 1024 * 1024 * 1024,  # 2B tokens * 2 bytes per token (uint16) = 4GB
):
    # we need to make a new tokenizer here because it's not pickleable
    tokenizer = Tokenizer.from_pretrained(tokenizer_id, truncate_to=None)

    # first memmap file will be created in the loop below
    memmap: Optional[MemmapFile] = None

    file_index = 0

    with ExitStack() as stack:
        # `tokenize_file` is a generator that yields tokenized documents
        for token_ids in tokenize_file(tokenizer=tokenizer, path=path):
            # if token_ids_to_still_write is not None it means that either memmap is None or it's full,
            # so we will need to create a new one later
            token_ids_to_still_write = memmap.write(token_ids) if memmap is not None else token_ids

            if token_ids_to_still_write is not None:
                # close the previous memmap (if one is open)
                stack.pop_all().close()

                # create a new memmap file; progressively name them with an index
                curr_memmap_path = f"{memmap_path}_{file_index:05d}.npy"
                memmap = stack.enter_context(MemmapFile(path=curr_memmap_path, dtype=dtype, max_tokens=max_tokens))

                # increment the file index and reset the tokens index
                file_index += 1

                # do the actual writing
                memmap.write(token_ids_to_still_write)

        # close the last memmap
        stack.pop_all().close()


def make_source_and_target(src: Tuple[str, ...], output: str) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    exploded_src: List[str] = []
    exploded_dst: List[str] = []

    parsed_output = MultiPath.parse(output)
    for prefix in src:
        parsed_prefix = MultiPath.parse(prefix)
        for path in recursively_list_files(parsed_prefix):
            exploded_src.append(path)
            exploded_dst.append((parsed_output / MultiPath.parse(path) - parsed_prefix).as_str.replace(".", "_"))

    return tuple(sorted(exploded_src)), tuple(sorted(exploded_dst))


@click.command()
@click.argument(
    "src",
    nargs=-1,
    type=str,
    required=True,
)
@click.option(
    "-o",
    "--output",
    type=str,
    help="Specify the output path.",
    prompt="Output directory",
)
@click.option(
    "--tokenizer", "tokenizer_id", type=str, help="Name of path of a pretrained tokenizer", default="gpt2"
)
@click.option("--dtype", "dtype_str", default="uint16")
@click.option("--validate/--no-validate", default=False)
@click.option(
    "--max-tokens",
    default=2 * 1024 * 1024 * 1024,
    type=int,
    help="Maximum number of tokens to store in a single memmap file (default: 2B tokens or 4GB)",
)
@click.option("--debug/--no-debug", default=False, help="Enable debug (single process mode)")
@click.option("-j", "--workers", "max_workers", type=int, default=None, help="Defaults to number of CPUs")
def main(
    src: Tuple[str, ...],
    output: str,
    tokenizer_id: str,
    dtype_str: str,
    validate: bool,
    max_tokens: int,
    debug: bool,
    max_workers: Optional[int] = None,
):
    dtype = np.dtype(dtype_str)
    src, dst = make_source_and_target(src=src, output=output)

    # creating a partial here with all the arguments we need to pass to fill_memmap except for the paths
    # so that we don't make mistakes between debug and non-debug mode
    fill_memmap_fn = functools.partial(fill_memmap, tokenizer_id=tokenizer_id, dtype=dtype, max_tokens=max_tokens)

    if debug:
        log.info("Running in debug mode. Only one process will be used.")
        for src_path, dst_path in zip(src, dst):
            fill_memmap_fn(path=src_path, memmap_path=dst_path)
        return

    # Now tokenizer all documents again and populate the memmap array. We do this in parallel.
    workers_cnt = min(max_workers or os.cpu_count() or 1, len(src))
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers_cnt) as executor:
        futures: List[Future[None]] = []
        for src_path, dst_path in zip(src, dst):
            future = executor.submit(fill_memmap_fn, path=src_path, memmap_path=dst_path)
            futures.append(future)
        with get_progress() as progress:
            for future in progress.track(
                concurrent.futures.as_completed(futures),
                description="Filling memmap arrays...",
                total=len(futures),
            ):
                future.result()

    log.info(f"Done! File written to {output}")

    # if validate:
    #     log.info("Validating...")
    #     tokenizer = Tokenizer.from_pretrained(tokenizer_id, truncate_to=None)
    #     memmap = np.memmap(output, mode="r", dtype=dtype, shape=(total_tokens,))
    #     # Should have an EOS token for every document.
    #     assert (memmap == tokenizer.eos_token_id).sum() == total_docs
    #     assert memmap[-1] == tokenizer.eos_token_id
    #     # Make sure all entries have been filled with actual token IDs.
    #     assert (memmap < tokenizer.vocab_size).all()
    #     log.info("All good!")


if __name__ == "__main__":
    mp.set_start_method("spawn")
    prepare_cli_environment()
    main()
