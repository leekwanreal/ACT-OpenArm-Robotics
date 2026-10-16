# Copyright 2026 Enactic, Inc.
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

"""Turn a recorded episode into its final, compressed file.

The recorder writes the images uncompressed, because compressing one sample
takes longer than a 50 Hz step; this compresses them afterwards, in the
background, into exactly what the recorder used to write:

    <episode>.partial.hdf5   uncompressed, complete = False   (recorder)
    <episode>.hdf5           images lzf, complete = True      (this)

The partial file is removed only once the final one is in place, so an
interrupted compression loses nothing; run it again by hand:

    python3 -m openarm_ui.episode_compress dataset/<day>/<episode>.partial.hdf5
"""

import os
import sys
from pathlib import Path

import h5py
import numpy as np

# Every camera's frames: <camera>_rgb and <camera>_depth, one or two cameras.
IMAGE_GROUP = "observations/images/"
# Frames compressed per read/write: a few hundred MB uncompressed at most.
BLOCK = 100


def final_path_for(partial):
    partial = Path(partial)
    name = partial.name
    if not name.endswith(".partial.hdf5"):
        raise ValueError(f"{partial} is not a recorder's .partial.hdf5")
    return partial.with_name(name[: -len(".partial.hdf5")] + ".hdf5")


def _copy(source, target, rows):
    """Every group, attribute and the first `rows` of every dataset."""
    target.attrs.update(dict(source.attrs))
    for name, item in source.items():
        if isinstance(item, h5py.Group):
            _copy(item, target.create_group(name), rows)
            continue
        path = item.name.lstrip("/")
        image = path.startswith(IMAGE_GROUP)
        shape = (rows,) + item.shape[1:]
        dataset = target.create_dataset(
            name, shape=shape, dtype=item.dtype,
            maxshape=(None,) + item.shape[1:],
            chunks=((1,) + item.shape[1:]) if image else item.chunks,
            compression="lzf" if image else None)
        dataset.attrs.update(dict(item.attrs))
        step = BLOCK if image else max(1, rows)
        for start in range(0, rows, step):
            stop = min(rows, start + step)
            dataset[start:stop] = item[start:stop]


def _rows(source):
    """Samples in a partial file. A recorder that was killed never wrote the
    count, and its datasets run on past the last sample with zeros."""
    stamps = source["timestamp_ns"][:]
    written = int(np.count_nonzero(stamps))
    return int(source.attrs.get("samples", written))


def compress(partial, final=None):
    """Write `final` (images lzf, complete) from `partial`, then drop it."""
    partial = Path(partial)
    final = Path(final) if final else final_path_for(partial)
    # Not *.hdf5 until it is whole, so nothing reads it half written.
    work = final.with_name(final.name + ".compressing")
    with h5py.File(partial, "r") as source, h5py.File(work, "w", libver="latest") as target:
        rows = _rows(source)
        if not rows:
            raise ValueError(f"{partial} holds no samples")
        _copy(source, target, rows)
        if "samples" not in source.attrs:
            stamps = source["timestamp_ns"][:rows]
            target.attrs["samples"] = rows
            target.attrs["duration_seconds"] = float(stamps[-1] - stamps[0]) / 1e9
        target.attrs["complete"] = True
    os.replace(work, final)
    partial.unlink()
    return final


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) not in (1, 2):
        print(__doc__, file=sys.stderr)
        return 2
    try:
        os.nice(10)             # after a recording, never in its way
    except OSError:
        pass
    try:
        final = compress(*argv)
    except (OSError, ValueError) as error:
        print(f"[episode_compress] {error}", file=sys.stderr)
        return 1
    print(f"[episode_compress] {final}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
