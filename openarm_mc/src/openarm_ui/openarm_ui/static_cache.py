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

"""The page's files, held in memory, gzipped, with what a browser caches by.

Nearly all of what the dashboard downloads is the robot's visual meshes:
about 70 MB of COLLADA, which is XML and gzips to under a third of that.
Reading and compressing them on every request made a page load take tens
of seconds, and since nothing told the browser what it already had, a
reload paid it all again. So each file is read and compressed once, kept
until it changes on disk, and served with an ETag the browser can send
back for a 304 instead of the bytes.
"""

import gzip
import hashlib
import mimetypes
import os
import threading
from dataclasses import dataclass
from typing import Optional

# Worth gzipping: text, which the meshes are. Images and the like are
# compressed already.
COMPRESSIBLE = {
    ".dae", ".urdf", ".xml", ".stl", ".obj", ".mtl",
    ".html", ".js", ".mjs", ".css", ".json", ".svg", ".txt", ".md",
}
# Below this, gzip's headers cost more than they save.
MIN_COMPRESS_BYTES = 1024
# Level 6 is gzip's default: within a few percent of the smallest output at
# a fraction of level 9's time, which matters on the first request.
GZIP_LEVEL = 6

for _type, _ext in (("text/javascript", ".mjs"), ("model/vnd.collada+xml", ".dae"),
                    ("application/xml", ".urdf"), ("model/stl", ".stl"),
                    ("model/obj", ".obj")):
    mimetypes.add_type(_type, _ext)


@dataclass(frozen=True)
class StaticFile:
    # Only one of the two is kept: the gzipped bytes when compressing paid,
    # the plain ones otherwise. Every browser takes gzip, so holding both
    # would double the memory for a client that does not exist.
    plain: Optional[bytes]
    gzipped: Optional[bytes]
    etag: str
    last_modified: float
    content_type: str

    def body(self):
        """The uncompressed bytes, for a client that does not take gzip."""
        return self.plain if self.plain is not None else gzip.decompress(self.gzipped)


class StaticCache:
    """Path -> StaticFile, rebuilt only when the file's mtime or size changes."""

    def __init__(self):
        self._files = {}
        self._lock = threading.Lock()

    def get(self, path):
        """The file at `path`, from memory when it has not changed on disk.

        Blocking (it may read and compress); call it off the event loop.
        """
        stat = os.stat(path)
        key = (stat.st_mtime_ns, stat.st_size)
        with self._lock:
            cached = self._files.get(path)
        if cached is not None and cached[0] == key:
            return cached[1]

        with open(path, "rb") as handle:
            body = handle.read()
        gzipped = None
        if (os.path.splitext(path)[1].lower() in COMPRESSIBLE
                and len(body) >= MIN_COMPRESS_BYTES):
            packed = gzip.compress(body, GZIP_LEVEL, mtime=0)
            if len(packed) < len(body):
                gzipped = packed
        kind, _ = mimetypes.guess_type(path)
        entry = StaticFile(
            plain=None if gzipped is not None else body,
            gzipped=gzipped,
            etag='"%s"' % hashlib.blake2b(body, digest_size=12).hexdigest(),
            last_modified=stat.st_mtime,
            content_type=kind or "application/octet-stream",
        )
        with self._lock:
            self._files[path] = (key, entry)
        return entry

    def warm(self, root, suffixes=(".dae", ".urdf", ".stl")):
        """Load and compress the heavy files ahead of the first page."""
        count = 0
        for directory, _dirs, names in os.walk(root, followlinks=True):
            for name in names:
                if os.path.splitext(name)[1].lower() in suffixes:
                    try:
                        self.get(os.path.join(directory, name))
                        count += 1
                    except OSError:
                        pass
        return count

    def stats(self):
        with self._lock:
            entries = [entry for _key, entry in self._files.values()]
        held = sum(len(e.gzipped if e.gzipped is not None else e.plain)
                   for e in entries)
        return len(entries), held
