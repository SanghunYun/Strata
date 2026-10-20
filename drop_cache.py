#!/usr/bin/env python3
"""Drop the page cache of the model files before a run (no root needed): posix_fadvise(DONTNEED) on every file.

  python3 drop_cache.py <pack dir> <gguf dir> ...
"""
import os, sys
from pathlib import Path
n = 0
for d in sys.argv[1:]:
    for f in (Path(d).rglob('*') if Path(d).is_dir() else [Path(d)]):
        if f.is_file():
            fd = os.open(f, os.O_RDONLY)
            try: os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); n += 1
            finally: os.close(fd)
print(f'page cache dropped for {n} files')
