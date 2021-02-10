from typing import Union, List

from pathlib import Path


def glob_all_video_paths(
        root: Union[str, Path],
        recursive: bool = False,
) -> List[Path]:
    suffixes = [
        '*.mov',
        '*.MOV',
        '*.mp4',
        '*.MP4'
    ]

    if recursive:
        return [
            p
            for suff in suffixes
            for p in Path(root).rglob(suff)
        ]
    else:
        return [
            p
            for suff in suffixes
            for p in Path(root).glob(suff)
        ]
