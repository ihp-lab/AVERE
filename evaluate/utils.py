import glob, os, tqdm, re, json

def strip_end_tag(text: str) -> str:
    """
    If `text` ends with the literal substring '</s>', remove it and return
    the shortened string; otherwise return `text` unchanged.
    """
    return text[:-4] if text.endswith("</s>") else text

def strip_trailing_unk(text: str) -> str:
    #  r'(?:<unk>\s*)+$'  ←  one or more "<unk>" tokens followed by optional spaces, anchored to the end
    return re.sub(r'(?:<unk>\s*)+$', '', text).rstrip()

def chunk_path_list(paths, batch_size):
    """
    Split *paths* into sub-lists of length *batch_size* (the last one may be shorter).

    Example:
        >>> chunk(['a', 'b', 'c', 'd', 'e'], 2)
        [['a', 'b'], ['c', 'd'], ['e']]
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")

    return [paths[i : i + batch_size]          # grab slices of size batch_size
            for i in range(0, len(paths), batch_size)]