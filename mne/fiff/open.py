# Authors: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#          Matti Hamalainen <msh@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

from .tag import read_tag_info, read_tag
from .tree import make_dir_tree
from .constants import FIFF


def allopen(fname, *args, **kwargs):
    ''' Generic file-like object open

    If input ``fname`` already looks like a file, pass through.
    If ``fname`` ends with recognizable compressed types, use python
    libraries to open as file-like objects (read or write)
    Otherwise, use standard ``open``.

    Copied from dipy
    '''
    if hasattr(fname, 'write'):
        return fname
    if args:
        mode = args[0]
    elif 'mode' in kwargs:
        mode = kwargs['mode']
    else:
        mode = 'rb'
    if fname.endswith('.gz'):
        if ('w' in mode and
            len(args) < 2 and
            not 'compresslevel' in kwargs):
            kwargs['compresslevel'] = default_compresslevel
        import gzip
        opener = gzip.open
    elif fname.endswith('.bz2'):
        if ('w' in mode and
            len(args) < 3 and
            not 'compresslevel' in kwargs):
            kwargs['compresslevel'] = default_compresslevel
        import bz2
        opener = bz2.BZ2File
    else:
        opener = open
    return opener(fname, *args, **kwargs)


def fiff_open(fname, verbose=False):
    """Open a FIF file.

    Parameters
    ----------
    fname: string
        name of the fif file

    verbose: bool
        verbose mode if True

    Returns
    -------
    fid: file
        The file descriptor of the open file

    tree: fif tree
        The tree is a complex structure filled with dictionaries,
        lists and tags.

    directory: list
        list of nodes.

    """
    fid = allopen(fname, "rb")  # Open in binary mode

    tag = read_tag_info(fid)

    #   Check that this looks like a fif file
    if tag.kind != FIFF.FIFF_FILE_ID:
        raise ValueError('file does not start with a file id tag')

    if tag.type != FIFF.FIFFT_ID_STRUCT:
        raise ValueError('file does not start with a file id tag')

    if tag.size != 20:
        raise ValueError('file does not start with a file id tag')

    tag = read_tag(fid)

    if tag.kind != FIFF.FIFF_DIR_POINTER:
        raise ValueError('file does have a directory pointer')

    #   Read or create the directory tree
    if verbose:
        print '    Creating tag directory for %s...' % fname

    dirpos = int(tag.data)
    if dirpos > 0:
        tag = read_tag(fid, dirpos)
        directory = tag.data
    else:
        fid.seek(0, 0)
        directory = list()
        while tag.next >= 0:
            pos = fid.tell()
            tag = read_tag_info(fid)
            tag.pos = pos
            directory.append(tag)

    tree, _ = make_dir_tree(fid, directory, verbose=verbose)

    if verbose:
        print '[done]'

    #   Back to the beginning
    fid.seek(0)

    return fid, tree, directory
