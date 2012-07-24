# Authors: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#          Matti Hamalainen <msh@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

import struct
import numpy as np

from .constants import FIFF


def fromfile(fid, dtype=float, count=-1, sep=''):
    if isinstance(fid, file):
        return np.fromfile(fid, dtype, count, sep)
    else:
        if count == -1 or sep != '':
            ValueError('not supported')
        n_bytes = count * np.dtype(dtype).itemsize
        tmp_str = fid.read(n_bytes)
        return np.fromstring(tmp_str, dtype, count, sep)


class Tag(object):
    """Tag in FIF tree structure

    Parameters
    ----------
    kind: int
        Kind of Tag

    type_: int
        Type of Tag

    size: int
        Size in bytes

    int: next
        Position of next Tag

    pos: int
        Position of Tag is the original file.

    """

    def __init__(self, kind, type_, size, next, pos=None):
        self.kind = int(kind)
        self.type = int(type_)
        self.size = int(size)
        self.next = int(next)
        self.pos = pos if pos is not None else next
        self.pos = int(self.pos)
        self.data = None

    def __repr__(self):
        out = "kind: %s - type: %s - size: %s - next: %s - pos: %s" % (
                self.kind, self.type, self.size, self.next, self.pos)
        if hasattr(self, 'data'):
            out += " - data: %s" % self.data
        out += "\n"
        return out

    def __cmp__(self, tag):
        is_equal = (self.kind == tag.kind and
                    self.type == tag.type and
                    self.size == tag.size and
                    self.next == tag.next and
                    self.pos == tag.pos and
                    self.data == tag.data)
        if is_equal:
            return 0
        else:
            return 1


def read_tag_info(fid):
    """Read Tag info (or header)
    """
    s = fid.read(4 * 4)
    tag = Tag(*struct.unpack(">iiii", s))
    if tag.next == 0:
        fid.seek(tag.size, 1)
    elif tag.next > 0:
        fid.seek(tag.next, 0)
    return tag


def read_tag(fid, pos=None):
    """Read a Tag from a file at a given position

    Parameters
    ----------
    fid: file
        The open FIF file descriptor

    pos: int
        The position of the Tag in the file.

    Returns
    -------
    tag: Tag
        The Tag read
    """
    if pos is not None:
        fid.seek(pos, 0)

    s = fid.read(4 * 4)
    tag = Tag(*struct.unpack(">iIii", s))

    #
    #   The magic hexadecimal values
    #
    is_matrix = 4294901760  # ffff0000
    matrix_coding_dense = 16384      # 4000
    matrix_coding_CCS = 16400      # 4010
    matrix_coding_RCS = 16416      # 4020
    data_type = 65535      # ffff
    #
    if tag.size > 0:
        matrix_coding = is_matrix & tag.type
        if matrix_coding != 0:
            matrix_coding = matrix_coding >> 16

            #   Matrices
            if matrix_coding == matrix_coding_dense:
                # Find dimensions and return to the beginning of tag data
                pos = fid.tell()
                fid.seek(tag.size - 4, 1)
                ndim = fromfile(fid, dtype='>i', count=1)
                fid.seek(-(ndim + 1) * 4, 1)
                dims = fromfile(fid, dtype='>i', count=ndim)[::-1]
                #
                # Back to where the data start
                #
                fid.seek(pos, 0)

                if ndim != 2:
                    raise Exception('Only two-dimensional matrices are '
                                     'supported at this time')

                matrix_type = data_type & tag.type

                if matrix_type == FIFF.FIFFT_INT:
                    tag.data = fromfile(fid, dtype='>i',
                                            count=dims.prod()).reshape(dims)
                elif matrix_type == FIFF.FIFFT_JULIAN:
                    tag.data = fromfile(fid, dtype='>i',
                                            count=dims.prod()).reshape(dims)
                elif matrix_type == FIFF.FIFFT_FLOAT:
                    tag.data = fromfile(fid, dtype='>f4',
                                            count=dims.prod()).reshape(dims)
                elif matrix_type == FIFF.FIFFT_DOUBLE:
                    tag.data = fromfile(fid, dtype='>f8',
                                            count=dims.prod()).reshape(dims)
                elif matrix_type == FIFF.FIFFT_COMPLEX_FLOAT:
                    data = fromfile(fid, dtype='>f4', count=2 * dims.prod())
                    # Note: we need the non-conjugate transpose here
                    tag.data = (data[::2] + 1j * data[1::2]).reshape(dims)
                elif matrix_type == FIFF.FIFFT_COMPLEX_DOUBLE:
                    data = fromfile(fid, dtype='>f8', count=2 * dims.prod())
                    # Note: we need the non-conjugate transpose here
                    tag.data = (data[::2] + 1j * data[1::2]).reshape(dims)
                else:
                    raise Exception('Cannot handle matrix of type %d yet' %
                                                                matrix_type)

            elif matrix_coding == matrix_coding_CCS or \
                                    matrix_coding == matrix_coding_RCS:
                from scipy import sparse
                # Find dimensions and return to the beginning of tag data
                pos = fid.tell()
                fid.seek(tag.size - 4, 1)
                ndim = int(fromfile(fid, dtype='>i', count=1))
                fid.seek(-(ndim + 2) * 4, 1)
                dims = fromfile(fid, dtype='>i', count=ndim + 1)
                if ndim != 2:
                    raise Exception('Only two-dimensional matrices are '
                                     'supported at this time')

                # Back to where the data start
                fid.seek(pos, 0)
                nnz = dims[0]
                nrow = dims[1]
                ncol = dims[2]
                sparse_data = fromfile(fid, dtype='>f4', count=nnz)
                shape = (dims[1], dims[2])
                if matrix_coding == matrix_coding_CCS:
                    #    CCS
                    sparse.csc_matrix()
                    sparse_indices = fromfile(fid, dtype='>i4', count=nnz)
                    sparse_ptrs = fromfile(fid, dtype='>i4', count=ncol + 1)
                    tag.data = sparse.csc_matrix((sparse_data, sparse_indices,
                                                 sparse_ptrs), shape=shape)
                else:
                    #    RCS
                    sparse_indices = fromfile(fid, dtype='>i4', count=nnz)
                    sparse_ptrs = fromfile(fid, dtype='>i4', count=nrow + 1)
                    tag.data = sparse.csr_matrix((sparse_data, sparse_indices,
                                                 sparse_ptrs), shape=shape)
            else:
                raise Exception('Cannot handle other than dense or sparse '
                                 'matrices yet')
        else:
            #   All other data types

            #   Simple types
            if tag.type == FIFF.FIFFT_BYTE:
                tag.data = fromfile(fid, dtype=">B1", count=tag.size)
            elif tag.type == FIFF.FIFFT_SHORT:
                tag.data = fromfile(fid, dtype=">h2", count=tag.size / 2)
            elif tag.type == FIFF.FIFFT_INT:
                tag.data = fromfile(fid, dtype=">i4", count=tag.size / 4)
            elif tag.type == FIFF.FIFFT_USHORT:
                tag.data = fromfile(fid, dtype=">H2", count=tag.size / 2)
            elif tag.type == FIFF.FIFFT_UINT:
                tag.data = fromfile(fid, dtype=">I4", count=tag.size / 4)
            elif tag.type == FIFF.FIFFT_FLOAT:
                tag.data = fromfile(fid, dtype=">f4", count=tag.size / 4)
            elif tag.type == FIFF.FIFFT_DOUBLE:
                tag.data = fromfile(fid, dtype=">f8", count=tag.size / 8)
            elif tag.type == FIFF.FIFFT_STRING:
                tag.data = fromfile(fid, dtype=">c", count=tag.size)
                tag.data = ''.join(tag.data)
            elif tag.type == FIFF.FIFFT_DAU_PACK16:
                tag.data = fromfile(fid, dtype=">h2", count=tag.size / 2)
            elif tag.type == FIFF.FIFFT_COMPLEX_FLOAT:
                tag.data = fromfile(fid, dtype=">f4", count=tag.size / 4)
                tag.data = tag.data[::2] + 1j * tag.data[1::2]
            elif tag.type == FIFF.FIFFT_COMPLEX_DOUBLE:
                tag.data = fromfile(fid, dtype=">f8", count=tag.size / 8)
                tag.data = tag.data[::2] + 1j * tag.data[1::2]
            #
            #   Structures
            #
            elif tag.type == FIFF.FIFFT_ID_STRUCT:
                tag.data = dict()
                tag.data['version'] = int(fromfile(fid, dtype=">i4",
                                                      count=1))
                tag.data['version'] = int(fromfile(fid, dtype=">i4",
                                                      count=1))
                tag.data['machid'] = fromfile(fid, dtype=">i4", count=2)
                tag.data['secs'] = int(fromfile(fid, dtype=">i4", count=1))
                tag.data['usecs'] = int(fromfile(fid, dtype=">i4", count=1))
            elif tag.type == FIFF.FIFFT_DIG_POINT_STRUCT:
                tag.data = dict()
                tag.data['kind'] = int(fromfile(fid, dtype=">i4", count=1))
                tag.data['ident'] = int(fromfile(fid, dtype=">i4", count=1))
                tag.data['r'] = fromfile(fid, dtype=">f4", count=3)
                tag.data['coord_frame'] = 0
            elif tag.type == FIFF.FIFFT_COORD_TRANS_STRUCT:
                tag.data = dict()
                tag.data['from'] = int(fromfile(fid, dtype=">i4", count=1))
                tag.data['to'] = int(fromfile(fid, dtype=">i4", count=1))
                rot = fromfile(fid, dtype=">f4", count=9).reshape(3, 3)
                move = fromfile(fid, dtype=">f4", count=3)
                tag.data['trans'] = np.r_[np.c_[rot, move],
                                           np.array([[0], [0], [0], [1]]).T]
                #
                # Skip over the inverse transformation
                # It is easier to just use inverse of trans in Matlab
                #
                fid.seek(12 * 4, 1)
            elif tag.type == FIFF.FIFFT_CH_INFO_STRUCT:
                d = dict()
                d['scanno'] = int(fromfile(fid, dtype=">i4", count=1))
                d['logno'] = int(fromfile(fid, dtype=">i4", count=1))
                d['kind'] = int(fromfile(fid, dtype=">i4", count=1))
                d['range'] = float(fromfile(fid, dtype=">f4", count=1))
                d['cal'] = float(fromfile(fid, dtype=">f4", count=1))
                d['coil_type'] = int(fromfile(fid, dtype=">i4", count=1))
                #
                #   Read the coil coordinate system definition
                #
                d['loc'] = fromfile(fid, dtype=">f4", count=12)
                d['coil_trans'] = None
                d['eeg_loc'] = None
                d['coord_frame'] = FIFF.FIFFV_COORD_UNKNOWN
                tag.data = d
                #
                #   Convert loc into a more useful format
                #
                loc = tag.data['loc']
                kind = tag.data['kind']
                if kind == FIFF.FIFFV_MEG_CH or kind == FIFF.FIFFV_REF_MEG_CH:
                    tag.data['coil_trans'] = np.r_[np.c_[loc[3:5], loc[6:8],
                                                        loc[9:11], loc[0:2]],
                                        np.array([0, 0, 0, 1]).reshape(1, 4)]
                    tag.data['coord_frame'] = FIFF.FIFFV_COORD_DEVICE
                elif tag.data['kind'] == FIFF.FIFFV_EEG_CH:
                    if np.linalg.norm(loc[3:5]) > 0:
                        tag.data['eeg_loc'] = np.c_[loc[0:2], loc[3:5]]
                    else:
                        tag.data['eeg_loc'] = loc[1:3]
                    tag.data['coord_frame'] = FIFF.FIFFV_COORD_HEAD
                #
                #   Unit and exponent
                #
                tag.data['unit'] = int(fromfile(fid, dtype=">i4", count=1))
                tag.data['unit_mul'] = int(fromfile(fid, dtype=">i4",
                                           count=1))
                #
                #   Handle the channel name
                #
                ch_name = fromfile(fid, dtype=">c", count=16)
                #
                # Omit nulls
                #
                tag.data['ch_name'] = ''.join(
                                    ch_name[:np.where(ch_name == '')[0][0]])

            elif tag.type == FIFF.FIFFT_OLD_PACK:
                offset = float(fromfile(fid, dtype=">f4", count=1))
                scale = float(fromfile(fid, dtype=">f4", count=1))
                tag.data = fromfile(fid, dtype=">h2",
                                       count=(tag.size - 8) / 2)
                tag.data = scale * tag.data + offset
            elif tag.type == FIFF.FIFFT_DIR_ENTRY_STRUCT:
                tag.data = list()
                for _ in range(tag.size / 16 - 1):
                    s = fid.read(4 * 4)
                    tag.data.append(Tag(*struct.unpack(">iIii", s)))
            else:
                raise Exception('Unimplemented tag data type %s' % tag.type)

    if tag.next != FIFF.FIFFV_NEXT_SEQ:
        # f.seek(tag.next,0)
        fid.seek(tag.next, 1)  # XXX : fix? pb when tag.next < 0

    return tag


def find_tag(fid, node, findkind):
    """Find Tag in an open FIF file descriptor
    """
    for p in range(node['nent']):
        if node['directory'][p].kind == findkind:
            return read_tag(fid, node['directory'][p].pos)
    tag = None
    return tag


def has_tag(node, kind):
    """Does the node contains a Tag of a given kind?
    """
    for d in node['directory']:
        if d.kind == kind:
            return True
    return False
