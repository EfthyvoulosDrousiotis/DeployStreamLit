from discretesampling.base.util import pad, restore
from mpi4py import MPI
import numpy as np
from math import log2



def variable_size_redistribution(particles, ncopies, exec):
    x = pad(particles, exec)

    x = fixed_size_redistribution(x, ncopies)

    particles = restore(x, particles)

    return particles



def fixed_size_redistribution(x, ncopies):

    if MPI.COMM_WORLD.Get_size() > 1:
        x, ncopies = rot_nearly_sort(x, ncopies)
        x, ncopies = rot_split(x, ncopies)

    x = sequential_redistribution(x, ncopies)

    return x

def rot_nearly_sort(x, ncopies):
    comm = MPI.COMM_WORLD
    loc_n = len(ncopies)
    P = comm.Get_size()
    N = loc_n * P
    rank = comm.Get_rank()
    base = rank * loc_n  # In the paper is named n*p

    x, ncopies, zeros = sequential_nearly_sort(x, ncopies)

    shifts = np.zeros_like(zeros)
    shifts_MPI_dtpe = MPI._typedict[zeros.dtype.char]
    comm.Exscan(sendbuf=[zeros, shifts_MPI_dtpe], recvbuf=[shifts, shifts_MPI_dtpe], op=MPI.SUM)

    # Compute the MSB to check (top) and the LSB to check (down)
    down = max(loc_n, 1)
    max_shifts = np.zeros_like(shifts)
    comm.Allreduce(sendbuf=[shifts if rank == P-1 else np.array(0), shifts_MPI_dtpe], recvbuf=[max_shifts, shifts_MPI_dtpe], op=MPI.SUM)
    top = 1 if max_shifts == 0 else 1 << int(log2(max_shifts))  # to fix math domain error on for loop, when top is 0

    if loc_n > 1:
        dist = 1
        send_partner = (rank - dist) & (P - 1)  # This only works if N and P are both powers of 2
        recv_partner = (rank + dist) & (P - 1)  # This only works if N and P are both powers of 2
        lsb = shifts & (down - 1) > 0

        ncopies_to_send, x_to_send, starter_to_send, ncopies, x, starter = divide_and_rotate(x, ncopies, shifts, down-1)

        ncopies_recv, x_recv, starter_recv = rotate(ncopies_to_send, x_to_send, starter_to_send, send_partner,
                                                    recv_partner)

        x, ncopies = accept(x, ncopies, x_recv, ncopies_recv, np.repeat(1, loc_n))
        starter = starter_recv if lsb == 1 and recv_partner > rank else starter
        shifts = starter

    # Iterate from the LSB to the MSB
    for k in 2 ** np.array(range(int(log2(down)), int(log2(top)) + 1)):
        dist = int(k / loc_n)
        send_partner = (rank - dist) & (P - 1)  # This only works if N and P are both powers of 2
        recv_partner = (rank + dist) & (P - 1)
        lsb = (shifts >> int(log2(k))) & 1

        ncopies_to_send, x_to_send, starter_to_send, ncopies, x, starter = divide(ncopies, x, shifts, k)

        ncopies_recv, x_recv, starter_recv = rotate(ncopies_to_send, x_to_send, starter_to_send, send_partner,
                                                    recv_partner)

        x, ncopies = accept(x, ncopies, x_recv, ncopies_recv, np.repeat(lsb == 0, loc_n))
        starter = starter_recv if lsb == 1 and recv_partner > rank else starter
        shifts = starter

    return x, ncopies


def rot_split(x, ncopies):
    comm = MPI.COMM_WORLD
    loc_n = len(ncopies)
    P = comm.Get_size()
    N = loc_n*P
    rank = comm.Get_rank()
    base = rank*loc_n  # In the paper is named n*p

    # Initialise the cumulative sum of ncopies, and trunk to zero those elements i for which ncopies[i] = 0
    csum = inclusive_prefix_sum(ncopies)*(ncopies > 0)
    starter = csum[0] - ncopies[0]

    # Compute the MSB to check (top) and the LSB to check (down)
    down = max(loc_n, 1)  # loc_n if loc_n > 1 else 1
    max_bit = np.max((ncopies > 0)*(csum - np.array(range(loc_n)) - base))
    top = np.zeros_like(max_bit)
    max_bit_MPI_dtype = MPI._typedict[max_bit.dtype.char]
    comm.Allreduce(sendbuf=[max_bit, max_bit_MPI_dtype], recvbuf=[top, max_bit_MPI_dtype], op=MPI.MAX)
    top = top >> 1 if (top & (~top + 1)) == top else 1 << int(log2(top))  # top & (~top + 1) is the MSB
    top = max(top, 1)  # to fix math domain error on the next line, when the previous returns 0

    # Iterate from the MSB to the LSB
    for k in 2**np.array(range(int(log2(top)), int(log2(down))-1, -1)):
        # Compute the MPI ranks to send to and receive from
        dist = int(k/loc_n)  # distance (in MPI ranks) to/from which send/receive
        send_partner = (rank + dist) & (P - 1)  # This only works if N and P are both powers of 2
        recv_partner = (rank - dist) & (P - 1)  # This only works if N and P are both powers of 2

        ncopies_to_send, x_to_send, starter_to_send, ncopies, x, starter = divide(ncopies, x, csum, starter, k)

        ncopies_recv, x_recv, starter_recv = rotate(ncopies_to_send, x_to_send, starter_to_send, send_partner, recv_partner)

        x, ncopies = accept(x, ncopies, x_recv, ncopies_recv, ncopies > 0)

        starter = starter_recv if (starter_recv > 0 and rank > recv_partner) else starter
        csum = starter + np.cumsum(ncopies)

    # Leaf stage of the binary tree
    if loc_n > 1:
        dist = 1
        send_partner = (rank + dist) & (P - 1)  # This only works if N and P are both powers of 2
        recv_partner = (rank - dist) & (P - 1)  # This only works if N and P are both powers of 2

        ncopies_to_send, x_to_send, ncopies, x = divide_and_rotate(x, ncopies, csum)

        ncopies_recv, x_recv, _ = rotate(ncopies_to_send, x_to_send, np.array([0]), send_partner, recv_partner)

        x, ncopies = accept(x, ncopies, x_recv, ncopies_recv, ncopies > 0)

    return x, ncopies

def rotate(ncopies, x, starter, send_partner, recv_partner):
    comm = MPI.COMM_WORLD
    temp_ncopies = np.zeros_like(ncopies)
    temp_x = np.zeros_like(x)
    temp_starter = np.zeros_like(starter)
    ncopies_MPI_dtype = MPI._typedict[ncopies.dtype.char]
    x_MPI_dtype = MPI._typedict[x.dtype.char]
    starter_MPI_dtype = MPI._typedict[starter.dtype.char]

    comm.Sendrecv(sendbuf=[ncopies, ncopies_MPI_dtype], dest=send_partner, sendtag=0,
                  recvbuf=[temp_ncopies, ncopies_MPI_dtype], source=recv_partner, recvtag=0)
    comm.Sendrecv(sendbuf=[x, x_MPI_dtype], dest=send_partner, sendtag=0,
                  recvbuf=[temp_x, x_MPI_dtype], source=recv_partner, recvtag=0)
    comm.Sendrecv(sendbuf=[starter, starter_MPI_dtype], dest=send_partner, sendtag=0,
                  recvbuf=[temp_starter, starter_MPI_dtype], source=recv_partner, recvtag=0)

    return temp_ncopies, temp_x, temp_starter

def accept(x, ncopies, x_recv, ncopies_recv, mask):
    return x*np.atleast_2d(mask).transpose()+x_recv, mask*ncopies+ncopies_recv

def inclusive_prefix_sum(array):
    comm = MPI.COMM_WORLD

    csum = np.cumsum(array).astype(array.dtype)
    offset = np.zeros(1, dtype=array.dtype)
    MPI_dtype = MPI._typedict[array.dtype.char]
    comm.Exscan(sendbuf=[csum[-1], MPI_dtype], recvbuf=[offset, MPI_dtype], op=MPI.SUM)

    return csum + offset

