import numpy as np
from libc.stdlib cimport qsort

DTYPE = np.float64

cdef int comparator(const void* a, const void* b) nogil:
    if a < b:
        return -1
    elif a == b:
        return 0
    else:
        return 1

# Computes the gini score for a split
# 0 < t < len(y)
cpdef double score(double[:] y, int t, double n_samples):

    cdef size_t length = y.shape[0]
    cdef double left_gini = 1.0
    cdef double right_gini = 1.0
    cdef double gini = 0
    cdef int i = 0
    cdef double temp = 0.0
    cdef double count = 0.0
    
    left = y[:t]
    right = y[t:]

    cdef double[:] left_view = left
    cdef double[:] right_view = right
    cdef int lv_length = left_view.shape[0]
    cdef int rv_length = right_view.shape[0]
    cdef int lv_strides = left_view.strides[0]
    cdef int rv_strides = right_view.strides[0]

    # Sort the arrays
    qsort(&left_view[0], lv_length, lv_strides, &comparator)
    qsort(&right_view[0], rv_length, rv_strides, &comparator)

    # Count unique elements
    temp = left_view[0]
    count = 1.0
    for i in range(1, lv_length):
        if left_view[i] > temp:
            count = count / lv_length
            count = count * count
            left_gini = left_gini - count

            count = 1.0
            temp = left_view[i]

        else:
            count = count + 1

    count = count / lv_length
    count = count * count
    left_gini = left_gini - count

    temp = right_view[0]
    count = 1.0
    for i in range(1, rv_length):
        if right_view[i] > temp:
            count = count / rv_length
            count = count * count
            right_gini = right_gini - count

            count = 1.0
            temp = right_view[i]

        else:
            count = count + 1

    count = count / rv_length
    count = count * count
    right_gini = right_gini - count

    gini = (lv_length / n_samples) * left_gini + (rv_length / n_samples) * right_gini
    return gini


