import numpy


# TODO 内聚更多工具方法

def argsort(x, topn=None, reverse=False):
    """Efficiently calculate indices of the `topn` smallest elements in array `x`.

    Parameters
    ----------
    x : array_like
        Array to get the smallest element indices from.
    topn : int, optional
        Number of indices of the smallest (greatest) elements to be returned.
        If not given, indices of all elements will be returned in ascending (descending) order.
    reverse : bool, optional
        Return the `topn` greatest elements in descending order,
        instead of smallest elements in ascending order?

    Returns
    -------
    numpy.ndarray
        Array of `topn` indices that sort the array in the requested order.

    """
    x = numpy.asarray(x)  # unify code path for when `x` is not a np array (list, tuple...)
    if topn is None:
        topn = x.size
    if topn <= 0:
        return []
    if reverse:
        x = -x
    if topn >= x.size or not hasattr(numpy, 'argpartition'):
        return numpy.argsort(x)[:topn]
    # np >= 1.8 has a fast partial argsort, use that!
    most_extreme = numpy.argpartition(x, topn)[:topn]
    return most_extreme.take(numpy.argsort(x.take(most_extreme)))
