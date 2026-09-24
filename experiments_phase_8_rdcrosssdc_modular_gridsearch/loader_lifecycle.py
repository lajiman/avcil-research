"""Explicit lifecycle helpers for persistent PyTorch DataLoaders."""


def _close_stopped_worker_handles(workers):
    """Release process sentinels retained by PyTorch's atexit callbacks."""
    for worker in workers:
        try:
            if worker.is_alive():
                continue
        except (AssertionError, ValueError):
            continue

        popen = getattr(worker, "_popen", None)
        close = getattr(popen, "close", None)
        if close is not None:
            close()


def close_data_loader(loader):
    """Stop a persistent loader and release its worker-side OS resources.

    PyTorch has no public DataLoader.close() API.  Calling the iterator's
    shutdown routine closes the pin-memory thread and multiprocessing queues;
    closing each stopped worker's Popen object also releases process sentinel
    descriptors that may remain referenced by PyTorch's atexit callbacks.
    """
    if loader is None:
        return

    iterator = getattr(loader, "_iterator", None)
    if iterator is None:
        return

    workers = tuple(getattr(iterator, "_workers", ()))
    try:
        iterator._shutdown_workers()
    except BaseException:
        _close_stopped_worker_handles(workers)
        raise
    else:
        loader._iterator = None
        _close_stopped_worker_handles(workers)
