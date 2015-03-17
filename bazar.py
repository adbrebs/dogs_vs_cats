def threaded_generator(generator, num_cached=5):
    """
    Simple queue
    """
    import Queue
    queue = Queue.Queue(maxsize=num_cached)
    sentinel = object()  # guaranteed unique reference

    # define producer (putting items into queue)
    def producer():
        for item in generator:
            queue.put(item)
        queue.put(sentinel)

    # start producer (in a background thread)
    import threading
    thread = threading.Thread(target=producer)
    thread.daemon = True
    thread.start()

    # run as consumer (read items from queue, in current thread)
    item = queue.get()
    while item is not sentinel:
        yield item
        queue.task_done()
        item = queue.get()


def threaded_generator2(generator, num_cached=5):
    """
    Completely random batches
    """
    import multiprocessing
    queue = multiprocessing.Queue(maxsize=num_cached)
    sentinel = object()  # guaranteed unique reference

    # define producer (putting items into queue)
    def producer():
        while True:
            queue.put(generator.generate_random())

    threads=[multiprocessing.Process(
        target=producer)
             for i in xrange(multiprocessing.cpu_count()-1)]
    for thread in threads:
        thread.start()  ## Create batches in parallel

    # run as consumer (read items from queue, in current thread)
    item = queue.get()
    for i in range(generator.n_batches):
        yield item
        item = queue.get()


def threaded_generator3(generator, num_cached=10):
    """
    Sequence
    """
    queue = multiprocessing.Queue(maxsize=num_cached)

    n_procs = multiprocessing.cpu_count()-1
    n_batches = generator.dataset.n_data / generator.batch_size
    batches = range(n_batches)
    n_batches_per_proc = n_batches/n_procs
    proc_idx_batches = [batches[i:i + n_batches_per_proc] for i in range(0, (n_procs-1)*int(n_batches_per_proc), int(n_batches_per_proc))]
    proc_idx_batches.append(batches[(n_procs-1)*n_batches_per_proc:])

    # define producer (putting items into queue)
    def producer(proc_idx_batch):
        for i in proc_idx_batch:
            queue.put(generator.generate_sequence(i))

    threads=[multiprocessing.Process(
        target=producer,
        args=(proc_idx_batches[i],))
             for i in xrange(n_procs)]
    for thread in threads:
        thread.start()  ## Create batches in parallel

    # run as consumer (read items from queue, in current thread)
    for i in range(n_batches):
        yield queue.get()


    # def generate_random(self):
    #     """
    #     Generate completely random batch
    #     """
    #     idx = numpy.random.randint(0, self.dataset.n_data, self.batch_size)
    #
    #     x = []
    #     for shape in self.shape_batch:
    #         x.append(numpy.zeros(shape, dtype=theano.config.floatX))
    #
    #     y = numpy.zeros((self.batch_size,), dtype='int32')
    #
    #     for i in xrange(self.batch_size):
    #         real_idx = idx[i]
    #         img = numpy.reshape(self.dataset.x[real_idx], self.dataset.s[real_idx])
    #         new_imgs = self.data_augmentator.process_image([img])
    #
    #         y[i] = self.dataset.y[real_idx]
    #         for a, img in enumerate(new_imgs):
    #             x[a][i] = img
    #
    #     for a in range(len(x)):
    #         x[a] = self.swap_axis_and_norm(x[a])
    #     return (x + [y], idx+self.dataset.start)