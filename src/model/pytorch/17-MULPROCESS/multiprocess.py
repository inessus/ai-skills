from datetime import datetime
sites = [
    'http://www.python.org/',
    'http://pypi.org/',
    'http://conda.io/',
    'http://www.jython.org/',
    'http://ironpython.net',
    'http://pypy.org/',
    'http://github.com/',
    'http://stackoverflow.com'
]
async def fetch_size(url: str):
    """Return the size of a website and the time to load it"""
    import requests
    from time import time
    start_time: float = time()
    page = requests.get(url).content
    duration: float = time() - start_time
    return url, len(page), duration


def sequential():
    result = {}
    for url in sites:
        _, size, duration = fetch_size(url)
        result[url] = size, duration
    return result

def multiprocess():
    from multiprocessing import Process, SimpleQueue
    queue = SimpleQueue()
    process = []
    for url in sites:
        p = Process(target = lambda url, q: q.put(fetch_size(url)),
            args=(url, queue))
        p.start()
        process.append(p)
    for p in process:
        p.join()
    result = {}
    while not queue.empty():
        url, size, duration = queue.get()
        result[url] = size, duration
    return result

def multiprocess2():
    from multiprocessing import Pool
    result = {}
    pool = Pool()
    for url, size, duration in pool.imap_unordered(fetch_size, sites):
        result[url] = size, duration
    pool.close()
    return result

def multiprocess3():
    from concurrent.futures import ProcessPoolExecutor
    result = {}
    with ProcessPoolExecutor() as pool:
        for url, size, duration in pool.map(fetch_size, sites):
            result[url] = size, duration
    return result

def multithread():
    from concurrent.futures import ThreadPoolExecutor
    result = {}
    with ThreadPoolExecutor() as pool:
        for url, size, duration in pool.map(fetch_size, sites):
            result[url] = size, duration
    return result


if __name__ == "__main__":
    
    start = datetime.now()
    # res = sequential()
    res = multithread()
    for k, v in res.items():
        print("{0:40} {1:7} {2:.5}".format(k, v[0], v[1]))
    print("sequential {}".format(datetime.now()-start))