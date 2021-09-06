---
layout: post
title:  Dropbox
categories: Interview
---

# Dropbox
## allocate ID
```py
"""
Dropbox
Id Allocator / Number / Phone Number
Write a class for an id allocator that can allocate and release ids
"""

class Allocator:

    def __init__(self, max_val):
        self.queue = collections.deque()
        self.first_pass_idx = 0 #your interview might not require this optimization
        self.allocated = set()
        self.max_val = max_val
        
    def allocate(self):
        """Returns an unallocated id"""
        result = None
        if self.first_pass_idx <= self.max_val:
            self.first_pass_idx += 1
            result = self.first_pass_idx - 1
        elif len(self.queue) > 0:
            result = self.queue.pop()
        if result is not None:
            self.allocated.add(result)
            return result
        else:
            raise CannotAllocateException("No ids available")

    def release(self, id):
        """Releases the id and allows it to be allocated"""
        if (not 0 <= id < self.max_val) or (id not in self.allocated):
            #You should say that you'd like to throw an exception in case of an error
            raise InvalidIdException(f"The id {id} cannot be released.")
        self.allocated.remove(id)
        self.queue.appendleft(id)

"""
FOLLOW UP:
You might be asked to estimate the amount of memory you need for the above implementation.
You will be asked to make a more space-efficient implementation in which allocate and release might take longer than O(1).
For this, you can use a boolean array (a.k.a. a BitSet in Java, a bit-vector in other languages)
This uses max_id // (8 * 1024 * 1024) MB
"""

class SpaceEfficientAllocator:

    def __init__(self, max_val):
        self.max_val = max_val
        self.bool_array = [False] * max_val

    def allocate(self):
        """Returns an unallocated id"""
        for id, value in enumerate(self.bool_array):
            if value == False: #The id has not been allocated
                self.bool_array[id] = True
                return id
        raise CannotAllocateException("No ids available")

    def release(self, id):
        """Releases the id and allows it to be allocated"""
        if (not 0 <= id < self.max_val) or (self.bool_array[id] == True):
            raise Exception(f"The id {id} cannot be released.")
        self.bool_array[id] = False

# Segment Tree
class Allocator:
    def __init__(self,size):
        self.capacity = size
        self.data = [False] * 2 * size
    
    def allocate(self):
        if self.data[1] == True:
            print("no avaiable slots")
            return
        idx = 1
        while idx < self.capacity:
            left, right = idx*2, idx*2+1
            if self.data[left] and self.data[right]:
                print("no avaiable slots")
                return
            if not self.data[left]:
                idx = left
            else:
                idx = right
        self.data[idx] = True
        print("allocating {}".format(idx-self.capacity))
        # updating segment tree
        while idx > 1:
            p = idx // 2
            left, right = p*2, p*2+1
            if  self.data[left] and self.data[right]:
                self.data[p] = True
                idx = p
            else:
                break


    def release(self, id):
        if id < 0 or id >= self.capacity:
            print("id out of range")
            return
        target = id + self.capacity
        if self.data[target] == False:
            print("{} is not occupied".format(id))
            return
        print("releasing {}".format(id))
        self.data[target] = False
        idx = target
        while idx > 1:
            p = idx // 2
            left, right = p*2, p*2+1
            if not self.data[left] and not self.data[right]:
                self.data[p] = False
                idx = p
            else:
                break
allocator = Allocator(10)
for i in range(11):
    allocator.allocate()
for i in range(11):
    allocator.release(i)
for i in range(3):
    allocator.release(i)
```
## Web Crawler
Design a web crawler, first single-threaded, then multithreaded.


Multithreading Things to consider:
1. which part is most time-consuming. The part that the thread knows the url to be visited and getting back list of URLs. Network latency, webpage content parser and processing.
2. Should visit URL? e.g. define the depth of crawling, type of urls e.g. the one without ending in .pdf, size of the result set etc.
3. Crawler failed, will throws ExecutionException
4. Sleep the master thread a little bit each time after the checking of futures… manager thread will not use all the resources

```py
"""
Given a URL, crawl that webpage for URLs, and then continue crawling until you've visited all URLs
Assume you have an API with two methods:
get_html_content(url) -> returns html of the webpage of url
get_links_on_page(html) -> returns array of the urls in the html
Do this in a breadth-first style manner (it's just easier).
"""



class WebCrawler:
    # get_html_content(url) -> returns html of the webpage of url
    # get_links_on_page(html) -> returns array of the urls in the html
    def __init__(self, url):
        import collections
        self.visited = set()
        self.q = collections.deque()
        self.q.append(url)
        self.visited.add(url)
        
    def process_url(self, url):
        try:
            html = get_html_content(url)
        except ConnectionError:
            return
        links = get_links_on_page(html)
        for link in links:
            if link not in self.visited:
                self.q.append(link)
                self.visited.add(link)
                
    def run(self):
        while self.q:
            cur_url = self.q.popleft()
            self.process_url(cur_url)

# bottleneck: if there are many urls pending, we can probably doing it in parallel


"""
Now you are asked what the bottleneck is. See the above comment.
How do you fix the bottleneck?
Use multithreading!
Some interviewers let you use the ThreadPoolExecutor (lets you queue the work and self-manages the threads)
"""
from concurrent.futures import ThreadPoolExecutor

class MultiThreadedWebcrawler:

    def __init__(self, url):
        self.visited_urls = set()
        self.lock = threading.Lock()
        self.url_queue = collections.deque()
        self.url_queue.appendleft_(url)
        self.active_futures = []
        self.max_active_jobs_in_pool = 50
    
    def process_url(self, url):
        try:
            html = get_html_content(url) #Interviewer asks which line is the bottleneck. It's this one!
        except ConnectionError:
            return #talk about retries, what to do in this case
        links = get_links_on_page(html)
        with self.lock: #this is the same as calling self.lock.acquire()
            for link in links:
                if link not in self.visited_urls:
                    self.visited_urls.add(link)
                    self.url_queue.appendleft(link)
        #and then calling self.lock.release()

    def run(self):
        with pool as ThreadPoolExecutor(max_workers=20):
            while True:
                with lock:
                    num_active_jobs = len(self.active_futures)
                    num_urls_to_crawl = len(self.url_queue)
                    if num_urls_to_crawl == 0 and num_active_jobs == 0:
                        #Termination - you have no urls left to crawl, and all of your 
                        #jobs in the pool are complete.
                        break
                    
                    #If you have too many jobs still running in the pool, then just let them run again
                    #Otherwise, if you have a manageable amount, then you can submit more. 
                    if num_active_jobs <= self.max_active_jobs_in_pool:
                        number_of_jobs_to_submit = min(
                            num_urls_to_crawl, 
                            self.max_active_jobs_in_pool - num_active_jobs
                        )
                        for _ in range(number_of_jobs_to_submit):
                            future = pool.submit(self.process_url, self.url_queue.pop())
                            self.active_futures.append(future)
                #Outside of the lock, you can remove completed futures from the active_futures
                self.active_futures = [future for future in self.active_futures if not future.done()]
                time.sleep(1) #Let someone else take the lock. 
                
        return list(self.visited_urls)
```

## Token bucket
类似Token Bucket, 换成了水龙头往桶里滴水


## Producer and Consumer
python queue implementation:
https://github.com/python/cpython/blob/main/Lib/multiprocessing/queues.py
```py


from collections import deque
from threading import *
import random
class cp_problem:
    def __init__(self,size):
        self.capacity = size
        self.q = deque()
        self.notFull =  Condition()
        self.notEmpty = Condition()
        self.lock = Lock()

    def put(self):
        self.notFull.acquire()
        while len(self.q) == self.capacity:
            print("wait for queue to be not full")
            self.notFull.wait()

        self.lock.acquire()
        item = len(self.q)
        self.q.append(item)
        print("put {}".format(item))

        self.lock.release()
        # must release after processing, otherwise it might cause other producer
        # to put things into it
        self.notFull.release() 

        self.notEmpty.acquire()
        self.notEmpty.notifyAll()
        self.notEmpty.release()

    def release(self):
        self.notEmpty.acquire()
        while len(self.q) == 0:
            print("wait for queue to be not empty")
            self.notEmpty.wait()


        self.lock.acquire()
        item = self.q.pop()
        print("release {}".format(item))
        self.lock.release()

        self.notEmpty.release()
        self.notFull.acquire()
        self.notFull.notifyAll()
        self.notFull.release()

    def run(self):
        from threading import Thread
        for i in range(10):
            t = Thread(target=self.put)
            t.start()
        for i in range(10):
            t = Thread(target=self.release)
            t.start()
cp = cp_problem(3)
cp.run()
```