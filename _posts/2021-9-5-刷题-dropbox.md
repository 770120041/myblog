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

这道题给了一个get_links(String initialUrl) 要你写个方法来搜集能从这个initialUrl能够搜到的所有url。最先的方法是BFS， 用一个Set记录已经crawl过的url， 用一个queue来记录还没crawl过的。 这里楼主使用的是BFS， 因为in practice大部分搜索引擎都是用BFS。 用DFS会有若干劣势， 比如1. Takes long time to find high quality pages. 2. make the crawler focus on a few sites

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
                with self.lock:
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

**注意： acquire lock only when needed**
using condition variable to handle wait first, this is important

after processing with the bucket, notify all


```py
from collections import deque
from threading import *
import random
import datetime
import threading
import time

class TokenBucket:
    def __init__(self, capacity, fillRate):
        self.capacity = capacity
        self.fillRate = fillRate
        self.lastFillts = datetime.datetime.now()
        self.bucket = []
        self.lock = Lock()
        self.notEmpty = Condition()
        self.notFull = Condition()

    def fill(self, id):
        print("fill#{}:: start trying to fill".format(id))

        self.notFull.acquire()
        while len(self.bucket) == self.capacity:
            print("fill#{}:: waiting notFULL".format(id))
            self.notFull.wait()

        self.lock.acquire()
        now = datetime.datetime.now()
        token_to_fill = min(int(self.convert_to_int(now) - self.convert_to_int(self.lastFillts))*self.fillRate,
                                self.capacity - len(self.bucket))
        self.lastFillts = now
        print("fill#{}:: filling {} item".format(id, token_to_fill))
        for _ in range(token_to_fill):
            item = len(self.bucket)
            self.bucket.append(item)
        self.lock.release()

        """ 
        same as 
            with self.notEmpty:
                self.notEmpty.notifyAll()     
        """
        self.notEmpty.acquire()
        self.notEmpty.notifyAll()
        self.notEmpty.release()
        
        self.notFull.release()

    def convert_to_int(self,t):
        ret = int(t.strftime("%Y%m%d%H%M%S"))
        return ret

    def get(self, id, n):
        if n < 0 or n > self.capacity:
            print("get#{}::invalid input {}".format(id,n))
        print("get#{}:: trying to get {} items".format(id,n))
        res = []
        num_acquired = 0
        # fair competition, each time we release one token
        while num_acquired < n:
            self.notEmpty.acquire()
            while len(self.bucket) == 0:
                print("get#{}:: wait notEmpty".format(id))
                self.notEmpty.wait()
            
            self.lock.acquire()
            item = self.bucket.pop()
            res.append(item)
            print("get#{}:: acquired item #{}, {} remaining".format(id, item, n-num_acquired))
            num_acquired += 1
            self.lock.release()

            self.notEmpty.release()
            self.notFull.acquire()
            self.notFull.notifyAll()
            self.notFull.release()
        return res
        



def fill_f(tb, id):
    while True:
        tb.fill(id)
        time.sleep(1)

def get_f(tb, id):
    while True:
        num_to_get = random.randint(1,10)
        tb.get(id, num_to_get)
        time.sleep(1)
tb = TokenBucket(10,20)
time.sleep(1)
for i in range(5):
    fill_t = threading.Thread(target=fill_f,args=(tb,i,))
    fill_t.start()
for i in range(5):
    get_t = threading.Thread(target = get_f, args=(tb,i,))
    get_t.start()
```

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

## Hit Counter
```py
# Hit(): add hits to queue
# getHit(): count number of hits
# draw back: memory inefficient if there are many hits
# Hit() is fast but getHit() is slow
import time
from math import floor
from collections import deque
class HitCounter:
    def __init__(self, span = 3):
        self.span = span
        self.q = deque()

    def current_milli_time(self):
        return round(time.time() * 1000)

    def hit(self):
        current_ts = floor(time.time()) 
        self.q.append((current_ts))

    def getHit(self):
        current_ts = floor(time.time()) 
        while self.q and current_ts - self.q[0][0] > 3:
            self.q.popleft()

# Optimize: Circular array
# using an array of size equal to given seconds
# store the timestamp and counts for that second


class HitCounter:
    def __init__(self, span=3):
        self.span = span
        self.counter = [None] * span

    def hit(self):
        current_ts = floor(time.time()) 
        slots = current_ts % self.span
        if not self.counter[slots] or self.counter[slots][0] != current_ts:
            self.counter[slots] = [current_ts,1]
        else:
            self.counter[slots][1] += 1
    
    def getHit(self):
        current_ts = floor(time.time()) 
        res = 0
        for i in range(len(self.counter)):
            ts, cnt = self.counter[i]
            if current_ts - ts <= self.span:
                res += cnt
            else:
                self.counter[i] = None
        return res

```


## Count and say
```py
class Solution(object):
    def countAndSay(self, n):
        """
        :type n: int
        :rtype: str
        """
        cur_str = "1"
        # 2 to n. n-1 times
        for i in range(2,n+1):
            count = -1
            cur_c = None
            next_str = []
            for j in range(len(cur_str)):
                if count < 0 or cur_c == cur[j]:
                    if count < 0:
                        cur_c = cur[j]
                        count = 1
                    else:
                        count += 1
                else:
                    next_str.append((count,cur_c))
                    count = 1
                    cur_c = cur[j]
                if j == len(cur)-1 and count > 0:
                    next_str.append((count,cur_c))            
            cur_str = ""
            for x in next_str:
                cur_str += str(x[0])+str(x[1])
            
        return cur_str
```

## Read Write lock
#### Simple
simple form, prioritize write request, if there are writers or write request, read lock will wait.

http://tutorials.jenkov.com/java-concurrency/read-write-locks.html
```py
from threading import *

class RWLock:
    def __init__(self):
        self.readers = 0
        self.writers = 0
        self.write_requests = 0
        self.has_writer = False
        self.Ready = Condition()

    def lockRead(self):
        print("trying to acquire read lock")
        self.Ready.acquire()
        while self.writers > 0 or self.write_requests > 0:
            print("read lock wait ")
            self.Ready.wait()
        self.readers += 1
        print("read lock acquired")
        self.Ready.release()

    def unlockRead(self):
        print("trying to unlock Read lock")
        with self.Ready:
            print("read lock released")
            self.readers -= 1
            self.Ready.notifyAll()

    def lockWrite(self):
        print("trying to acquire write lock")
        self.Ready.acquire()
        self.write_requests += 1
        while self.readers > 0 or self.writers > 0:
            print("write lock wait")
            self.Ready.wait()
        self.write_requests -= 1
        self.writers += 1
        print("write lock acquired")
        self.Ready.release()

    def unlockWrite(self):
        print("trying to unlock Write lock")
        with self.Ready:
            print("write lock released")
            self.writers -= 1
            self.Ready.notifyAll()

def writeLockTest(rwl):
    rwl.lockWrite()
    rwl.unlockWrite()
rwl = RWLock()
rwl.lockRead()
rwl.lockRead()
write_lock = Thread(target = writeLockTest, args=(rwl, ))
write_lock.start()
rwl.unlockRead()
rwl.unlockRead()
```


#### reentrance
Read / Write Lock Reentrance
The ReadWriteLock class shown earlier is not reentrant. If a thread that has write access requests it again, it will block because there is already one writer - itself. Furthermore, consider this case:

Thread 1 gets read access.

Thread 2 requests write access but is blocked because there is one reader.

Thread 1 re-requests read access (re-enters the lock), but is blocked because there is a write request
In this situation the previous ReadWriteLock would lock up - a situation similar to deadlock. No threads requesting neither read nor write access would be granted so.

To make the ReadWriteLock reentrant it is necessary to make a few changes. Reentrance for readers and writers will be dealt with separately.

Read Reentrance
To make the ReadWriteLock reentrant for readers we will first establish the rules for read reentrance:

A thread is granted read reentrance if it can get read access (no writers or write requests), or if it already has read access (regardless of write requests).


#### Read Reentrance
To make the ReadWriteLock reentrant for readers we will first establish the rules for read reentrance:

A thread is granted read reentrance if it can get read access (no writers or write requests), or if it already has read access (regardless of write requests).

Since we only ask the thread to wait when there are writers or write_requests, we don't have to modify the code

#### Write Reentrance
Write reentrance is granted only if the thread has already write access. Here is how the lockWrite() and unlockWrite() methods look after that change:
```py
from threading import *
import threading

class RWLock:
    def __init__(self):
        self.readers = 0
        self.writers = 0
        self.write_requests = 0
        self.has_writer = False
        self.Ready = Condition()
        self.writer_thread = None

    def __canGrantReadLock(self, thread_ident):
        if self.writer_thread == thread_ident:
            return True
        if self.writers > 0 or self.write_requests > 0:
            return False
        return False
    def lockRead(self):
        thread_id = threading.get_ident()
        print("#{} trying to acquire read lock".format(thread_id))
        self.Ready.acquire()
        while not self.__canGrantReadLock(thread_id):
            print("read lock wait ")
            self.Ready.wait()
        self.readers += 1
        print("read lock acquired")
        self.Ready.release()

    def unlockRead(self):
        print("trying to unlock Read lock")
        with self.Ready:
            print("read lock released")
            self.readers -= 1
            self.Ready.notifyAll()

    def __canGrantWrite(self, thread_ident):
        if thread_ident == self.writer_thread:
            return True
        if self.readers > 0 or self.writers > 0:
            return True
        return False

    def lockWrite(self):
        thread_id = threading.get_ident()
        print("#{} trying to acquire write lock".format(thread_id))
        self.Ready.acquire()
        self.write_requests += 1
       
        while not self.__canGrantWrite(thread_id):
            print("write lock wait")
            self.Ready.wait()
        self.write_requests -= 1
        self.writers += 1
        self.writer_thread = thread_id
        print("write lock acquired")
        self.Ready.release()

    def unlockWrite(self):
        print("trying to unlock Write lock")
        with self.Ready:
            print("write lock released")
            self.writers -= 1
            if self.writers == 0:
                self.writer_thread = None
            self.Ready.notifyAll()

def writeLockTest(rwl):
    rwl.lockWrite()
    rwl.lockWrite()
    rwl.lockRead()
    rwl.unlockWrite()
    rwl.unlockRead()
    rwl.unlockWrite()
rwl = RWLock()
rwl.lockRead()
rwl.lockRead()
write_lock = Thread(target = writeLockTest, args=(rwl, ))
write_lock.start()
rwl.unlockRead()
rwl.unlockRead()
rwl.lockRead()
```


#### reader acuquire writelock
need to check if the writer is the only reader
```py
    def __canGrantWrite(self, thread_ident):
        if self.isOnlyReader(thread_ident):
            return True
        if thread_ident == self.writer_thread:
            return True
        if self.readers > 0 or self.writers > 0:
            return True
        return False

    def isOnlyReader(self, thread_ident):
        if len(self.reading_thread) == 1 and self.reading_thread[0] == thread_ident:
            return True
        return False
```
#### Write to Read Reentrance
check if the thread is a writer
```

    def __canGrantReadLock(self, thread_ident):
        if self.writer_thread == thread_ident:
            return True
        if self.writers > 0 or self.write_requests > 0:
            return False
        return False
```

## Space Panorama
Create an API to read and write files and maintain access to the least-recently written file. Then scale it up to a pool of servers.

Solution

## Phone Number / Dictionary
- Given a phone number, consider all the words that could be made on a T9 keypad. Return all of those words that can be found in a dictionary of specific words.
This question is sometimes asked to college students and sometimes asked in phone screens. It isn't asked a lot in onsites.

Solution

## Sharpness Value
This question is usually phrased like "find the minimum value along all maximal paths". It's a dynamic programming question.
Occasionally asked in phone screens. Might be asked in onsites for new hires.

Solution

## Find Byte Pattern in a File
Determine whether a pattern of bytes occurs in a file. You need to understand the Rabin-Karp style rolling hash to do well.
Somewhat frequently asked in onsite interviews. Might be asked in phone screens.


## Download File / BitTorrent
Create a class that will receive pieces of a file and tell whether the file can be assembled from the pieces.
This question is mostly for new graduates/phone screens.

## Game of Life
Conway's Game of Life - Problem on LeetCode
This question is EXTREMELY popular for phone screens.

Solution


## KV store

KV Store这道题的问题是设计一个transaction， 有一个start() 方法， 返回一个transaction id. 有一个put(transactionId, String key, int value), 有一个get(transactionId, String key), 和一个commit(transactionID)。 要理解这道题到底什么意思， 首先得先翻翻Database的书， 看下transaction那一章， transaction的四个属性ACID。 主要是Isolation level。 transaction有四个level， Read Uncommitted, Read Committed, Repeatable Read和Serializable。 根据面试官的要求， 你要实现其中的一个level， 比如面试官如果说这个transaction会用在bank system里面， 那么最好就是实现Repeatable Read那个level。 因为这个level可以避免dirty reads, non-repeatable reads, and lost updates。想象下如果有两个transaction以下面这个顺序进行读写(假设a之前的值为1)
start()  // start transaction 1
start() // start transaction 2
int val1 = get(1, "a");
int val2 = get(2, "a");
put(1, "a", val1+1);
put(2, "a", val2+1);
commit(1);
commit(2);
那么transaction 1的操作就会overwritten， 这个就是update lost。
解决update lost就需要实现repeatable read， 意思就是当一个transaction得到一个key的读的lock时， 要一直hold这个lock到这个transaction结束为止。 这样一来当例子中的第二个transaction要读a的时候就会被拒绝。 根据面试官的要求， 一般会直接throw一个error然后把第二个transaction取消掉（rollback之前所有已做过的操作）。 还有一点要注意的是， 这其实是一个单线程题， 所以不需要考虑多线程， 比如上一个例子中所有code都是有时间先后顺序的， 因为他们都发生在同一个thread里。 至于在实现锁的这块， 可以使用Map来记录当前哪些key已经有读的锁， 哪些有写的锁。 如果一个key已经有读的锁， 那么其他transaction只能获得读的锁， 如果一key已经有写的锁， 那么其他transaction不能再获得读或写的锁。 还有一点要注意的是， 一个key上的锁如果只有一个transaction并且此时要求锁的那个transaction就是holding锁的那个transaction， 那么这个transaction的读或写应该被允许。 如果一个transaction之前改变了一个值， 在后来的操作发现有conflict， 那么要把这个transaction之前修改过的值都改回原先值。
比如说有这些操作，
start()  // start transaction 1
start() // start transaction 2
int val1 = get(1, "a");
put(2, "b", 2);
put(2, "a", 2);
commit(1);
commit(2);

当transaction 2 改b的值时， b的值可以被成功修改， 但是当transaction 2 修改a的值时，因为此时a的lock已经被transaction 1 hold， 所以这里有一个conflict， 所以transaction 2要被cancel掉。这里可以用一个Map来记录一个key和它原先的值， 每一个transaction都有一个这样的map。 当发现一个transaction需要被cancel时， 把这个transaction之前改过的所有key都要恢复原来的值。这道题的基本就是这个意思了。面试的时候要跟面试官讨论到底要实现那种isolation level。 isolation level越高越复杂。