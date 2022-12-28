import copy
from typing import Optional

from algoHelper import *
import numpy as np

def lc_0001():
    # 2和
    def _2Sum(arr, t):
        dic = {}
        for i, ele in enumerate(arr):
            if t - ele in dic:
                return [ele, t-ele]
            dic[ele] = True
        return


def lc_0704():
    #二分搜索，包含返回floor value的变化
    def search(nums: List[int], target: int) -> int:
        l, r = 0, len(nums) - 1
        while l <= r:
            m = l + ((r-l) >> 1)
            if nums[m] == target:
                return m
            elif nums[m] > target:
                r = m - 1
            elif nums[m] < target:
                l = m + 1
        return -1
        # return r if r >= 0 else r + 1 # 返回小于target的最大值
    arr = [1,2,3,4]
    target = 2.0000000000001
    print(search(arr, target))


def lc_0026():
    # 数组中去掉重复数
    def compare(nums):
        return len(sorted(list(np.unique(nums))))


    def removeDuplicates(nums: List[int]):
        # 双指针占0，1位，快指针和当前慢指针位置比较，如果不等，慢指针+1并交换快慢值;快指针每次都向右挪一个，直到快指针走完
        # o(N), O(1)
        if len(nums) < 2:
            return 1
        i, j = 0,1
        while j < len(nums):
            if nums[i] != nums[j]:
                i += 1
                nums[i], nums[j] = nums[j], nums[i]
            j += 1
        return i + 1


    def removeDuplicates2(nums):
        # 填数法
        k = 0
        for i in range(len(nums)):
            if nums[k] != nums[i]:
                k += 1
                nums[k] = nums[i]
            i += 1
        return k+1
    for i in range(50000):
        n = random.randint(1, 1000)
        nums = sorted(list(np.random.randint(-10000, 10000, n)))
        target = random.randint(-10000, 10000)
        A = removeDuplicates2(copy.copy(nums))
        B = compare(copy.copy(nums))
        if A != B:
            print((nums, A, B))
            break


def lc_0027():
    def compare(nums, val):
        return len([i for i in nums if i != val])

    def removeElement(nums: List[int], val: int) -> int:
        # 快指针只9要不是target value就填到慢指针上，然后两个指针都往右走
        k = 0
        for i in range(len(nums)):
            if nums[i] != val:
                nums[k] = nums[i]
                k += 1
            i += 1
        return k

    for i in range(100000):
        n = random.randint(1, 10)
        nums = sorted(list(np.random.randint(-10000, 10000, n)))
        target = random.randint(-10000, 10000)

        # nums = [-7801, -5450, -3529, 933, 2341, 7326, 7326]
        # target = 7326
        A = removeElement(copy.copy(nums), target)
        B = compare(copy.copy(nums), target)
        if A != B:
            print((nums, A, B))
            break


def lc_0283():
    def compare(nums):
        left = [i for i in nums if i != 0]
        return left + [0] * (len(nums)-len(left))

    def moveZeroes(nums: List[int]) -> None:
        i, j = 0, 0
        for i in range(len(nums)):
            if nums[i] != 0:
                nums[i], nums[j] = nums[j], nums[i]
                j += 1
            i += 1
        return

    for i in range(50000):
        n = random.randint(1, 2000)
        nums = sorted(list(np.random.randint(-1000, 1000, n)))
        A = copy.copy(nums)
        B = copy.copy(nums)
        moveZeroes(A)
        B = compare(B)

        if not all([x==y for x, y in zip(A, B)]):
            print((nums, A, B))
            break


def lc_0034():
    def compare(nums, target):
        lb, rb = -1, -1
        found = False
        for i, ele in enumerate(nums):
            if ele == target and not found:
                lb = i
                rb = i
            elif ele == target:
                rb = i
        return [lb, rb]
    def searchRange(nums, target):
        # 查找target在有序数组sums上的范围，返回范围的左右边界，没有的话返回[-1,-1]
        def lb(nums, target):
            l, r = 0, len(nums)-1
            while l <= r:
                m = l+((r-l)>>1)
                if nums[m] == target:
                    r = m - 1
                elif nums[m] > target:
                    r = m - 1
                elif nums[m] < target:
                    l = m + 1
            if l == len(nums):
                return -1
            return l if nums[l] == target else -1

        def rb(nums, target):
            l, r = 0, len(nums)-1
            while l <= r:
                m = l+((r-l)>>1)
                if nums[m] == target:
                    l = m+1
                elif nums[m] > target:
                    r = m-1
                elif nums[m] < target:
                    l = m+1
            if r < 0:
                return -1
            return r if nums[r] == target else -1


        return [lb(nums, target), rb(nums, target)]

    for i in range(50000):
        n = random.randint(1, 1000)
        nums = sorted(list(set(list(np.random.randint(-10000, 10000, n)))))
        target = random.randint(-10000, 10000)
        A = searchRange(nums, target)
        B = compare(nums, target)
        if not all([a==b for a,b in zip(A, B)]):
            print((nums, target))
            break


def lc_0035():
    def compare(arr, target):
        if target > arr[-1]:
            return len(arr)
        if target < arr[0]:
            return 0
        for i, ele in enumerate(arr):
            if ele >= target:
                return i
    def searchInsert(nums: List[int], target: int) -> int:
        # 找在有序数组中target的位置或者插入target的位置
        # 使用对数器检查出这个二分搜索不适合有重复值的arr
        # 二分查找[l,r]与[l,r)的写法不同，思考这个边界以及target大于mid的含义
        l, r = 0, len(nums) - 1
        while l <= r:
            m = l + ((r-l) >> 1)
            if nums[m] == target:
                return m
            elif nums[m] < target:
                l = m+1
            else:
                r = m-1
        return r+1 # 这里刚开始是标准的二分查找-1，之后可以通过找规律发现返回r + 1,第二次刷的时候发现返回0 if l < 0 else l也可以
        # 说明二分查找找不到元素的时候，最终l会跑到r的下一个位置

    for i in range(50000):
        n = random.randint(1, 1000)
        nums = sorted(list(set(list(np.random.randint(-10000, 10000, n)))))
        target = random.randint(-10000, 10000)

        if searchInsert(nums, target) != compare(nums, target):
            print((nums, target))
            break


def lc_0844():
    def backspaceCompare1(s: str, t: str) -> bool:
        def moveValues(sList):
            i, j = 0, len(sList) -1
            while j < len(sList):
                if sList[j] == "#" and i > 0:
                    i -= 1
                elif sList[j] == "#":
                    sList[i], sList[j] = sList[j], sList[i]
                j += 1
            return i
        sL, tL = list(s), list(t)
        ss, tt = moveValues(sL), moveValues(tL)
        if ss != tt:
            return False
        for i in range(ss):
            if sL[i] != tL[i]:
                return False
        return True

    def backspaceCompare(s: str, t: str) -> bool:
        def modStr(string):
            res = []
            for i in range(len(string)):
                if string[i] != "#":
                    res.append(string[i])
                elif len(res) > 0:
                    res.pop()
            return "".join(res)
        return modStr(s) == modStr(t)

    for i in range(50000):
        n = random.randint(1, 1000)
        s = sorted(list(set(list(np.random.randint(-10000, 10000, n)))))
        t = []
        target = random.randint(-10000, 10000)
        A = backspaceCompare(s, t)
        B = backspaceCompare1(s, t)
        if A != B:
            print((s, t))
            break


def lc_0709():
    def compare(nums, target):
        for i, ele in enumerate(nums):
            if ele == target:
                return i
        return -1

    def search(nums: List[int], target: int) -> int:
        # 二分搜索，请习惯使用闭区间，比较好解释
        l, r = 0, len(nums)-1
        while l <= r:
            m = l + ((r-l) >> 1)
            if nums[m] == target:
                return m
            elif nums[m] > target:
                r = m - 1
            else:
                l = m + 1
        return -1

    for i in range(50000):
        n = random.randint(1, 1000)
        nums = sorted(list(set(list(np.random.randint(-10000, 10000, n)))))
        target = random.randint(-10000, 10000)

        if search(nums, target) != compare(nums, target):
            print((nums, target))
            break


def lc_0069():
    def compare(x):
        return int(np.sqrt(x))

    def mySqrt1(x: int) -> int:
        # 二分搜索方法
        l,r = 0, x
        while l <= r:
            m = l + ((r-l) >> 1)
            if m * m <= x:
                l = m+1
            else:
                r = m-1
        return r
    def mySqrt(x: int) -> int:
        # 牛顿法万岁，这玩意比二分查找快多了
        x0 = x
        c = 100
        while x0 * x0 > x and c > 0:
            x0 = (x0 + x/x0) / 2
            c -= 1

        return int(x0)

    for i in range(2**20):
        if compare(i) != mySqrt(i):
            print(i)
            break


def lc_0367():
    def compare(x):
        return abs(int(np.sqrt(x)) - np.sqrt(x)) < 1e-9

    def isPerfectSquare(num: int) -> bool:
        def newton(num):
            x0 = num
            c = 100
            while x0 * x0 > num and c > 0:
                x0 = (x0 + num/x0)/2
                c -= 1
            return x0
        sqrtt = newton(num)
        return abs(int(sqrtt) - sqrtt) < 1e-9

    for i in range(2 ** 20):
        if isPerfectSquare(i) != compare(i):
            print(i)
            break


def lc_0066():
    def compare(digits):
        num = int("".join([str(i) for i in digits])) + 1
        return list([int(i) for i in num])

    def plusOne(digits: List[int]) -> List[int]:
        n = len(digits)
        nextdig = 0
        for i in range(n-1, -1, -1):
            if digits[i] == 9:
                digits[i] = 0
                nextdig = 1
            else:
                digits[i] += nextdig
                # 接下来不会有进位了直接跳出来就行了
                break
        if digits[0] == 0:
            digits.insert(0, 1)
        return digits

    def plusOneOptimized(digits):
        n = len(digits)
        for i in range(n-1, -1, -1):
            if digits[i] == 9:
                digits[i] = 0
            else:
                digits[i] += 1
                return digits
        digits.append(0)
        digits[0] = 1
        return digits


    digits = [8,9,9,9,9,9]
    print(plusOneOptimized(digits))


def lc_0088():
    def compare(nums1, nums2, m, n):
        nums1[m:m+n] = nums2
        return sorted(nums1)

    def merge(nums1, m, nums2, n):
        # 给两个有序数组inplace merge成一个有序数组，其中nums1数组末尾处刚开始时有占位元素
        """
        :type nums1: List[int]
        :type m: int
        :type nums2: List[int]
        :type n: int
        :rtype: None Do not return anything, modify nums1 in-place instead.
        """
        k = len(nums1)-1
        p1, p2 = m-1, n-1
        while k >= 0:
            while p1 >= 0 and p2 >= 0:
                if nums1[p1] > nums2[p2]:
                    nums1[k] = nums1[p1]
                    p1 -= 1
                else:
                    nums1[k] = nums2[p2]
                    p2 -= 1
                k -= 1
            while p1 >= 0:
                nums1[k] = nums1[p1]
                p1 -= 1
                k -= 1
            while p2 >= 0:
                nums1[k] = nums2[p2]
                p2 -= 1
                k -= 1
        return

    m = random.randint(1,200)
    n = random.randint(1,200)
    nums1 = sorted(list(np.random.randint(-1e9, 1e9, m))) + [0 for i in range(n)]
    nums2 = sorted(list(np.random.randint(-1e9, 1e9, n)))
    for i in range(50000):
        A = compare(nums1, nums2, m, n)
        merge(nums1, m, nums2, n)
        B = nums1
        if not all([a == b for a,b in zip(A, B)]):
            print(nums1)
            print(nums2)
            print(m)
            print(n)
            break


def lc_0209():
    def minSubArrayLen(target: int, nums: List[int]) -> int:
        if sum(nums) < target:
            return 0
        if len(nums) == 1:
            return 1 if nums[0]==target else 0
        i, j = 0, 0
        le = 1000000000
        sm = 0
        n = len(nums)
        while i <= n - 1:
            if j <= n - 1 and sm < target:
                sm += nums[j]
                j += 1
            elif j <= n - 1 and sm >= target:
                le = min(le, j - i)
                sm -= nums[i]
                i += 1
            elif j > n - 1 and sm < target:
                i += 1
            elif j > n - 1 and sm >= target:
                le = min(le, j - i)
                sm -= nums[i]
                i += 1
        return le

    def minSubArrayLenClean(target: int, nums: List[int]) -> int:
        # start:end: range, curr: subarray sum, l: 长度
        start, end, curr, l = 0, 0, 0, float("inf")
        while end < len(nums):
            if curr < target:
                curr += nums[end]
                end += 1
            while curr >= target:
                l = min(l, end - start) # end 加完range值之后多向后走了一步
                curr -= nums[start]
                start += 1
        # 循环走完，curr==0的情况只有一种，就是没有碰到第二个while loop，也就是说所有值加起来都不够target
        return 0 if curr == float("inf") else l





    nums = [1,2,3,4,5]
    target=15
    print(minSubArrayLen(target, nums))


def lc_0904():
    # 水果篮问题
    def totalFruit2(fruits: List[int]) -> int:

        # 更快一些的解法，记录某种水果最后出现的位置，左指针可以直接跳到合理的位置
        start = 0
        end = 0
        maxlen = 0
        d = {} # {fruittype:the rightmostlocation of this type of fruit}

        while end < len(fruits):
            d[fruits[end]] = end
            if len(d) >= 3:
                #print(min(d.values()))
                minval = min(d.values()) # 直接删掉最靠左的水果记录，然后从下一个index开始当成start
                del d[fruits[minval]]
                start = minval+1
            maxlen = max(maxlen, end-start+1)
            end +=1
        return maxlen

    def totalFruit(fruits: List[int]) -> int:
        # 一个类似于最短覆盖子串的思路，双指针同时维持一个长度最多为2的字典
        cdict = {}
        i, j = 0, 0
        maxNum = 0

        while j < len(fruits):
            if fruits[j] not in cdict:
                cdict[fruits[j]] = 1
            else:
                cdict[fruits[j]] += 1

            while len(cdict) > 2 and i <= j:
                if fruits[i] in cdict and cdict[fruits[i]] > 1:
                    cdict[fruits[i]] -= 1
                    i += 1
                elif fruits[i] in cdict and cdict[fruits[i]] == 1:
                    del cdict[fruits[i]]
                    i += 1
            if len(cdict) <= 2:
                maxNum = max(maxNum, j - i + 1)
            j += 1
        return maxNum

    print(totalFruit([0,1,2,2]))

def lc_0076():
    def minWindow(s: str, t: str) -> str:
        # 最小覆盖子串问题
        needMap = collections.Counter(t)
        i, j = 0, 0
        res = ""
        minL = float("inf")
        uniqueC = len(needMap)
        while j < len(s):
            if s[j] in needMap:
                needMap[s[j]] -= 1
            if s[j] in needMap and needMap[s[j]] == 0:
                uniqueC -= 1
            while uniqueC == 0:
                if j - i + 1 < minL:
                    minL = j - i + 1
                    res = s[i:j+1]
                    print(res)
                if s[i] in needMap:
                    needMap[s[i]] += 1
                if needMap[s[i]] == 1:
                    uniqueC += 1
                i += 1
            j += 1
        return res

    s = "ADOBECODEBANC"
    t = "ABC"
    print(minWindow(s, t))

def lc_0054():
    def spiralOrder(matrix: List[List[int]]) -> List[int]:
        # 螺旋打印数组
        def rotate(matrix, u, d, l, r, out):
            # top row
            for i in range(l, r):
                out.append(matrix[u][i])
            # right column
            for i in range(u, d):
                out.append(matrix[i][r])
            # bottom row
            for i in range(r, l, -1):
                out.append(matrix[d][i])
            # left column
            for i in range(d, u, -1):
                out.append(matrix[i][l])
            return

        out = []
        m, n = len(matrix), len(matrix[0])
        u, d, l, r = 0, m-1, 0, n-1
        while u < d and l < r:
            rotate(matrix, u, d, l, r, out)
            u += 1
            d -= 1
            l += 1
            r -= 1

        if u == d:
            for i in range(l, r+1):
                out.append(matrix[u][i])
        elif l == r:
            for i in range(u, d+1):
                out.append(matrix[i][l])
        return out

def lc_0059():
    ## 885, 2326 for III and IV
    def generateMatrix(n: int) -> List[List[int]]:
        mat = [[0 for i in range(n)] for j in range(n)]
        u, d, l, r = 0, n-1, 0, n-1
        c = 1
        while u < d and l < r:
            for i in range(l, r):
                mat[u][i] = c
                c += 1
            for i in range(u, d):
                mat[i][r] = c
                c += 1
            for i in range(r, l, -1):
                mat[d][i] = c
                c += 1
            for i in range(d, u, -1):
                mat[i][l] = c
                c += 1
            u += 1
            d -= 1
            l += 1
            r -= 1

        if u == d:
            for i in range(l, r+1):
                mat[u][i] = c
                c += 1
        elif l == r:
            for i in range(u, d+1):
                mat[i][l] = c
                c += 1
        return mat
    print(generateMatrix(3))

def lc_1704():
    def halvesAreAlike(s: str) -> bool:
        # 检查一个string左半边和右半边字符串元音字母数量是不是相等
        vowels = {'a','e','i','o','u','A','E','I','O','U'}
        l, r = 0, len(s) - 1
        lvo, rvo = 0, 0
        while l < r:
            if s[l] in vowels:
                lvo += 1
            if s[r] in vowels:
                rvo += 1
            l += 1
            r -= 1
        return lvo == rvo
    st = "halves"
    print(st)
    print(halvesAreAlike(st))

def lc_0203():
    class ListNode:
        def __init__(self, val, nxt=None):
            self.val = val
            self.next = nxt

    def removeElements(head: Optional[ListNode], val: int) -> Optional[ListNode]:
        if not head:
            return
        dummy = ListNode(0)
        dummy.next = head
        prev = dummy
        curr = head
        while curr:
            if curr.val == val:
                prev.next = curr.next
                curr = curr.next
            else:
                prev = prev.next
                curr = curr.next
        return dummy.next

def lc_0206():
    class ListNode:
        def __init__(self, val, nxt=None):
            self.val = val
            self.next = nxt

    def reverseList(head: Optional[ListNode]) -> Optional[ListNode]:
        if not head:
            return
        prev = None
        curr = head
        while curr:
            next_ = curr.next
            curr.next = prev
            prev = curr
            curr = next_
        return

    def reverseList2(head: Optional[ListNode]) -> Optional[ListNode]:
        def reverse(prev, curr):
            if not curr:
                return prev

            next_ = curr.next
            curr.next = prev
            return reverse(curr, next_)
        return reverse(None, head)

def lc_0707():
    # 链表设计
    # 这是个好题，经常写一写
    class ListNode:
        def __init__(self, val=None, _next=None):
            self.val = val
            self._next = _next

    class MyLinkedList:
        def printLinkedList(self):
            curr = self.head
            while curr:
                print(curr.val)
                curr = curr._next

        def __init__(self):
            # 初始化时只初始化虚拟节点，头节点在后面插入的时候单独设置
            # 注意由于这个设定，所有的add，get和 addat实现都要单独考虑头节点的问题
            self.head = None
            self._dummy = ListNode(val=None, _next=self.head)
            return

        def get(self, index: int) -> int:
            if index < 0:
                # 不合理的index
                return -1

            ret = self.head
            while index > 0 and ret:
                ret = ret._next
                index -= 1

            if index == 0 and ret:
                # index变为0而且当前的位置不是最后面(i.e.当前位置有node)
                return ret.val
            else:
                # index变为0而且当前的位置在最后面(i.e.当前位置没有node)
                return -1

        def addAtHead(self, val: int) -> None:
            newNode = ListNode(val, self.head)
            self._dummy._next = newNode
            self.head = newNode
            return

        def addAtTail(self, val: int) -> None:
            prev = self._dummy
            curr = self.head
            while curr:
                curr = curr._next
                prev = prev._next

            prev._next = ListNode(val)
            if self.head is None:
                # corner case: 当前的链表刚刚初始化或者目前状态只有虚拟节点而无头节点
                # 则检查头节点是否存在，不存在的话把刚加进来的虚拟节点的下一个节点设为头节点
                self.head = prev._next
            return

        def addAtIndex(self, index: int, val: int) -> None:
            if index < 0:
                # 不合理的index
                return

            prev = self._dummy
            curr = self.head
            while curr and index > 0:
                curr = curr._next
                prev = prev._next
                index -= 1

            if index > 0:
                return
            newNode = ListNode(val, curr)
            prev._next = newNode
            if self.head is None:
                # 如果当前的头为空，则是一个空链表，头节点还没设置，需要在这里设置
                self.head = newNode
            return

        def deleteAtIndex(self, index: int) -> None:
            if index < 0:
                # 不合理的index
                return

            prev = self._dummy
            curr = self.head
            while curr and index > 0:
                curr = curr._next
                prev = prev._next
                index -= 1

            if index > 0:
                return
            if curr:
                # 当前位置有node，则连到下一个node
                prev._next = curr._next
            else:
                # 当前位置没有node，不需要任何删除直接返回
                return

            if curr == self.head:
                # 防止删掉的是头节点
                self.head = self.head._next
            del curr
            return

    class MyLinkedList2:
        # 只定义虚拟头节点而不定义真实头节点
        # 这样写比较简洁，试一试能不能通过
        def printLinkedList(self):
            curr = self.head
            while curr:
                print(curr.val)
                curr = curr._next

        def __init__(self):
            self._dummy = ListNode()
            self.size = 0
            return

        def get(self, index: int) -> int:
            if index < 0 or index >= self.size:
                return -1
            curr = self._dummy
            for i in range(index+1):
                curr = curr.val
            return curr.val


        def addAtHead(self, val: int) -> None:
            newNode = ListNode(val)
            if self.size == 0:
                self._dummy._next = newNode
            else:
                newNode._next = self._dummy._next
                self._dummy._next = newNode
            self.size += 1
            return

        def addAtTail(self, val: int) -> None:
            curr = self._dummy
            for i in range(self.size):
                curr = curr._next
            curr._next = ListNode(val)
            self.size += 1
            return

        def addAtIndex(self, index: int, val: int) -> None:
            if index <= 0:
                self.addAtHead(val)
            elif index > self.size:
                return
            elif index == self.size:
                self.addAtTail(val)
            else:
                # [1...size-1]
                newNode = ListNode(val)
                curr = self._dummy
                for i in range(index):
                    curr = curr._next
                newNode._next = curr._next
                curr._next = newNode
                self.size += 1
            return


        def deleteAtIndex(self, index: int) -> None:
            if index < 0 or index >= self.size:
                return

            if index == 0:
                self._dummy._next = self._dummy._next._next
                self.size -= 1
                return
            elif index > 0:
                prev = self._dummy
                curr = self._dummy._next
                for i in range(index):
                    curr = curr._next
                    prev = prev._next
                prev._next =  curr._next
                self.size -= 1
            return

    myLinkedList = MyLinkedList()

    print(myLinkedList.addAtHead(2))
    print(myLinkedList.deleteAtIndex(1))
    print(myLinkedList.addAtHead(2))
    print(myLinkedList.addAtHead(7))
    print(myLinkedList.addAtHead(3))
    print(myLinkedList.addAtHead(2))
    print(myLinkedList.addAtHead(5))
    print(myLinkedList.addAtTail(5))
    print(myLinkedList.get(5))
    print(myLinkedList.deleteAtIndex(6))
    print(myLinkedList.deleteAtIndex(4))

    print("list")
    myLinkedList.printLinkedList()

def lc_0024():
    def swapPairs(head: Optional[ListNode]) -> Optional[ListNode]:
        if not head:
            return
        if not head.next:
            return head
        _dummy = ListNode(None, head)
        p0, p1, p2 = _dummy, head, head.next
        while p1 and p2:
            p0.next = p2
            p1.next = p2.next
            p2.next = p1

            p0 = p0.next.next
            if p2.next and p2.next.next:
                p1 = p2.next.next
            else:
                break
            if p1.next:
                p2 = p1.next
            else:
                break
        return _dummy.next

    def swapPairsClean(head: Optional[ListNode]) -> Optional[ListNode]:
        _dummy = ListNode(None, head)
        prev = _dummy
        while head and head.next:
            first = head
            second = head.next

            prev.next = second
            first.next = second.next
            second.next = first

            prev = head
            head = first.next
        return _dummy.next


    lst = [1,2,3,4,5,6]
    head = ListNode(lst[0])
    curr = head
    for i in range(1, len(lst)):
        node = ListNode(lst[i])
        curr.next = node
        curr = curr.next
    newHead = swapPairsClean(head)
    while newHead:
        print(newHead.val)
        newHead = newHead.next

def lc_0019():
    def removeNthFromEnd(head: Optional[ListNode], n: int) -> Optional[ListNode]:
        _dummy = ListNode(None, head)
        p1 = head
        while n > 0 and p1:
            p1 = p1.next
            n -= 1
        p2 = _dummy.next
        p3 = _dummy
        while p1: # p1会走到最后一个节点的下一个，which is 空，所以p2位置就是要删掉的点
            p1 = p1.next
            p2 = p2.next
            p3 = p3.next

        p3.next = p2.next
        del p2 # 手动释放p2
        ret = _dummy.next
        del _dummy # 手动释放掉_dummy
        return ret

    lst = [1,2,3,4,5,6]
    rm=6
    head = ListNode(lst[0])
    curr = head
    for i in range(1, len(lst)):
        node = ListNode(lst[i])
        curr.next = node
        curr = curr.next
    head = removeNthFromEnd(head, rm)
    while head:
        print(head.val)
        head = head.next


def lc_0160():
    def getIntersectionNode(headA: ListNode, headB: ListNode) -> Optional[ListNode]:
        p1 = headA
        p2 = headB
        n1, n2 = 1, 1
        while p1.next:
            n1 += 1
            p1 = p1.next
        while p2.next:
            n2 += 1
            p2 = p2.next
        if p1 != p2:
            return
        diff = abs(n1 - n2)

        p1 = headA
        p2 = headB
        if n1 >= n2:
            while diff > 0:
                p1 = p1.next
                diff -= 1
        elif n1 < n2:
            while diff > 0:
                p2 = p2.next
                diff -= 1
        while p1 != p2:
            p1 = p1.next
            p2 = p2.next
        return p1

    lst = [4,1,8,4,5]
    headA = ListNode(lst[0])
    curr = headA
    for i in range(1, len(lst)):
        node = ListNode(lst[i])
        curr.next = node
        curr = curr.next
    headB = ListNode(5)
    headB.next = ListNode(6)
    headB.next.next = ListNode(1)
    headB.next.next.next = headA.next.next
    Inter = getIntersectionNode(headA, headB)
    print(Inter.val)

def lc_0142():
    def detectCycle(head: Optional[ListNode]) -> Optional[ListNode]:
        if not head:
            return None
        first = True
        f, s = head, head
        while f.next and f.next.next and (first or f != s):
            s = s.next
            f = f.next.next
            first = False

        if f != s or first:
            return None

        f = head
        while f != s:
            f = f.next
            s = s.next
        return f


def lc_0242():
    from collections import Counter
    def isAnagram(s: str, t: str) -> bool:
        # 同分异构词
        if len(s) != len(t):
            return False
        sC = Counter(s)
        tC = Counter(t)
        for k, v in sC.items():
            if k not in tC or tC[k] != v:
                return False
        return True


def lc_0349():
    def intersection(nums1: List[int], nums2: List[int]) -> List[int]:
        # 两个数组的交集
        # 和350是一个题，350需要考虑重复
        # s1 = set(nums1)
        # s2 = set(nums2)
        # return list(s1.intersection(s2))
        s1 = {}
        ret = []
        for num in nums1:
            s1[num] = 1
        for num in nums2:
            if num in s1 and s1[num] == 1:
                ret.append(num)
                s1[num] = 0
                # 350: s1[num] -= 1
        return ret


def lc_0202():
    def isHappy(n: int) -> bool:
        # 快乐数
        s = set()
        ssum = 0
        while n:
            nr = n % 10
            ssum += nr ** 2
            n = n // 10

        while ssum not in s and ssum != 1:
            s.add(ssum)
            n = ssum
            ssum = 0
            while n:
                nr = n % 10
                ssum += nr ** 2
                n = n // 10

        return ssum == 1

    for i in range(1000):
        if isHappy(i):
            print(i)

def lc_0454():
    def fourSumCount(nums1: List[int], nums2: List[int], nums3: List[int], nums4: List[int]) -> int:
        # 四个数组的四和问题
        count = 0
        AB = {}
        for a in nums1:
            for b in nums2:
                if a+b in AB:
                    AB[a+b] += 1
                else:
                    AB[a+b] = 1
        for c in nums3:
            for d in nums4:
                if 0-c-d in AB:
                    count += AB[0-c-d]

        return count


def lc_0383():
    from collections import Counter
    def canConstruct(self, ransomNote: str, magazine: str) -> bool:
        #杂志选字母拼字问题，可以使用数组实现因为题目规定了只有小写字母
        cmag = Counter(magazine)
        for c in ransomNote:
            if c not in cmag or cmag[c] == 0:
                return False
            cmag[c] -= 1
        return True


def lc_0015():
    def threeSum(nums: List[int]) -> List[List[int]]:
        # 三和
        # s双指针法，加一些去重的逻辑会显著快于使用哈希结构的解法
        res = []
        n = len(nums)
        nums.sort()
        for i in range(n):
            if nums[i] > 0:
                break
            if i > 0 and nums[i] == nums[i-1]:
                continue

            l = i + 1
            r = n - 1
            while l < r:
                if nums[i] + nums[l] + nums[r] == 0:
                    res.append([nums[i], nums[l], nums[r]])
                    while l != r and nums[l+1] == nums[l]: l += 1
                    while l != r and nums[r-1] == nums[r]: r -= 1
                    l += 1
                    r -= 1
                elif nums[i] + nums[l] + nums[r] > 0:
                    r -= 1
                else:
                    l += 1
            return res

        return res


def lc_0018():
    def fourSum(nums: List[int], target: int) -> List[List[int]]:
        # 四和，本质和三和一样，要额外考虑合理的去重方法
        n = len(nums)
        nums.sort()
        res = []
        for i in range(n):
            if nums[i] > target and nums[i] >= 0:
                # >=0是因为如果小于0且target也是小于0的数的话光靠nums[i]>target就break掉会漏解，因为后面可能有负树数把整体和继续变小
                break
            if i > 0 and nums[i] == nums[i-1]:
                continue
            for j in range(i+1, n):
                if nums[i] + nums[j] > target and (nums[i] + nums[j] >= 0):
                    # >=0是因为如果小于0且target也是小于0的数的话光靠nums[i]>target就break掉会漏解，因为后面可能有负树数把整体和继续变小
                    break
                if j > i+1 and nums[j] == nums[j-1]:
                    continue
                l = j + 1
                r = n - 1
                while l < r:
                    if nums[i] + nums[j] + nums[l] + nums[r] == target:
                        res.append([nums[i], nums[j], nums[l], nums[r]])
                        while l != r and nums[l] == nums[l+1]: l += 1
                        while l != r and nums[r] == nums[r-1]: r -= 1
                        l += 1
                        r -= 1
                    elif nums[i] + nums[j] + nums[l] + nums[r] > target:
                        r -= 1
                    else:
                        l += 1
        return res


def lc_0344():
    def reverseString(s: List[str]) -> None:
        # inplace 反转字符数组
        """
        Do not return anything, modify s in-place instead.
        """
        i, j = 0, len(s) - 1
        while i < j:
            s[i], s[j] = s[j], s[i]
            i += 1
            j -= 1
        return


def lc_0541():
    def reverseStr(s: str, k: int) -> str:
        # 字符串每隔2k个元素反转前k个字符， 若str长度小于k则全部反转
        def reverse(lst, start, end):
            # helper function to revert the element in lst [start...end]
            if end >= len(lst):
                end = len(lst) - 1
            while start < end:
                lst[start], lst[end] = lst[end], lst[start]
                start += 1
                end -= 1
            return
        n = len(s)
        charLst = list(s)
        for i in range(0, n, 2*k):
            start = i
            end = i + k - 1
            reverse(charLst, start, end)
        return "".join(charLst)


def lc_cn_offer05():
    def replaceString(s):
        # 把字符串中的空格替换成%20，返回新字符串
        sList = list(s)
        n = 0
        for ele in sList:
            if ele == " ":
                n += 1
        sList += [None] * 2 * n
        p1, p2 = len(s) - 1, len(sList) - 1
        while p1 >= 0:
            if s[p1] == " ":
                sList[p2] = "0"
                sList[p2-1] = "2"
                sList[p2-2] = "%"
                p1 -= 1
                p2 -= 3
            else:
                sList[p2] = s[p1]
                p1 -= 1
                p2 -= 1
        return "".join(sList)
    print(replaceString("We are cham pion."))


def lc_0151():
    # 这个题目用python的reverse函数效率很高，应该用c++从头实现一下，问到这个问题大概率需要从头实现
    def reverseWords(s: str) -> str:
        def removeExtraSpace(charList):
            idx, i = 0, 0
            while i < len(charList):
                if charList[i] != " ":
                    if idx != 0:
                        # 最前面的空格不需要，只有中间的单词之间需要一个空格
                        charList[idx] = " "
                        idx += 1
                    while i < len(charList) and charList[i] != " ":
                        # 最后面位置的空格不需要，快慢指针停在最后一个非空的位置的下一个位置
                        charList[idx] = charList[i]
                        i += 1
                        idx += 1
                else:
                    i += 1
            # 返回[:idx],不会包含idx位置字符
            return charList[:idx]


        def reverseCharList(charList, start, end):
            i, j = start, end
            if end > len(charList):
                end = len(charList)-1
            while i < j:
                charList[i], charList[j] = charList[j], charList[i]
                i += 1
                j -= 1
            return

        cList = list(s)
        cList = removeExtraSpace(cList)
        s,e = 0,0
        while e < len(cList):
            while e < len(cList) and cList[e] != " ":
                e += 1
            reverseCharList(cList, s, e-1)
            e += 1
            s = e
        reverseCharList(cList, 0, len(cList)-1)
        return "".join(cList)

    print(reverseWords("a good   example  "))


def lc_cn_offer58():
    def leftRotate(charList, num):
        def reverseCharList(start, end, charList):
            i, j = start, end
            while i < j:
                charList[i], charList[j] = charList[j], charList[i]
                i += 1
                j -= 1
            return


        reverseCharList(0, num-1, charList)
        reverseCharList(num, len(charList)-1, charList)
        reverseCharList(0, len(charList)-1, charList)
        return
    input_str = list("astringtoberotated")
    leftRotate(input_str, 3)
    print("".join(input_str))

def lc_0028():
    def getNext(str2):
        res = [None for _ in range(len(str2))]
        if len(res) == 1:
            res[0] = -1
            return

        res[0] = -1
        res[1] = 0
        cn = 0
        i = 2
        while i < len(str2):
            if str2[cn] == str2[i - 1]:
                cn += 1
                res[i] = cn
                i += 1
            elif cn > 0:
                cn = res[cn]
            else:
                res[i] = 0
                i += 1
        return res


    def strStr(haystack: str, needle: str) -> int:
        if haystack is None or needle is None or len(needle) > len(haystack):
            return -1
        next_ = getNext(needle)
        i1, i2 = 0, 0
        while i1 < len(haystack) and i2 < len(needle):
            if haystack[i1] == needle[i2]:
                i1 += 1
                i2 +=1
            elif i2 != 0:
                i2 = next_[i2]
            else:
                i1 += 1
        return i1 - i2 if i2 == len(needle) else -1

    print(strStr("abababa", "bab"))


def lc_0459():
    def kmp(str1):
        def getNext(str2):
            res = [None for _ in range(len(str2))]
            if len(res) == 1:
                res[0] = -1
                return res
            res[0] = -1
            res[1] = 0
            i = 2
            cn = 0
            while i < len(str2):
                if str2[cn] == str2[i-1]:
                    cn += 1
                    res[i] = cn
                    i += 1
                elif cn > 0:
                    cn = res[cn]
                else:
                    res[i] = 0
                    i += 1
            return res


        if str1 is None:
            return -1

        next_arr = getNext(str1)
        str1_ = 2*str1[1:-1]
        i1, i2 = 0, 0
        while i1 < len(str1_) and i2 < len(str1):
            if str1_[i1] == str1[i2]:
                i1 += 1
                i2 += 1
            elif i2 != 0:
                i2 = next_arr[i2]
            else:
                i1 += 1
        return True if i2 == len(str1) else False

    def repeatedSubstringPattern(s: str) -> bool:
        return (s+s)[1:-1].find(s) != -1

    print(kmp("abababa"))


def lc_0232():
    # 用栈实现队列
    class MyQueue:
        def __init__(self):
            self.cs = deque([])
            self.ps = deque([])


        def push(self, x: int) -> None:
            self.cs.append(x)
            return


        def pop(self) -> int:
            self._move()
            return self.ps.pop()



        def peek(self) -> int:
            self._move()
            return self.ps[-1]


        def empty(self) -> bool:
            return len(self.cs) == 0 and len(self.ps) == 0

        def _move(self):
            if len(self.ps) == 0:
                while self.cs:
                    self.ps.append(self.cs.pop())
            return


def lc_0225():
    # 用队列实现栈
    class MyStack:

        def __init__(self):
            self.main = deque([])
            self.bu = deque([])
            return

        def push(self, x: int) -> None:
            self.main.append(x)


        def pop(self) -> int:
            while len(self.main) > 1:
                self.bu.append(self.main.popleft())
            ret = self.main.popleft()
            while self.bu:
                self.main.append(self.bu.popleft())
            return ret

        def top(self) -> int:
            return self.main[-1]


        def empty(self) -> bool:
            return len(self.main) == 0 and len(self.bu) == 0


def lc_0020():
    # 字符串中括号是否正确匹配
    def isValid(s: str) -> bool:
        def match(a, b):
            if a + b in {"()", "{}", "[]"}:
                return True
            else:
                return False

        stack = deque([])
        for e in s:
            if e in {"[", "{", "("}:
                stack.append(e)
            elif e in {"]", "}", ")"}:
                if len(stack) == 0 or not match(stack.pop(), e):
                    return False
        return len(stack) == 0

def lc_1047():
    # 字符消消乐
    def removeDuplicates(s: str) -> str:
        stack = []
        for e in s:
            if len(stack) == 0 or stack[-1] != e:
                stack.append(e)
            else:
                stack.pop()
        return "".join(stack)


def lc_0150():
    # 逆波兰表达式求值
    def evalRPN(tokens: List[str]) -> int:
        stack = []
        for token in tokens:
            if token not in {"+", "-", "*", "/"}:
                stack.append(int(token))
            else:
                if len(stack) < 2:
                    return None
                n2 = stack.pop()
                n1 = stack.pop()
                if token == "+":
                    stack.append(n1 + n2)
                elif token == "-":
                    stack.append(n1 - n2)
                if token == "*":
                    stack.append(n1 * n2)
                if token == "/":
                    stack.append(int(n1 / n2))
        return stack[0]


def lc_0239():
    def maxSlidingWindow(nums: List[int], k: int) -> List[int]:
        class monoQueue:
            def __init__(self):
                self.q = deque([])
                return
            def pop(self, val):
                while self.q and val == self.q[0]:
                    self.q.popleft()
                return
            def push(self, val):
                while self.q and self.q[-1] < val:
                    self.q.pop()
                self.q.append(val)
                return
            def front(self):
                return self.q[0]

        mq = monoQueue()
        ret = [None for _ in range(len(nums)-k+1)]

        for i, ele in enumerate(nums):
            if i < k-1:
                mq.push(ele)
            elif i == k-1:
                mq.push(ele)
                ret[i-k+1] = mq.front()
            else:
                mq.pop(nums[i-k])
                mq.push(ele)
                ret[i-k+1] = mq.front()
        return ret


def lc_0347():
    from collections import Counter
    def topKFrequent(nums: List[int], k: int) -> List[int]:
        # 前K个最经常出现的字符
        mp = Counter(nums)
        topK = []
        heapq.heapify(topK)
        for val, fr in mp.items():
            # 根据freq进行heap的组织排序， 默认小根堆
            heapq.heappush(topK, (fr, val))
            if len(topK) > k:
                # 长度只要超过k就pop掉频率最低的， 这里len最多比k多1个，把这个东西当成是stack就好了
                heapq.heappop(topK)
        res = [val for fr, val in topK]
        return res
    topKFrequent([5,3,1,1,1,3,73,1], 2)


def lc_0144():
    def preorderTraversal(root: Optional[BinaryTree]) -> List[int]:
        vec = []
        if not root:
            return vec
        s = deque([root])
        while s:
            cur = s.pop()
            vec.append(cur.value)
            if cur.right:
                s.append(cur.right)
            if cur.left:
                s.append(cur.left)

        return vec


def lc_0094():
    def inorderTraversal(root: Optional[BinaryTree]) -> List[int]:
        vec = []
        if not root:
            return vec

        s = deque([])
        while root:
            s.append(root)
            root = root.left

        while s:
            cur = s.pop()
            vec.append(cur.value)
            cur_R = cur.right
            while cur_R:
                s.append(cur_R)
                cur_R = cur_R.left

        return vec


def lc_0145():
    def postorderTraversal(root: Optional[BinaryTree]) -> List[int]:
        vec = []
        if not root:
            return vec

        cs = deque([root])
        ps = deque([])
        while cs:
            cur = cs.pop()
            ps.append(cur.value)
            if cur.left:
                cs.append(cur.left)
            if cur.right:
                cs.append(cur.right)
        while ps:
            vec.append(ps.pop())
        return vec


def lc_0102():
    '''
    102.二叉树的层序遍历
    :return:
    '''

    from collections import deque
    def levelOrder(root: Optional[BinaryTree]) -> List[List[int]]:
        ret = []
        if not root:
            return ret
        q = deque([root])
        while q:
            l = len(q)
            curlevel = []
            for i in range(l):
                cur = q.popleft()
                curlevel.append(cur.value)
                if cur.left:
                    q.append(cur.left)
                if cur.right:
                    q.append(cur.right)
            ret.append(curlevel)
        return ret


def lc_0107():
    '''
    107.二叉树的层次遍历II
    :return:
    '''
    from collections import deque
    def levelOrderBottom(root: Optional[BinaryTree]) -> List[List[int]]:
        ret = []
        if not root:
            return ret
        q = deque([root])
        while q:
            l = len(q)
            curlevel = []
            for i in range(l):
                cur = q.popleft()
                curlevel.append(cur.value)
                if cur.left:
                    q.append(cur.left)
                if cur.right:
                    q.append(cur.right)
            ret.append(curlevel)
        return reversed(ret) # 可以骗过leetcode OJ但是返回的是个iterator不是list


def lc_0199():
    '''
    199.二叉树的右视图
    :return:
    '''
    def rightSideView(root: Optional[BinaryTree]) -> List[int]:
        ret = []
        if not root:
            return ret
        q = deque([root])
        while q:
            l = len(q)
            for i in range(l):
                cur = q.popleft()
                if i == l - 1:
                    ret.append(cur.value)
                if cur.left:
                    q.append(cur.left)
                if cur.right:
                    q.append(cur.right)
        return ret


def lc_0637():
    '''
    637.二叉树的层平均值
    :return:
    '''
    def averageOfLevels(root: Optional[BinaryTree]) -> List[float]:
        ret = []
        if not root:
            return ret
        q = deque([root])
        while q:
            l = len(q)
            acc = 0
            ct = 0
            for i in range(l):
                cur = q.popleft()
                acc += cur.value
                ct += 1
                if cur.left: q.append(cur.left)
                if cur.right: q.append(cur.right)
            ret.append(acc/ct)
        return ret


def lc_0429():
    '''
    429.N叉树的层序遍历
    :return:
    '''
    class Node:
        def __init__(self, val=None, children=None):
            self.val = val
            self.children = children

    def levelOrder(root: Node) -> List[List[int]]:
        ret = []
        if not root:
            return ret
        q = deque([root])
        while q:
            l = len(q)
            curlevel = []
            for i in range(l):
                cur = q.popleft()
                curlevel.append(cur.val)

                if cur.children:
                    for child in cur.children:
                        if child: q.append(child)
            ret.append(curlevel)
        return ret


def lc_0515():
    '''
    515.在每个树行中找最大值
    :return:
    '''
    def largestValues(root: Optional[BinaryTree]) -> List[int]:
        ret = []
        if not root:
            return ret

        q = deque([root])
        while q:
            l = len(q)
            mx = float('-inf')
            for i in range(l):
                cur = q.popleft()
                mx = max(mx, cur.value)
                if cur.left: q.append(cur.left)
                if cur.right: q.append(cur.right)
            ret.append(mx)
        return ret


def lc_0116():
    '''
    116.填充每个节点的下一个右侧节点指针
    117.填充每个节点的下一个右侧节点指针II
    代码是一样的但是题目有些许不一样，116假设树是完美的，117是一般的
    :return:
    '''
    class Node:
        def __init__(self, val: int = 0, left: 'Node' = None, right: 'Node' = None, next: 'Node' = None):
            self.val = val
            self.left = left
            self.right = right
            self.next = next

    def connect(root: Optional[Node]) -> Optional[Node]:
        if not root:
            return
        q = deque([root])
        while q:
            l = len(q)
            _dummy = Node(None, None, None, None)
            prev = _dummy
            for i in range(l):
                cur = q.popleft()
                prev.next = cur
                if cur.left: q.append(cur.left)
                if cur.right: q.append(cur.right)
                prev = cur
            del _dummy
        return root


def lc_0104():
    '''
    104.二叉树的最大深度
    :return:
    '''
    def maxDepth(self, root: Optional[BinaryTree]) -> int:
        maxDepth = 0
        if not root:
            return maxDepth
        q = deque([root])
        while q:
            l = len(q)
            for i in range(l):
                cur = q.popleft()
                if cur.left: q.append(cur.left)
                if cur.right: q.append(cur.right)
            maxDepth += 1
        return maxDepth


def lc_0111():
    '''
    111.二叉树的最小深度
    :return:
    '''
    def minDepthTemplate(root: Optional[BinaryTree]) -> int:
        # 树形dp解法，需要考虑特殊的链表形式的树，不是太直观但是也能work，而且实际观测到的时间比层序遍历要慢一些
        class Info:
            def __init__(self, depth):
                self.depth = depth
        def md(node):
            if not node:
                return Info(0)
            l = md(node.left)
            r = md(node.right)
            # 防止出现只有一个branch时某一侧树深度为0的情况
            if l.depth == 0:
                return Info(1+r.depth)
            if r.depth == 0:
                return Info(1+l.depth)
            return Info(1+min(r.depth, l.depth))

        return md(root).depth


    def minDepth(self, root: Optional[BinaryTree]) -> int:
        ret = 0
        if not root:
            return ret

        q = deque([root])
        while q:
            l = len(q)
            ret += 1
            for i in range(l):
                cur = q.popleft()
                if (not cur.left) and (not cur.right):
                    return ret
                if cur.left:
                    q.append(cur.left)
                if cur.right:
                    q.append(cur.right)
        return ret


def lc_0226():
    '''
    226.翻转二叉树
    :return:
    '''
    def invertTreeRecursive(self, root: Optional[BinaryTree]) -> Optional[BinaryTree]:
        def inv(node):
            # 前序后序
            if not node:
                return
            # node.left, node.right = node.right, node.left # 或这里
            inv(node.left)
            inv(node.right)
            node.left, node.right = node.right, node.left
            return

        def inv2(node):
            # 中序
            if not node:
                return
            inv2(node.left)
            node.left, node.right = node.right, node.left
            inv2(node.left)
            return

        inv(root)
        return root

    def invertTreeIterative(self, root: Optional[BinaryTree]) -> Optional[BinaryTree]:
        if not root:
            return

        s = deque([root])
        while s:
            cur = s.pop()
            cur.left, cur.right = cur.right, cur.left
            if cur.right: s.append(cur.right)
            if cur.left: s.append(cur.left)
        return root


def lc_0101():
    '''
    验证轴对称的二叉树
    :return:
    '''
    def isSymmetric(root: Optional[BinaryTree]) -> bool:
        def isValid(n1, n2):
            if (not n1) and (not n2):
                return True
            elif n1 and (not n2):
                return False
            elif n2 and (not n1):
                return False
            else:
                return (n1.value == n2.value) and isValid(n1.left, n2.right) and isValid(n1.right, n2.left)

        return isValid(root, root)


def lc_0100():
    '''
    100: 验证相同二叉树
    :return:
    '''
    def isSameTree(p: Optional[BinaryTree], q: Optional[BinaryTree]) -> bool:
        def valid(n1, n2):
            if (not n1) and (not n2):
                return True
            elif n1 and (not n2):
                return False
            elif n2 and (not n1):
                return False
            else:
                return (n1.value == n2.value) and valid(n1.left, n2.left) and valid(n1.right, n2.right)
        return valid(p, q)


def lc_0572():
    '''
    572.另一个树的子树,给根节点a，b查看b是不是a的一个子树
    :return:
    '''
    def isSubtree(root: Optional[BinaryTree], subRoot: Optional[BinaryTree]) -> bool:

        def same(n1, n2):
            if (not n1) and (not n2): return True
            elif n1 and (not n2): return False
            elif n2 and (not n1): return False
            else:
                return (n1.value == n2.value) and same(n1.left, n2.left) and same(n1.right, n2.right)

        if not root:
            return False
        # 前序遍历每一个点看是不是子树
        if same(root, subRoot): return True
        if isSubtree(root.left, subRoot): return True
        if isSubtree(root.right, subRoot): return True
        return False


def lc_0559():
    '''
    559.n叉树的最大深度
    :return:
    '''
    class Node:
        def __init__(self, val=None, children=None):
            self.val = val
            self.children = children

    def maxDepth(root: Node) -> int:
        mxd = 0
        if not root: return mxd
        q = deque([root])
        while q:
            l = len(q)
            for i in range(l):
                cur = q.popleft()
                for c in cur.children:
                    if c: q.append(c)
            mxd += 1
        return mxd


def lc_0222():
    '''
    222.完全二叉树的节点个数,时间复杂度小于o(n)的解法
    :return:
    '''
    def countNodes(root: Optional[BinaryTree]) -> int:
        if not root:
            return 0

        ld, rd = 1, 1
        l = root.left
        r = root.right
        # 这里在找当前节点左右最大深度，如果相等，说明这个节点的树是满的，那么就不用遍历累加有几个点了，直接用公式算
        while l:
            ld += 1
            l = l.left
        while r:
            rd += 1
            r = r.right
        if ld == rd: return 2 ** ld - 1
        return countNodes(root.left) + countNodes(root.right) + 1


def lc_0110():
    '''
    验证平衡二叉树：任意一个点为头的树，左树右树高度相差最多为1即为平衡二叉树
    A height-balanced binary tree is a binary tree in which the depth of the two subtrees of every node never differs by
    more than one.
    :return:
    '''
    def isBalanced(root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        class Info:
            def __init__(self, isB, H):
                self.isB = isB
                self.H = H

        def isBal(node):
            if not node:
                return Info(True, 0)

            li = isBal(node.left)
            ri = isBal(node.right)

            isB = li.isB and ri.isB and abs(li.H - ri.H) <= 1
            H = max(li.H, ri.H) + 1
            return Info(isB, H)

        return isBal(root).isB


def lc_0257():
    '''
    257. 二叉树的所有路径
    :return:
    '''
    def binaryTreePaths(root: Optional[BinaryTree]) -> List[str]:
        """
        :type root: TreeNode
        :rtype: List[str]
        """
        def binaryTP(node, cur, res):
            if (not node.left) and (not node.right):
                cur += str(node.value)
                res.append(cur)
                return
            if node.left:
                binaryTP(node.left, cur + f"{node.value}->", res)
            if node.right:
                binaryTP(node.right, cur + f"{node.value}->", res)
            return

        res = []
        binaryTP(root, "", res)
        return res

def lc_0404():
    '''
    404.左叶子之和
    :return:
    '''
    def sumOfLeftLeaves(root: Optional[BinaryTree]) -> int:
        if not root:
            return 0
        if (not root.left) and (not root.right):
            # 叶节点没有左叶节点和
            return 0
        l, r = 0, 0
        l = sumOfLeftLeaves(root.left)
        if root.left and (not root.left.left) and (not root.left.right):
            # 找到左叶子节点的处理
            l = root.left.value
        # 右侧的值
        r = sumOfLeftLeaves(root.right)
        # 没找到左叶子的话就往下一层找左叶子，找到了左叶子的话l2根据定义下面会返回0

        return l + r


def lc_0513():
    '''
    513.找树左下角的值
    :return:
    '''
    def findBottomLeftValueRecursive(self, root: Optional[BinaryTree]) -> int:
        # 没这个必要，这个题用层序遍历很好理解
        mx = float("-inf")
        lm = 0
        def depth(node, dept):
            nonlocal mx, lm
            if (not node.left) and (not node.right):
                if dept > mx:
                    mx = dept
                    lm = node.value
            if node.left:
                depth(node.left, dept + 1)
            if node.right:
                depth(node.right, dept + 1)

        depth(root, 0)
        return lm

    def findBottomLeftValue(root: Optional[BinaryTree]) -> int:
        ret = 0
        if not root:
            return ret
        q = deque([root])
        while q:
            l = len(q)
            for i in range(l):
                cur = q.popleft()
                if i == 0:
                    ret = cur.value
                if cur.left:
                    q.append(cur.left)
                if cur.right:
                    q.append(cur.right)
        return ret


def lc_0112():
    '''
    112. 返回二叉树中是否有一条路径沿途总和等于target
    :return:
    '''
    def hasPathSum(root: Optional[BinaryTree], targetSum: int) -> bool:
        if not root:
            return False
        if (not root.left) and (not root.right):
            return root.value == targetSum
        if (not root.left) and (not root.right):
            return False
        elif not root.left:
            return hasPathSum(root.right, targetSum - root.value)
        elif not root.right:
            return hasPathSum(root.left, targetSum - root.value)
        else:
            return hasPathSum(root.left, targetSum - root.value) or hasPathSum(root.right, targetSum - root.value)


def lc_0113():
    '''
    113. 返回所有二叉树中路径总和等于target的路径
    :return:
    '''
    def pathSum(root: Optional[BinaryTree], targetSum: int) -> List[List[int]]:
        def pS(node, targetSum, curpath, res):
            # if not node:
            #     return
            if (not node.left) and (not node.right):
                if node.value == targetSum:
                    curpath += [node.val]
                    res.append(curpath)
            else:
                if node.left:
                    pS(node.left, targetSum - node.value, curpath + [node.value], res)
                if node.right:
                    pS(node.right, targetSum - node.value, curpath + [node.value], res)
            return
        res = []
        if not root:
            return res
        pS(root, targetSum, [], res)
        return res


def lc_0106():
    '''
    106.中序后序数组构建二叉树
    :return:
    '''
    def buildTree(inorder: List[int], postorder: List[int]) -> Optional[BinaryTree]:
        if len(postorder) == 0: return None
        rootVal = postorder[-1]
        root = BinaryTree(rootVal)
        if len(postorder) == 1: return root

        i = 0
        while i < len(inorder):
            if inorder[i] == rootVal:
                break
            i += 1

        # 这里切割时假设[start:end)
        lin = inorder[0:i]
        rin = inorder[i+1:]
        lpost = postorder[0:i]
        rpost = postorder[i:-1]

        root.left = buildTree(lin, lpost)
        root.right = buildTree(rin, rpost)
        return root

    def buildTreePtr(inorder, postorder, inl, inr, pol, por):
        # 根据我的习惯使用左闭右闭区间
        # 当前处理的中序部分为inorder[inl:inr]包含两个边界点， 后续部分postorder[pol:por]包含两个边界点
        if pol > por: return None
        rootVal = postorder[por]
        root = BinaryTree(rootVal)
        if por == pol: return root


        dist = 0 # 定义这个距离来计算从当前中序部分起点到找到中序root要经过几个元素， 后面的index都是根据这个距离退出来的
        while inl + dist < inr:
            if inorder[inl + dist] == rootVal:
                break
            dist += 1

        linl, linr = inl, inl + dist - 1
        rinl, rinr = inl + dist + 1, inr
        lpol, lpor = pol, pol + dist - 1
        rpol, rpor = pol + dist, por - 1

        # print(inorder[linl:linr+1])
        # print(inorder[rinl:rinr+1])
        # print(postorder[lpol:lpor+1])
        # print(postorder[rpol:rpor+1])

        root.left = buildTreePtr(inorder, postorder, linl, linr, lpol, lpor)
        root.right = buildTreePtr(inorder, postorder, rinl, rinr, rpol, rpor)
        return root


    inorder = [9,3,15,20,7]
    postorder = [9,15,7,20,3]
    root = buildTreePtr(inorder, postorder, 0, len(inorder)-1, 0, len(postorder)-1)
    printBinaryTree(root)


def lc_0105():
    '''
    105.前序中序数组构建二叉树
    :return:
    '''
    def buildTree(self, preorder: List[int], inorder: List[int]) -> Optional[BinaryTree]:
        def buildTreePtr(inorder, preorder, lin, rin, lpre, rpre):
            if lpre > rpre: return None
            rootVal = preorder[lpre]
            root = BinaryTree(rootVal)
            if lpre == rpre: return root

            dist = 0
            while lin + dist <= rin:
                if inorder[lin + dist] == rootVal:
                    break
                dist += 1
            linl, linr = lin, lin + dist - 1
            rinl, rinr = lin + dist + 1, rin
            lprel, lprer = 1 + lpre, 1 + lpre + dist - 1
            rprel, rprer = 1 + lpre + dist, rpre

            root.left = buildTreePtr(inorder, preorder, linl, linr, lprel, lprer)
            root.right = buildTreePtr(inorder, preorder, rinl, rinr, rprel, rprer)
            return root

        return buildTreePtr(inorder, preorder, 0, len(inorder)-1, 0, len(preorder)-1)

def lc_0654():
    '''
    654.最大二叉树:给定一个不含重复元素的整数数组。返回一个以此数组构建的最大二叉树的根节点
    :return:
    '''
    def constructMaximumBinaryTree(nums: List[int]) -> Optional[BinaryTree]:
        def construct(nums, l, r):
            if l > r: return None
            rootVal = max(nums[l: r+1])
            root = BinaryTree(rootVal)
            if l == r: return root
            idx = l
            for idx in range(l, r+1):
                if nums[idx] == rootVal:
                    break

            root.left = construct(nums, l, idx-1)
            root.right = construct(nums, idx + 1, r)
            return root

        return construct(nums, 0, len(nums)-1)

    printBinaryTree(constructMaximumBinaryTree([6,8,5,9,0,11,10,3,4,2,-1]))


def lc_0617():
    '''
    617.合并二叉树,尽量利用已有的点不要创新的点
    给定两个二叉树，想象当你将它们中的一个覆盖到另一个上时，两个二叉树的一些节点便会重叠。
    你需要将他们合并为一个新的二叉树。合并的规则是如果两个节点重叠，那么将他们的值相加作为节点合并后的新值，否则不为 NULL 的节点将直接作为新二叉树的节点。
    :return:
    '''
    def mergeTrees(root1: Optional[BinaryTree], root2: Optional[BinaryTree]) -> Optional[BinaryTree]:

        if (not root1) and (not root2): return None
        elif not root1:
            return root2
        elif not root2:
            return root1
        else:
            root1.value += root2.value

            root1.left = mergeTrees(root1.left, root2.left)
            root1.right = mergeTrees(root1.right, root2.right)
            return root1

def lc_0700():
    def searchBST(root: Optional[BinaryTree], val: int) -> Optional[BinaryTree]:
        while root:
            if val == root.value:
                return root
            elif val < root.value:
                root = root.left
            else:
                root = root.right
        return root

def lc_0530():
    '''
    530.二叉搜索树的最小绝对差,利用中序遍历记录前一个节点值并每次更新最小绝对差
    :return:
    '''
    def getMinimumDifference(root: Optional[BinaryTree]) -> int:
        diff = float("inf")
        prev = float("-inf")
        def getDiff(root):
            nonlocal diff, prev
            if not root: return None
            getDiff(root.left)
            diff = min(diff, abs(root.value - prev))
            prev = root.value
            getDiff(root.right)
            return

        getDiff(root)
        return diff


def lc_0501():
    def findModeExtraSpace(root: Optional[BinaryTree]) -> List[int]:
        mode = {}
        res = []
        if not root:
            return res
        s = deque([])
        while root:
            s.append(root)
            root = root.left
        while s:
            cur = s.pop()
            if cur.value not in mode:
                mode[cur.value] = 1
            else:
                mode[cur.value] += 1
            cur_R = cur.right
            while cur_R:
                s.append(cur_R)
                cur_R = cur_R.left
        mxVal = max(mode.values())
        for k, v in mode.items():
            if v == mxVal:
                res.append(k)
        return res

    def findMode(self, root: Optional[BinaryTree]) -> List[int]:
        # 中序遍历并使用有限个变量，clear的时间复杂度取决于原树中重复数字的多少以及对应频率是否一致，频率越一致则代价越小
        maxCount = 0
        count = 0
        preval = None
        res = []
        if not root:
            return res
        s = deque([])
        while root:
            s.append(root)
            root = root.left
        while s:
            cur = s.pop()
            if preval is None:
                count = 1
            elif preval == cur.value:
                count += 1
            else:
                count = 1
            preval = cur.value
            if count > maxCount:
                maxCount = count
                res.clear()
                res.append(cur.value)
            elif count == maxCount:
                res.append(cur.value)


def lc_0236():
    '''
    236. 二叉树的最近公共祖先
    :return:
    '''

    def lowestCommonAncestor(self, root: BinaryTree, p: BinaryTree, q: BinaryTree) -> BinaryTree:
        if root == p or root == q or (not root):
            # 当前根节点是p或q，或者根节点为空
            return root
        # 找左右子树的LCA
        l = self.lowestCommonAncestor(root.left, p, q)
        r = self.lowestCommonAncestor(root.right, p, q)

        # 如果左右子树LCA均为非空则说明要找的p和q分别在左右子树中，此时当前根节点为LCA
        if l and r: return root
        # 如果当前任何一个子树根LCA为空，说明p，q均在另一棵子树中，对应子树根为LCA
        return l if l else r


def lc_0235():
    '''
    235. BST的最低公共祖先：题目给出p和q都是BST中的点所以不需要考虑空的或者p,q不属于BST的情况，
    由于BST，所以可以根据p,q,root的值的大小关系判断根的往哪个方向递归寻找LCA
    思路：BST值搜索，找第一个节点值使得其在p,q点的值之间，则找到的节点为LCA
    :return:
    '''
    def lowestCommonAncestor(root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        if p.val < root.val and q.val < root.val:
            return lowestCommonAncestor(root.left, p, q)
        elif p.val > root.val and q.val > root.val:
            return lowestCommonAncestor(root.right, p, q)
        else:
            return root

    def lowestCommonAncestorIterative(root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        while True:
            if root.val < p.val and root.val <  q.val:
                root = root.right
            elif root.val > p.val and root.val > q.val:
                root = root.left
            else:
                return root

def lc_0701():
    '''
    701.二叉搜索树中的插入操作
    :return:
    '''
    def insertIntoBST(self, root: Optional[BinaryTree], val: int) -> Optional[BinaryTree]:
        # 原来树里啥都没有，直接创一个节点返回
        if not root: return BinaryTree(val)
        cur = root
        prev = None
        # 寻找插入位置记录前面一个节点，找到位置之后创新节点并加在前一个节点的孩子上， 插入的位置一定在最后一层
        while cur:
            prev = cur
            if val > cur.value:
                cur = cur.right
            elif val < cur.value:
                cur = cur.left

        if prev.value > val: prev.left = BinaryTree(val)
        else: prev.right = BinaryTree(val)
        return root

def deleteNode(root: Optional[BinaryTree], key: int) -> Optional[BinaryTree]:
    parent = None
    cur = root
    while cur:
        if cur.value == key:
            if (not cur.left) and (not cur.right):
                if parent:
                    if cur == parent.left:
                        parent.left = None
                    elif cur == parent.right:
                        parent.right = None
                else:
                    return None

            elif not cur.left:
                parent.right = cur.right
            elif not cur.right:
                parent.left = cur.left
            else:
                leftChild = cur.left
                rightChild = cur.right
                rightLM = cur.right
                while rightLM.left:
                    rightLM = rightLM.left
                rightLM.left = leftChild
                if parent:
                    parent.left = rightChild
                else:
                    root = cur.right
                del cur
            break
        else:
            parent = cur
            if cur.value < key:
                cur = cur.right
            else:
                cur = cur.left
    return root

# TODO: lc 0071

if __name__ == "__main__":
    root = treeLevelOrderDeSerialization("5_3_6_2_4_#_7_#_#_#_#_#_#_")
    root = deleteNode(root, 7)
    printBinaryTree(root)