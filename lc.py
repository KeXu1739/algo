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

def lc_0450():
    '''
    450.删除二叉搜索树中的节点
    没找到点，找到点：叶子，root，只有左，只有右，有左也有右（inordersuccessor）
    :return:
    '''
    def deleteNode(root: Optional[BinaryTree], key: int) -> Optional[BinaryTree]:
        def delete(root):
            if not root: return root
            if not root.right: return root.left
            cur = root.right
            while cur.left:
                cur = cur.left
            cur.left = root.left
            return root
        if not root: return root
        prev = None
        cur = root
        while cur:
            if cur.value == key: break
            prev = cur
            if cur.value > key: cur = cur.left
            else: cur = cur.right

        # 没找到符合key的点
        if not cur: return root

        # 找到了符合key的点，但是prev节点为空，说明找到了头
        if not prev: return delete(root)
        if prev.left and prev.left.value == key:
            prev.left = delete(cur)
        if prev.right and prev.right.value == key:
            prev.right = delete(cur)
        return root


def lc_0669():
    '''
    669. 修剪二叉搜索树
    :return:
    '''
    def trimBST(root: Optional[BinaryTree], low: int, high: int) -> Optional[BinaryTree]:
        if not root: return root
        if root.value < low:
            # 说明root的左树已经完全不符合条件，这时我们trim右边，直接返回trim完之后右边的根节点，隐含删除了当前节点
            return trimBST(root.right, low, high)
        if root.value > high:
            # 说明root的右树已经完全不符合条件，这时我们trim左边，直接返回trim完之后左边的根节点， 隐含删除了当前节点
            return trimBST(root.left, low, high)

        # 到这里为止，当前节点value在low和high之间，则当前节点保留，分别递归trim左右树并将根节点传给当前树的左右孩子
        root.left = trimBST(root.left, low, high)
        root.right = trimBST(root.right, low, high)
        return root


def lc_0108():
    '''
    108.将有序数组转换为二叉搜索树
    :return:
    '''
    def sortedArrayToBST(nums: List[int]) -> Optional[BinaryTree]:
        def arr2BST(nums, l, r):
            if l > r:
                return None
            elif l == r:
                val = nums[l]
                return BinaryTree(val)
            else:
                m = l + ((r-l) >> 1)
                val = nums[m]
                root = BinaryTree(val)
                root.left = arr2BST(nums, l, m-1)
                root.right = arr2BST(nums, m+1, r)
                return root
        return arr2BST(nums, 0, len(nums)-1)

    def sortedArrayToBSTClean(self, nums: List[int]) -> Optional[BinaryTree]:
        def arr2BST(nums, l, r):
            if l > r: return None

            m = l + ((r-l) >> 1)
            val = nums[m]
            root = BinaryTree(val)
            root.left = arr2BST(nums, l, m-1)
            root.right = arr2BST(nums, m+1, r)
            return root
        return arr2BST(nums, 0, len(nums)-1)

def lc_0538():
    '''
    538.把二叉搜索树转换为累加树:二叉树双指针，中序遍历反过来就是右左中，符合累加的顺序，然后用全局变量记录前一个节点和累加和
    :return:
    '''
    def convertBST(self, root: Optional[BinaryTree]) -> Optional[BinaryTree]:
        pre = None
        cumsum = 0
        def traverse(root):
            nonlocal pre, cumsum
            if not root: return
            traverse(root.right)
            if pre is None: cumsum = root.value
            else: cumsum += root.value
            root.value = cumsum
            pre = root
            traverse(root.left)

        traverse(root)
        return root

def lc_0077():
    def combine(n: int, k: int) -> List[List[int]]:
        res = []
        cur = []
        def bt(n, k, start):
            if len(cur) == k:
                res.append(cur[:])
                return
            nd = k - len(cur)
            end = n-nd+1
            for i in range(start, end+1):
                cur.append(i)
                bt(n, k, i + 1)
                cur.pop()
            return

        bt(n,k,1)
        return res

    n=4
    k=2
    print(combine(n, k))


def lc_0216():
    '''
    216.组合总和III
    :return:
    '''
    res = []
    cur = []
    def combinationSum3(k: int, n: int) -> List[List[int]]:
        res = []
        cur = []
        def bt(k, n, summ, start):
            if k == len(cur) :
                if summ == n:
                    res.append(cur[:])
                return

            # 剪枝1
            if summ > n:
                return
            nd = k - len(cur)
            end = 9 - nd + 1
            # 剪枝2
            for i in range(start, end + 1):
                cur.append(i)
                bt(k, n, summ + i, i + 1)
                cur.pop()

            return
        bt(k, n, 0, 1)
        return res


def lc_0017():
    def letterCombinations(digits: str) -> List[str]:
        mp ={
            "2":['a', 'b', 'c'],
            "3":['d', 'e', 'f'],
            "4":['g', 'h', 'i'],
            "5":['j', 'k', 'l'],
            "6":['m', 'n', 'o'],
            "7":['p', 'q', 'r', 's'],
            "8":['t', 'u', 'v'],
            "9":['w', 'x', 'y', 'z'],
        }

        res = []

        def comb(digits, cur, di):
            if len(cur) == len(digits):
                res.append(cur)
                return

            for i in range(di, len(digits)):
                for l in mp[digits[i]]:
                    comb(digits, cur + l, i + 1)

            return
        if len(digits) == 0: return res
        comb(digits, "", 0)
        return res

def lc_0039():
    '''
    39. 组合总和: 给定一个无重复元素的数组 candidates 和一个目标数 target ，找出 candidates 中所有可以使数字和为 target 的组合。
    可重复使用元素
    :return:
    '''
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        res = []
        cur = [] # append copy
        def bt(candidates, target, idx):
            if target < 0: return
            if target == 0:
                res.append(cur[:])
                return
            for i in range(idx,len(candidates)):
                cur.append(candidates[i])
                target -= candidates[i]
                bt(candidates, target, i) # 这里留在原始i位置则意味着可以重复选相同的元素，下一层递归进这里之后还是从i开始
                target += candidates[i]
                cur.pop()
            return

        bt(candidates, target, 0)
        return res

def lc_0040():
    '''
    40.组合总和II, 每一个candidate只能用一次，且candidates中有重复数字，最后的组合不能有重复的组合结果
    :return:
    '''
    def combinationSum2(candidates: List[int], target: int) -> List[List[int]]:
        res = []
        cur = []
        def bt(candidates, target, idx):
            if target < 0: return
            if target == 0:
                res.append(cur[:])
                return
            i = idx
            while i <= len(candidates)-1:
                if i > idx and candidates[i] == candidates[i-1]:
                    i += 1
                    continue
                cur.append(candidates[i])
                bt(candidates, target - candidates[i], i + 1)
                cur.pop()
                i += 1
            return
        candidates.sort()
        bt(candidates, target, 0)
        return res

    print(combinationSum2([1,1,1,1,1,1,1,1,2,2,2,2,2], 5))

def lc_0131():
    '''
    131.分割回文串
    :return:
    '''
    def partition(s: str) -> List[List[str]]:
        def check(s, i, j):
            while i < j:
                if s[i] != s[j]:
                    return False
                i += 1
                j -= 1
            return True

        res=[]
        cur=[]
        def bt(s, idx):
            if idx == len(s):
                res.append(cur[:])
                return

            for i in range(idx, len(s)):
                if check(s, idx, i):
                    cur.append(s[idx:i+1])
                    bt(s, i + 1)
                    cur.pop()
                else: continue
            return
        bt(s, 0)
        return res

def lc_0093():
    '''
    93.复原IP地址:有效的 IP 地址 正好由四个整数（每个整数位于 0 到 255 之间组成，且不能含有前导 0, eg:025，000不合法，但是0合法），
    整数之间用 '.' 分隔
    :return:
    '''
    def restoreIpAddresses( s: str) -> List[str]:
        res = []
        cur = []
        def bt(s, n, idx):
            if n < 0: return
            if n == 0 and idx == len(s):
                res.append(".".join(cur))
                return
            for i in range(idx, len(s)):
                if 0 <= int(s[idx:i+1]) <= 255:
                    if i > idx and s[idx] == "0":
                        continue
                    cur.append(s[idx:i+1])
                    bt(s, n - 1, i + 1)
                    cur.pop()
            return
        bt(s, 4, 0)
        return res

def lc_0078():
    '''
    78.子集: 给定一组不含重复元素的整数数组 nums，返回该数组所有可能的子集（幂集）。说明：解集不能包含重复的子集。
    :return:
    '''
    def subsets(nums: List[int]) -> List[List[int]]:
        res = []
        cur = []
        def bt(nums, idx):
            # 注意理解这里和之前求组合的不同，从递归树的角度考虑，子集是在搜集树的每一个节点，而组合是在搜集固定长度的从根到叶节点的路径
            res.append(cur[:])

            if idx == len(nums):
                # res.append(cur[:])
                return

            for i in range(idx, len(nums)):
                cur.append(nums[i])
                bt(nums, i+1)
                cur.pop()
            return

        bt(nums, 0)
        return res
    print(subsets([1,2,3,4]))

def lc_0090():
    '''
    子集问题，原数组有重复元素的
    :return: 
    '''
    def subsetsWithDup(nums: List[int]) -> List[List[int]]:
        res = []
        cur = []
        used = [False for i in range(len(nums))]
        def bt(nums, idx):
            res.append(cur[:])
            if idx == len(nums):
                return

            for i in range(idx, len(nums)):
                if i != idx and nums[i] == nums[i-1] and used[i-1]:
                    continue
                cur.append(nums[i])
                used[i]=True
                bt(nums, i + 1)
                cur.pop()
                used[i]=False
            return

        nums.sort()
        bt(nums, 0)
        return res
    print(subsetsWithDup([1,1,2,3,4]))

def lc_0491():
    '''
    491.递增子序列:无序有重复数的数组中找递增子序列（不连续），这里注意去重技巧：set在全局定义则代表在每个子序列中都不可以重复
    在bt函数里每次定义代表每个bt函数内不可使用重复的元素，即递归树中每一层去遍历nums可能性的时候不能重复递归用过的元素的case
    本题求自增子序列，所以不能改变原数组顺序
    :return:
    '''
    def findSubsequences(self, nums: List[int]) -> List[List[int]]:
        res = []
        cur = []

        def bt(nums, idx):
            if len(cur) > 1:
                res.append(cur[:])
            if idx == len(nums):
                return
            used = set()
            for i in range(idx,len(nums)):
                if nums[i] in used:
                    continue
                if (len(cur)>0 and nums[i] < cur[-1]):
                    continue
                used.add(nums[i])
                cur.append(nums[i])
                bt(nums, i + 1)
                cur.pop()
            return

        bt(nums, 0)
        return res

def lc_0046():
    '''
    46.全排列: 给定一个 没有重复 数字的序列，返回其所有可能的全排列。used数组放在全局，保证每个permutation不取重复元素，每个排列顺序不同，
    所以可能遍历到之前的元素，不用idx
    :return:
    '''
    def permute(nums: List[int]) -> List[List[int]]:
        res = []
        cur = []
        used = [False for _ in range(len(nums))]
        def bt(nums):
            if len(cur) == len(nums):
                res.append(cur[:])
                return

            for i in range(0, len(nums)):
                if used[i]: continue
                used[i] = True
                cur.append(nums[i])
                bt(nums)
                cur.pop()
                used[i] = False
            return

        bt(nums)
        return res

def lc_0047():
    '''
    47.全排列: 给定一个 有重复数字的序列，返回没有重复的全排列
    :return:
    '''
    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        res = []
        cur = []
        used = [False for _ in range(len(nums))]
        def bt(nums):
            if len(cur) == len(nums):
                res.append(cur[:])
                return

            for i in range(len(nums)):
                # not used[i-1]表明同树层用了第二个重复元素，这里进行剪枝
                if i != 0 and nums[i] == nums[i-1] and not used[i-1]:
                    continue
                if used[i]: continue
                used[i] = True
                cur.append(nums[i])
                bt(nums)
                cur.pop()
                used[i] = False
            return
        nums.sort()
        bt(nums)
        return res

def lc_0455():
    '''
    455.分发饼干
    :return:
    '''
    def findContentChildren(g: List[int], s: List[int]) -> int:
        g.sort(reverse=True)
        s.sort(reverse=True)
        i, j = 0, 0
        ret = 0
        while i < len(g) and j < len(s):
            if s[j] >= g[i]:
                ret += 1
                i += 1
                j += 1
            else:
                i += 1
        return ret


def lc_0332():
    '''
    332.重新安排行程:每条航线都必须走且只走一次，返回字典序最小的航线，python这里要注意因为没有treemap所以targets里面记录终点的数据结构使用
    list，而且需要排序，每次删掉前面的，回溯完append到最后面
    :return:
    '''
    from collections import defaultdict
    def findItinerary(tickets: List[List[str]]) -> List[str]:
        res = ["JFK"]
        targets = defaultdict(list)
        for s, e in tickets: targets[s].append(e)
        for s in targets: targets[s].sort()

        def bt(start):
            if len(res) == len(tickets) + 1: return True
            for _ in targets[start]:
                dest = targets[start].pop(0)
                res.append(dest)
                if bt(dest): return True
                res.pop()
                targets[start].append(dest)

        bt("JFK")
        return res

    print(findItinerary([["JFK","KUL"],["JFK","NRT"],["NRT","JFK"]]))

def lc_0051():
    '''
    51. N-Queens,返回棋盘形状，以行递归，以列遍历
    :return:
    '''
    def solveNQueens(n: int) -> List[List[str]]:
        res = []
        cb = [["." for i in range(n)] for j in range(n)]
        def valid(r, c, cb):
            for i in range(len(cb)):
                if cb[i][c] == "Q":
                    return False

            i, j = r - 1, c - 1
            while i >=0 and j >=0:
                if cb[i][j] == "Q":
                    return False
                i -= 1
                j -= 1

            i, j = r - 1, c + 1
            while i >= 0 and j < n:
                if cb[i][j] == "Q":
                    return False
                i -= 1
                j += 1
            return True

        def bt(r, cb):
            n = cb
            if r == n:
                tmp = []
                for row in cb: tmp.append("".join(row))
                res.append(tmp)
                return
            for c in range(len(cb)):
                if valid(r, c, cb):
                    cb[r][c] = "Q"
                    bt(r + 1, cb)
                    cb[r][c] = "."

        bt(0, cb)
        return res


def lc_0037():
    '''
    37. 解数独
    :return:
    '''
    def solveSudoku(board: List[List[str]]) -> None:
        digits = "123456789"
        def valid(r, c, val, board):
            for i in range(len(board)):
                if board[i][c] == val:
                    return False
            for j in range(len(board)):
                if board[r][j] == val:
                    return False
            uli, ulj = r // 3 * 3, c // 3 * 3
            for i in range(uli, uli + 3):
                for j in range(ulj, ulj + 3):
                    if board[i][j] == val:
                        return False
            return True

        def bt(board):
            for i in range(len(board)):
                for j in range(len(board)):
                    if board[i][j] != ".": continue
                    for k in digits:
                        # 找到了一个可以填数字的地方，遍历尝试所有9个数
                        if valid(i, j, k, board):
                            # 填一个数，判断是不是填完了，没完的话递归填下一个位置的数字，填完了返回True
                            board[i][j] = k
                            if bt(board): return True
                            # board还没填满，并且填这个数这条路走不通，需要回溯然后尝试别的数
                            board[i][j] = "."
                    return False
            # 棋盘满了，所有位置都不是'.'直接返回true
            return True

        return bt(board)

def lc_0376():
    def wiggleMaxLength(nums: List[int]) -> int:
        n = len(nums)
        if n == 0 or n == 1: return n

        res = 1
        prevdiff = 0

        for i in range(1,n):
            curdiff = nums[i] - nums[i-1]
            if (curdiff > 0 and prevdiff <= 0) or (curdiff < 0 and prevdiff >= 0):
                res += 1
                prevdiff = curdiff
        return res

def lc_0053():
    '''
    最大连续子序列和以及对应起止位置的下标
    :return:
    '''
    def maxSubArray(nums: List[int]) -> int:
        cur, mx = 0, float("-inf")
        for num in nums:
            cur += num # 当前位置累加上
            mx = max(mx, cur) # 更新最大值
            cur = max(0, cur) # 如果累加完当前元素的cur小于0了，说明当前这个value会把面的累加和变为负数，那么就需要从下一个元素
            # 开始重新累加 （cur
        return mx

    def maxSubArrayRange(nums: List[int]) -> int:
        cur, mx = 0, float("-inf")
        l, r = 0, 0
        for i, num in enumerate(nums):
            cur += num # 当前位置累加上
            if cur > mx:
                mx = cur # 更新最大值
                r = i
            if cur < 0:
                cur = 0 # 如果累加完当前元素的cur小于0了，说明当前这个value会把面的累加和变为负数，那么就需要从下一个元素开始重新累加
                l = i+1

        if l > r:
            # 如果元素全是负的，前面的逻辑会有问题，这里强行找到最大的负数然后返回index
            l, r = nums.index(mx), nums.index(mx)
        return mx, l, r

    print(maxSubArrayRange([]))

def lc_0122():
    '''
    122.买卖股票的最佳时机II
    :return:
    '''
    def maxProfit(prices: List[int]) -> int:
        if len(prices) == 1: return 0
        diff = []
        su = 0
        for i in range(1, len(prices)):
            if prices[i] > prices[i-1]:
                su += prices[i] - prices[i-1]
        return su

def lc_0055():
    '''
    55. 跳跃游戏： 给定一个非负整数数组，你最初位于数组的第一个位置。数组中的每个元素代表你在该位置可以跳跃的最大长度。
    判断你是否能够到达最后一个位置。
    :return:
    '''
    def canJump(nums: List[int]) -> bool:
        if len(nums) == 1: return True
        rg = 0
        i = 0
        while i <= rg:
            rg = max(rg, i + nums[i]) # 每个元素看能不能让range变大
            if rg >= len(nums)-1: return True
            i += 1
        return False

def lc_0045():
    '''
    45. 跳跃游戏： 给定一个非负整数数组，你最初位于数组的第一个位置。数组中的每个元素代表你在该位置可以跳跃的最大长度。
    判断你最少跳几次可以跳到最后一个位置。
    :return:
    '''
    def jump(nums: List[int]) -> int:
        j = 0
        crm = 0
        nrm = 0
        for i, num in enumerate(nums):
            nrm = max(nrm, i+nums)
            if i == crm and i != len(nums)-1:
                j += 1
                crm = nrm
        return j


def lc_1005():
    '''
    1005.K次取反后最大化的数组和
    :return:
    '''
    def largestSumAfterKNegations(nums: List[int], k: int) -> int:
        # sort()
        # try to flip every negative val
        # if # of neg > k, then thats it, sum and return
        # if # of neg < k and after some flips all nums are positive, then flip only element with smallest abs value
        idx = 0
        cneg = 0
        dist = float("inf")
        nums.sort()
        for i, num in enumerate(nums):
            if num < 0:
                cneg += 1
            if abs(num) < dist:
                dist = abs(num)
                idx = i
        res = 0
        if cneg < k:
            t = k - cneg
            for num in nums:
                if num < 0:
                    res -= num
                else:
                    res += num
            if t % 2 == 1:
                res -= 2 * abs(nums[idx])
        else:
            for num in nums:
                if num < 0 and k > 0:
                    res -= num
                    k -= 1
                else:
                    res += num
        return res

    def largestSumAfterKNegations(nums, k):
        # 空间复杂度高但是时间最优, 如果能改原数组且k远小于len(nums)则这个最优，lc里面k和len(nums)数量级一致则复杂度一致
        import heapq
        hp = [e for e in nums]
        heapq.heapify(hp)
        s = 0
        while k > 0:
            ele = heapq.heappop(hp)
            k -= 1
            heapq.heappush(hp, -1 * ele)
        return sum(hp)


def lc_0134():
    '''
    134. 加油站
    :return:
    '''
    def canCompleteCircuit(gas: List[int], cost: List[int]) -> int:
        if sum(gas) < sum(cost): return -1
        rest = [g - c for g,c in zip(gas, cost)]
        cumSum = 0
        s = 0
        for i, r in enumerate(rest):
            cumSum += r
            if cumSum < 0:
                cumSum = 0
                s = i + 1
        return s


def lc_0135():
    '''
    根据rating分糖果，高rating糖果数高于邻居们
    :return:
    '''
    def candy(ratings: List[int]) -> int:
        res = [1 for r in ratings]
        if len(res) == 1: return sum(res)
        for i in range(1, len(ratings)):
            if ratings[i] > ratings[i-1] and res[i] <= res[i-1]:
                res[i] = res[i-1] + 1
        for i in range(len(ratings)-2, -1, -1):
            if ratings[i] > ratings[i+1] and res[i] <= res[i+1]:
                res[i] = res[i+1] + 1
        return sum(res)
def lc_0806():
    '''
    860.柠檬水找零， 5,10,20
    :return:
    '''
    def lemonadeChange(bills: List[int]) -> bool:
        fv = 0
        tn = 0
        for num in bills:
            if num == 5:
                fv += 1
            elif num == 10:
                fv -= 1
                tn += 1
            elif num == 20:
                if tn >= 1:
                    tn -= 1
                    fv -= 1
                else:
                    fv -= 3
            if fv < 0 or tn < 0: return False
        return True

def lc_0406():
    '''
    406.根据身高重建队列：根据身高和队列中前面有多少个高于自己的人重建队列
    :return:
    '''
    def lcOptimal(people: List[List[int]]) -> List[List[int]]:
        people.sort(key=lambda x:(x[0], -x[1]), reverse=True)
        ls = []
        for ele in people:
            ls.insert(ele[1], ele)
        return ls

    def reconstructQueue(people: List[List[int]]) -> List[List[int]]:
        class Node:
            def __init__(self, val, n=None):
                self.val = val
                self.n = n

        people.sort(key=lambda x: (x[0], -x[1]), reverse=True)

        head = Node(people[0])
        _dummy = Node(None, head)
        for i in range(1, len(people)):
            cur = _dummy
            count = 0

            while (cur.n and count < people[i][1]):
                count += 1
                cur = cur.n
            if not cur.n:
                cur.n = Node(people[i])
            else:
                tmp = Node(people[i], cur.n)
                cur.n = tmp

        res = []
        cur = _dummy.n
        while cur:
            res.append(cur.val)
            cur = cur.n
        return res
def lc_0452():
    '''
    452. 用最少数量的箭引爆气球
    :return:
    '''
    def lcOptimal(self, points: List[List[int]]) -> int:
        points.sort(key=lambda x: x[1])
        lastIdx = float("-inf")
        res = 0
        for l, r in points:
            if l > lastIdx:
                res += 1
                lastIdx = r
        return res

    def findMinArrowShots(points: List[List[int]]) -> int:
        points.sort()
        curInterval = [float("-inf"), float("inf")]
        res = 1
        for ele in points:
            # 取交集，如果没交集就更新curInterval并增加结果集，相交的气球集合用一只箭可以射爆
            curInterval[0] = max(curInterval[0], ele[0])
            curInterval[1] = min(curInterval[1],ele[1])
            if curInterval[0] > curInterval[1]:
                res += 1
                curInterval = ele
        return res

def lc_0435():
    '''
    435. 无重叠区间
    :return: 返回需要删掉几个区间
    '''
    def eraseOverlapIntervals(intervals: List[List[int]]) -> int:
        if len(intervals) == 1: return 0
        intervals.sort(key=lambda x: x[1])
        res = 0
        cur = intervals[0]
        for i in range(1, len(intervals)):
            if intervals[i][0] < cur[1]:
                # 重叠就res+1，这时cur不发生变化，下一轮还可能和下一个区间继续重叠
                res += 1
            else:
                # 不重叠，就把cur移到当前的interval，这样还有跟下面interval重叠的可能
                cur = intervals[i]
        return res

def lc_0763():
    '''
    763.划分字母区间
    统计每一个字符最后出现的位置
    从头遍历字符，并更新字符的最远出现下标，如果找到字符最远出现位置下标和当前下标相等了，则找到了分割点
    :return:
    '''
    def partitionLabels(s: str) -> List[int]:
        b = {}
        for i,e in enumerate(s):
            b[e] = i

        res = []
        lg = 0
        cnt = 0
        for i in range(len(s)):
            cnt += 1
            lg = max(lg, b[s[i]])
            if i == lg:
                res.append(cnt)
                cnt = 0

        return res

def lc_0056():
    '''
    56. 合并区间
    :return:
    '''
    def merge(intervals: List[List[int]]) -> List[List[int]]:
        intervals.sort()
        res = []
        cur = intervals[0]
        for i in range(1, len(intervals)):
            if cur[1] >= intervals[i][0]:
                # 如果集合相交，取并集，否则更新cur集合并把之前的集合放入结果集
                cur[0] = min(cur[0], intervals[i][0])
                cur[1] = max(cur[1], intervals[i][1])
            else:
                res.append(cur[:])
                cur = intervals[i]
        res.append(cur)
        return res

def lc_0738():
    '''
    738.单调递增的数字
    :return:
    '''
    def lcOptimal(n: int) -> int:
        cL = list(str(n))
        for i in range(len(cL)-1, 0, -1):
            if cL[i] < cL[i-1]:
                cL[i:] = ['9']*(len(cL)-i)
                cL[i-1] = str(int(cL[i-1]-1))
        return int("".join(cL))

    def monotoneIncreasingDigits(n: int) -> int:
        charList = [int(i) for i in str(n)] # 常数空间
        start = -1
        for i in range(len(charList)-1, 0, -1):
            if charList[i-1] > charList[i]:
                start = i
                charList[i] = 9
                charList[i-1] -= 1
        if start != -1:
            for i in range(start, len(charList)):
                charList[i] = 9
        return int("".join([str(i) for i in charList]))

def lc_0714():
    '''
    714. 买卖股票的最佳时机含手续费,
    Greedy:最重要的地方是连续递增的股价时如何只计算一次交易费用，这个处理非常的巧妙
    :return:
    '''
    def maxProfitGreedy(prices: List[int], fee: int) -> int:
        p = 0
        mp = prices[0]
        for i in range(1, len(prices)):
            if prices[i] < mp:
                mp = prices[i]
            elif prices[i] - mp - fee < 0: continue
            elif prices[i] - mp - fee >= 0:
                p += prices[i] - mp - fee
                mp = prices[i] - fee # 这里模拟一下就可以看出来这样设置之后
        return p
    maxProfitGreedy([1,5,10,9,13], 3)

def lc_0968():
    '''
    968.监控二叉树:二叉树节点上摆放监控相机的最优策略
    :return:
    '''
    def minCameraCover(root: Optional[BinaryTree]) -> int:
        # 3 status: 0. no cover, 1. has camera 2. has cover
        # for empty node: needs to be status 2 to prevent leaf-setting-camera
        # 贪心策略：在叶子节点安排相机效率低，因为一个相机只能覆盖这个叶子节点和他的父节点，而一个父节点可以覆盖两个叶子（或两个孩子）以及
        # 自己的父节点，所以我们在叶子节点上不安排相机
        res = 0
        def rc(root):
            nonlocal res
            # 因为叶子节点不能安排相机，那么空节点要设计成被覆盖状态，这样可以保证叶子节点不放相机同时叶子节点也不被覆盖
            if not root: return 2
            l = rc(root.left)
            r = rc(root.right)
            # 左和右孩子同时为被覆盖状态，则当前节点在本层递归是不被覆盖状态
            if l == 2 and r == 2: return 0
            # 左或者右边有一个没覆盖，那么当前节点需要有一个照相机
            # left == 0 && right == 0 左右节点无覆盖
            # left == 1 && right == 0 左节点有摄像头，右节点无覆盖
            # left == 0 && right == 1 左节点有无覆盖，右节点摄像头
            # left == 0 && right == 2 左节点无覆盖，右节点覆盖
            # left == 2 && right == 0 左节点覆盖，右节点无覆盖
            elif l == 0 or r == 0:

                res += 1
                return 1
            # 剩下的状态是：左右孩子一个覆盖一个有相机，或者同时有相机（不可能同时有覆盖因为第一个case排除掉了），那么
            # 当前一定是被覆盖状态
            # left == 1 && right == 2 左节点有摄像头，右节点有覆盖
            # left == 2 && right == 1 左节点有覆盖，右节点有摄像头
            # left == 1 && right == 1 左右节点都有摄像头
            else: return 2

        status = rc(root)
        if status == 0: res += 1
        return res

def lc_0509():
    '''
    509. 斐波那契数
    :return:
    '''
    def fibdp(n):
        if n == 0: return 0
        if n == 1: return 1
        dp = [None for i in range(n+1)]
        # dp[i]: n=i时的fib数
        dp[0] = 0
        dp[1] = 1
        for i in range(2, len(dp)):
            dp[i] = dp[i-1] + dp[i-2]

        return dp[-1]


    def fib(n: int) -> int:
        if n == 0: return 0
        if n == 1: return 1
        f,s = 0,1
        for i in range(n-1):
            f,s = s, f+s
        return s
    print(fibdp(3))

def lc_0070():
    '''
    70. 爬楼梯的方法数
    :return:
    '''
    def climbStairsDP(n: int) -> int:
        if n == 1: return 1
        if n == 2: return 2
        dp = [None for i in range(n+1)]
        # 爬到第i层楼梯，有dp[i]种方法
        dp[1] = 1
        dp[2] = 2
        for i in range(3, len(dp)):
            dp[i] = dp[i-1] + dp[i-2]
        return dp[-1]

    def climbStairs(n):
        if n == 1: return 1
        if n == 2: return 2
        f,s = 1, 2
        for i in range(3, n+1):
            f,s = s, f+s
        return s

def lc_0746():
    '''
    746. 使用最小花费爬楼梯:每个位置有个cost，代表从这个台阶爬出去需要的cost
    :return:
    '''
    def minCostClimbingStairs(self, cost: List[int]) -> int:
        dp = [0 for i in range(len(cost)+1)]
        # dp[i]: 到达cost[i]位置需要的cost，这里多申请一个位置，最后一个位置代表cost最后一个元素的下一个位置即为顶部
        # 第一步从dp[0]或dp[1]开始，所以没有cost，从这两个位置跳出去到下一个位置才开始产生cost
        for i in range(2, len(cost)+1):
            dp[i] = min(dp[i - 2] + cost[i - 2], dp[i - 1] + cost[i - 1])
        return dp[-1]
    def minCostClimbingStairsOptimal(self, cost: List[int]) -> int:
        # 带空间优化的版本，按照之前的定义，每一个位置只依赖其前面一个和两个位置，所以不用申请数组，只需要申请两个变量存数就可以了
        f,s = 0, 0
        for i in range(2, len(cost) + 1):
            f,s = s,min(f+cost[i-2], s+cost[i-1])
        return s

def lc_0062():
    '''
    62.不同路径:从左上角出发到右下角
    :return:
    '''
    def uniquePaths(m: int, n: int) -> int:
        # dp[i][j]:从左上角到达坐标(i,j)位置有几种方法
        dp = [[None for _ in range(n)] for __ in range(m)]
        dp[0][0] = 1
        for j in range(1, n): dp[0][j] = 1
        for i in range(1, m): dp[i][0] = 1
        for i in range(1, m):
            for j in range(1, n):
                dp[i][j] = dp[i-1][j] + dp[i][j-1]
        return dp[-1][-1]

    def uniquePathsOptimal(m: int, n: int) -> int:
        # 当前位置只依赖自己左方和上方的位置，空间优化，用更短的那个维度维持dp数组
        dp = [1 for i in range(m)] if m < n else [1 for i in range(n)]
        if m < n:
            for j in range(1, n):
                for i in range(1, m):
                    dp[i] = dp[i] + dp[i-1]
        else:
            for i in range(1, m):
                for j in range(1, n):
                    dp[j] = dp[j] + dp[j - 1]
        return dp[-1]

def lc_0063():
    '''
    63. 不同路径 II:从左上角出发到右下角带障碍物
    :return:
    '''
    def uniquePathsWithObstacles(obstacleGrid: List[List[int]]) -> int:
        m,n = len(obstacleGrid), len(obstacleGrid[0])
        dp = [[None for _ in range(n)] for __ in range(m)]
        val = 1
        for j in range(n):
            if obstacleGrid[0][j]: val = 0
            dp[0][j] = val
        val = 1
        for i in range(m):
            if obstacleGrid[i][0]: val = 0
            dp[i][0] = val
        for i in range(1, m):
            for j in range(1, n):
                if obstacleGrid[i][j]:
                    dp[i][j] = 0
                else:
                    dp[i][j] = dp[i-1][j] + dp[i][j-1]
        return dp[-1][-1]

if __name__ == "__main__":
    lc_0509()
    # TODO: lc 0071