import copy
from typing import Optional

from algoHelper import *
import numpy as np

def lc_0001():
    # 2和
    def sol(arr, t):
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
        return r+1 # 这里刚开始是标准的二分查找-1，之后可以通过找规律发现返回r + 1

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

if __name__ == "__main__":
    lc_0202()
