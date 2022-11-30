from algoHelper import *


def lc_0001():
    # 2和
    def sol(arr, t):
        dic = {}
        for i, ele in enumerate(arr):
            if t - ele in dic:
                return [ele, t-ele]
            dic[ele] = True
        return

def lc_0026():
    # 数组中去掉重复数
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
        return k
    in_ = [1,2,2,2,3,3,4,4,4,5,6,22,23,23,23]
    print(removeDuplicates2(in_))
    print(in_)

def lc_0027():
    def removeElement(nums: List[int], val: int) -> int:
        # 快指针只要不是target value就填到慢指针上，然后两个指针都往右走
        k = 0
        for i in range(len(nums)):
            if nums[i] != val:
                nums[k] = nums[i]
                k += 1
            i += 1
        return k

    nums = [1,4,4,4,4,4,2,2,3]
    t = 2
    print(removeElement(nums, t))
    print(nums)

def lc_0066():
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

    digits = [9,8,7,6,5,4,3,2,1,0]
    print(plusOne(digits))

if __name__ == "__main__":
    lc_0066()
