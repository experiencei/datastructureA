#  Valid Palindrome

# Question -->  A phrase is a palindrome if, after converting all uppercase letters into lowercase letters and removing all non-alphanumeric characters, it reads the same forward and backward. 
# Alphanumeric characters include letters and numbers.
# Given a string s, return true if it is a palindrome, or false otherwise.

# Example 1:

# Input: s = "A man, a plan, a canal: Panama"
# Output: true
# Explanation: "amanaplanacanalpanama" is a palindrome.

# Solution --> 
class Solution:
    def isPalindrome(self, s: str) -> bool:
        l, r = 0, len(s) - 1
        while l < r:
            while l < r and not self.alphanum(s[l]):
                l += 1
            while l < r and not self.alphanum(s[r]):
                r -= 1
            if s[l].lower() != s[r].lower():
                return False
            l += 1
            r -= 1
        return True

    # Could write own alpha-numeric function
    def alphanum(self, c):
        return (
            ord("A") <= ord(c) <= ord("Z")
            or ord("a") <= ord(c) <= ord("z")
            or ord("0") <= ord(c) <= ord("9")
        )

# Valid Palindrome II

# Question --> 
#     Given a string s, return true if the s can be palindrome after deleting at most one character from it.

# Example 2:

# Input: s = "abca"
# Output: true
# Explanation: You could delete the character 'c'.

class Solution:
    def validPalindrome(self, s: str) -> bool:
        
        # if empty return True
        if not s:
            return True
        
        start = 0
        end = len(s)-1
        
        #starts scanning from both ends inwards break at unmatched characters
        while start <= end and s[start]==s[end]:
            start += 1
            end -= 1
        
        #if no unmatched characters found
        if end <= start:
            return True
        
        # function to check if string is palindrome
        def isPalindrome(start,end):
            while start <= end:
                if s[start] != s[end]:
                    return False
                start += 1
                end   -=1
            return True
        
        #deleting either characters can result in palindrome hence checking both
        if isPalindrome(start+1,end) or isPalindrome(start,end-1):
            return True
        
        return False 


#  Minimum Difference Between Highest and Lowest of K Scores


# Question --> You are given a 0-indexed integer array nums, where nums[i] represents the score of the ith student.
#  You are also given an integer k.
# Pick the scores of any k students from the array so that the difference between the highest and the lowest of the k scores is minimized.

# Return the minimum possible difference.

 
# Input: nums = [9,4,1,7], k = 2
# Output: 2




class Solution:
    def minimumDifference(self, nums: list[int], k: int) -> int:
        # [1,4,7,9] we have to sort it to get the optimum difference between the value
        nums.sort()
        # to get size k of sliding window
        l, r = 0, k - 1
        res = float("inf")
        
        while r < len(nums):
            res = min(res, nums[r] - nums[l])
            l, r = l + 1, r + 1
        return res



# 344. Reverse String


# Question -->
# Write a function that reverses a string. The input string is given as an array of characters s.
# You must do this by modifying the input array in-place with O(1) extra memory.

 

# Example 1:

# Input: s = ["h","e","l","l","o"]
# Output: ["o","l","l","e","h"]



class Solution:
    def reverseString(self, s: list[str]) -> None:
        """
        Do not return anything, modify s in-place instead.
        """
        l = 0
        r = len(s) - 1
        while l < r:
            s[l],s[r] = s[r],s[l]
            l += 1
            r -= 1

# Merge Sorted Array
# Question --> You are given two integer arrays nums1 and nums2, sorted in non-decreasing order, and two integers m and n, 
# representing the number of elements in nums1 and nums2 respectively.

# Merge nums1 and nums2 into a single array sorted in non-decreasing order.

# The final sorted array should not be returned by the function, but instead be stored inside the array nums1. 
# To accommodate this, nums1 has a length of m + n, where the first m elements denote the elements that should be merged, and the last n elements are set to 0 and should be ignored.
#  nums2 has a length of n.


# Example 1:

# Input: nums1 = [1,2,3,0,0,0], m = 3, nums2 = [2,5,6], n = 3
# Output: [1,2,2,3,5,6]
# Explanation: The arrays we are merging are [1,2,3] and [2,5,6].
# The result of the merge is [1,2,2,3,5,6] with the underlined elements coming from nums1.

# Solution --> it will be convenient to merge from back since it already sorted

class Solution:
    def merge(self, nums1: list[int], m: int, nums2: list[int], n: int) -> None:
        """
        Do not return anything, modify nums1 in-place instead.
        """
        #merge in reverse order
        while m > 0 and n > 0:
            #replace last index with nums1 and decrement the pointer
            #the minus 1 because of 0 index
            if nums1[m-1] >= nums2[n-1]:
                nums1[m+n-1] = nums1[m-1]
                m -= 1
            #replace last index with nums2 and decrement the pointer
            else:
                nums1[m+n-1] = nums2[n-1]
                n -= 1
        # fill nums1 with leftover nums2 elements
        if n > 0:
            nums1[:n] = nums2[:n]



# Move Zeroes

# Question -->  Given an integer array nums, move all 0's to the end of it while maintaining the relative order of the non-zero elements.

# Note that you must do this in-place without making a copy of the array.

# Example 1:

# Input: nums = [0,1,0,3,12]
# Output: [1,3,12,0,0]
# Example 2:

# Input: nums = [0]
# Output: [0]

# Solution : the question said we should move all 0's to the end , wouldn't it be great to move the 
# non zero's to the the left hand side ( by rewording the question )

class Solution:
    def moveZeroes(self, nums: list[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        slow = 0
        # the fast pointer will be moving through the entire array
        for fast in range(len(nums)):
            
            if nums[fast] != 0 and nums[slow] == 0:
                nums[slow], nums[fast] = nums[fast], nums[slow]

            if nums[slow] != 0:
              # we want to make sure we increment the slow pointer
                slow += 1



# Remove Duplicates from Sorted Array

# Question --> Given an integer array nums sorted in non-decreasing order, remove the duplicates in-place such that each unique element appears only once.
#  The relative order of the elements should be kept the same. Then return the number of unique elements in nums.

# Consider the number of unique elements of nums to be k, to get accepted, you need to do the following things:

# Change the array nums such that the first k elements of nums contain the unique elements in the order they were present in nums initially.
# The remaining elements of nums are not important as well as the size of nums.
# Return k.
# Custom Judge:

# The judge will test your solution with the following code:

# int[] nums = [...]; // Input array
# int[] expectedNums = [...]; // The expected answer with correct length

# int k = removeDuplicates(nums); // Calls your implementation

# assert k == expectedNums.length;
# for (int i = 0; i < k; i++) {
#     assert nums[i] == expectedNums[i];
# }
# If all assertions pass, then your solution will be accepted.

 

# Example 1:

# Input: nums = [1,1,2]
# Output: 2, nums = [1,2,_]
# Explanation: Your function should return k = 2, with the first two elements of nums being 1 and 2 respectively.
# It does not matter what you leave beyond the returned k (hence they are underscores).

# Solution : we want to use 2 pointer for comparison , Left pointer and Right pointer will be starting at index 1
# bcos zero index is unique.

class Solution:
    def removeDuplicates(self, nums: list[int]) -> int:
        # left pointer starting at index 1 because the first value is still unique
        L = 1
        
        for R in range(1, len(nums)):
            # we want to replace left pointer only when the Right pointer has change in value
            if nums[R] != nums[R - 1]:
                nums[L] = nums[R]
                L += 1
        return L


# Remove Duplicates from Sorted Array II

# Question --> Given an integer array nums sorted in non-decreasing order, remove some duplicates in-place such that each unique element appears at most twice.
#  The relative order of the elements should be kept the same.

# Since it is impossible to change the length of the array in some languages, you must instead have the result be placed in the first part of the array nums. 
# More formally, if there are k elements after removing the duplicates, then the first k elements of nums should hold the final result. It does not matter what you leave beyond the first k elements.

# Return k after placing the final result in the first k slots of nums.

# Do not allocate extra space for another array. You must do this by modifying the input array in-place with O(1) extra memory.

# Custom Judge:

# The judge will test your solution with the following code:

# int[] nums = [...]; // Input array
# int[] expectedNums = [...]; // The expected answer with correct length

# int k = removeDuplicates(nums); // Calls your implementation

# assert k == expectedNums.length;
# for (int i = 0; i < k; i++) {
#     assert nums[i] == expectedNums[i];
# }
# If all assertions pass, then your solution will be accepted.

 

# Example 1:

# Input: nums = [1,1,1,2,2,3]
# Output: 5, nums = [1,1,2,2,3,_]
# Explanation: Your function should return k = 5, with the first five elements of nums being 1, 1, 2, 2 and 3 respectively.
# It does not matter what you leave beyond the returned k (hence they are underscores).

# Solution --> 
class Solution:
    def removeDuplicates(self, nums: list[int]) -> int:
        l, r = 0, 0

        while r < len(nums):
            # keeping track of the count r
            count = 1
            while r + 1 < len(nums) and nums[r] == nums[r + 1]:
                r += 1
                count += 1
            
            for i in range(min(2, count)):
                nums[l] = nums[r]
                l += 1
            r += 1
        return l



# 167. Two Sum II - Input Array Is Sorted
# Medium

# 10712

# 1294

# Add to List

# Share
# Given a 1-indexed array of integers numbers that is already sorted in non-decreasing order, find two numbers such that they add up to a specific target number. Let these two numbers be numbers[index1] and numbers[index2] where 1 <= index1 < index2 < numbers.length.

# Return the indices of the two numbers, index1 and index2, added by one as an integer array [index1, index2] of length 2.

# The tests are generated such that there is exactly one solution. You may not use the same element twice.

# Your solution must use only constant extra space.

 

# Example 1:

# Input: numbers = [2,7,11,15], target = 9
# Output: [1,2]
# Explanation: The sum of 2 and 7 is 9. Therefore, index1 = 1, index2 = 2. We return [1, 2].



# Solution : --> [ 1 , 3 , 4 , 5 , 7 , 11]
#                  |                    |
# since we know the array is already sorted 
# we will be using 2 pointer 1 to the left and 1 to the right
# 1 + 11 > 9 we will reduce the right pointer 
# 1 + 7 < 8 we will increasing the left pointer since it lesser than the value


class Solution:
    def twoSum(self, numbers: list[int], target: int) -> list[int]:
        l, r = 0, len(numbers) - 1

        while l < r:
            curSum = numbers[l] + numbers[r]

            if curSum > target:
                r -= 1
            elif curSum < target:
                l += 1
            else:
                return [l + 1, r + 1]


# 3Sum

# Question --> Given an integer array nums, return all the triplets [nums[i], nums[j], nums[k]] such that i != j, i != k, and j != k, and nums[i] + nums[j] + nums[k] == 0.
# Notice that the solution set must not contain duplicate triplets.