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




# class Solution:
    def minimumDifference(self, nums: List[int], k: int) -> int:
        nums.sort()
        l, r = 0, k - 1
        res = float("inf")
        
        while r < len(nums):
            res = min(res, nums[r] - nums[l])
            l, r = l + 1, r + 1
        return res



# 344. Reverse String
# Easy

# 7852

# 1122

# Add to List

# Share
# Write a function that reverses a string. The input string is given as an array of characters s.
# 6
# You must do this by modifying the input array in-place with O(1) extra memory.

 

# Example 1:

# Input: s = ["h","e","l","l","o"]
# Output: ["o","l","l","e","h"]



# class Solution:
#     def reverseString(self, s: List[str]) -> None:
#         """
#         Do not return anything, modify s in-place instead.
#         """
#         l = 0
#         r = len(s) - 1
#         while l < r:
#             s[l],s[r] = s[r],s[l]
#             l += 1
#             r -= 1