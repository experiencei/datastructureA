# contain duplicate
# QUESTION -->  Given an integer array nums, return true if any value appears at least twice in the array,
# and return false if every element is distinct.

# Example 1:

# Input: nums = [1,2,3,1]
# Output: true

# solution 1 : comparing of single integer in an array with other interger i.e 1 == 2 , 1==3 , 1==1
#  which is TIME complexity of 0(N2)
#  solution 2 : sorting of the array from [1 , 2 , 3 , 1] --> [1 , 1, 2 , 3]
# which is TIME complexity of 0(nLOGn)
# solution 3 : sacrificing of Space for Time with HASHSET which store a unique integer
# which is TIME complexity of 0(n) and SPACE complexity of 0(n)
class Solution:
    def containsDuplicate(self, nums: list[int]) -> bool:
        hashSet = set()

        for n in nums:
            if n in hashSet:
                return True
            hashSet.add(n)
        return False

# valid anagram
# QUESTION -->  Given two strings s and t, return true if t is an anagram of s, and false otherwise.

# Solution : counting each value and putting it an hashMap and compare if it true or false:
#  {a : 3 , g : 1 , r : 1 , m : 1} and compare with other hashmap and validate the same length as well 
#  cause there's no catch if the LENGTH isn't the same

class solution:
        def isAnagram(self , s : str , t : str) -> bool:
                if len(s) != len(t):
                        return False
                countS , countT = {} , {}

                for i in range(len(s)):
                        countS[s[i]] = 1 + countS.get(s[i] , 0)
                        countS[t[i]] = 1 + countT.get(t[i] , 0)
                return countS == countT

class Solution:
    def isAnagram(self, s: str, t: str) -> bool:
        if len(s) != len(t):
            return False

        countS, countT = {}, {}

        for i in range(len(s)):
            countS[s[i]] = 1 + countS.get(s[i], 0)
            countT[t[i]] = 1 + countT.get(t[i], 0)
        return countS == countT

# Concatenation of an array
#QUESTION -->  Given an integer array nums of length n, you want to create an array ans of length 2n where ans[i] == nums[i] and ans[i + n] == nums[i] for 0 <= i < n (0-indexed).
# Specifically, ans is the concatenation of two nums arrays.

# Solution : is concatenating array is by tutning [ 1 , 3 , 5] to [1 , 3 , 5 , 1 , 3 , 5].
# the first approach is creating an empty [] and concatenate in n time which is two times and 
# the Time complexity is 0(n)
class Solution:
    def concatenateArray(self, nums: list[int])-> list[int]:
        ans = []
        for i in range(2):
            for n in nums:
                ans.append(n)
        return ans


# Replace Elements with Greatest Element on Right Side
# QUESTION --> Given an array arr, replace every element in that array with the greatest element among the elements to its right, 
# and replace the last element with -1.

# After doing so, return the array.

# Example 1:

# Input: arr = [17,18,5,4,6,1]
# Output: [18,6,6,6,1,-1]

# Solution : [17,18,5,4,6,1] to get the new maximum value in array 
# e.g for new[0] = Max(arr[1: 5]) meaning we're getting for five different value
        # new[1] = Max(arr[2: 5])
        # new[2] = Max(arr[3: 5]) instead of repititive work what if we shorten it to a comparison
        # TIME complexity of 0(n2)
        # Efficient
# now in reverse order getting the maximal values from back first and store it before comparison
#    which is now new[0] = Max(arr[i] , prev[i])   meaning we're getting for five different value
class Solution:
    def replaceElement(self, arr:list[int]) -> list[int]:
        #initial Max = -1
        # reverse iteration
        # new max = max(oldmax , arr[i])
        rightMax = -1
        for i in range(len(arr) -1 , -1  , -1):
            newMax = max(arr[i], rightMax)
            arr[i] = rightMax
            rightMax = newMax
        return arr

# is Subsequence
# Question Given two strings s and t, return true if s is a subsequence of t, or false otherwise.

# A subsequence of a string is a new string that is formed from the original string by deleting some (can be none) of the characters without disturbing the relative positions of the remaining characters. (i.e., "ace" is a subsequence of "abcde" while "aec" is not).

 

# Example 1:

# Input: s = "abc", t = "ahbgdc"
# Output: true
# Given two string S and T s=ace and t= amcnme , as long as we can 
# find string s in correct order in t string.
#  Solution : --> we have 2 pointer , 1 on string S and other on string T we keep moving the pointer on T as 
# long as we find string S
    # sample -- > s="muf" and t="nfmliutgfy"
    #                 |1           |2

class Solution:
    def isSubsequence(self ,  s : str , t : str) -> bool:
        m , n = 0 , 0
        while m < len(s) and n < len(t):
            if s[m] == t[n]:
                m += 1
                n += 1
            else :
                n += 1
        return True if m == len(s) else False

class Solution:
    def isSubsequence(self , s : str , t : str) -> bool:
        i , r = 0 , 0
        while i < len(s) and r < len(t):
            if s[i] == t[r]:
                i += 1
                j += 1
            else :
                j += 1
        return True if i == len(s) else False

#  length of last word
# Question --> Given a string s consisting of words and spaces, return the length of the last word in the string.
# A word is a maximal substring consisting of non-space characters only.

# we want to return the lenth of the last word after space (" ")
# sample s = " what is my name ? is ibrahim "

class solution:
    def lengthOflastword(self , s : str) -> int:
        n , length = len(s) - 1 , 0
        while s[n] == " ":
            n -= 1
        while s[n] != " " and n >= 0:
            length += 1
            n -= 1
        return length


# 2 sum array that sum to a target
# Question --> Given an array of integers nums and an integer target,
#  return indices of the two numbers such that they add up to target.
# You may assume that each input would have exactly one solution, and you may not use the same element twice.
# Solution: [ 2 , 5 , 6 , 4 , 1] target = 9
    # we can loop through the array by taking one value and adding 
    # the next value and adjusting the pointer which is basically 0(n2)
# Efficient --> we have a hashmap {value : index} take the 
        # different btw the target and the current value and look for the value in Hashmap
class Solution:
     def twoSum(self, nums: list[int], target: int) -> list[int]:
        prevMap = {}  #val : index
        for i , n in enumerate(nums):
            val = target - n
            if val in prevMap:
                return [prevMap[val] , i]
            prevMap[n] = 1
        return

            
