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
        prevMap = {} # value : index
        for i , n in enumerate(nums):
            diff = target - n
            if diff in prevMap:
                return [prevMap[diff] , i]
            prevMap[n] = i
        return

# Longest Common Prefix
#Question -->  Write a function to find the longest common prefix string amongst an array of strings.
# If there is no common prefix, return an empty string "".

# SOluiton : to check the LCP we will take the first str in array and loop through it and 
# take the first string and compare simultaneouly

class Solution:
    def longestcommonPrefix(self , strs: list[str]) -> str:
        res =""
        for i in range(len(strs[0])):
            for str in strs:
                if i == len(str) and str[i] != strs[0][i]:
                 return res
            res += strs[0][i]
        return res

# Groups Anagram
# QUESTION --> Given an array of strings strs, group the anagrams together. 
# You can return the answer in any order.


# Input: strs = ["eat","tea","tan","ate","nat","bat"]
# Output: [["bat"],["nat","tan"],["ate","eat","tea"]]

# Solution : we can sort evry string in the array and group the match together in Result
#   --> in efficient as Sorting takes 0(nlogn) * m
# EFficient Solution would be getting each str and counting the occurence of it.
# {["eat" , "tea" , "ate"] all has 1 - a , 1 - t and 1 - e}


class Solution:
    def groupAnagrams(self, strs: list[str]) -> list[list[str]]:
        ans = collections.defaultdict(list)

        for s in strs:
            #initializing coumt with 0
            count = [0] * 26
            for c in s:
                # assigning every characters to acsii value of it and the increment to count them
                count[ord(c) - ord("a")] += 1
            ans[tuple(count)].append(s)
        return ans.values()

# Pascal Triangle
# Question --> Given an integer numRows, return the first numRows of Pascal's triangle.
# In Pascal's triangle, each number is the sum of the two numbers directly above it as

# Solution  : we want to initaial the 1st one since we knew it going to be 1.


class Solution:
    def generateTriangle(self, numRows : int) -> list[list[int]]:
        res = [[1]]

#   the first loop is length of pascal - 1 CAUSE we already did the first one
        for i in range(numRows - 1):
            temp = [0] + res[-1] + [0]
            row = []
            # loop for building the next row , which is length of previous row + 1
            for j in range(len(res[-1]) + 1):
                row.append(temp[j] + temp[j + 1])
            res.append(row)
        return res

# Remove Element
#QUESTION -->  Given an integer array nums and an integer val, remove all occurrences of val in nums in-place. 
# The order of the elements may be changed. 
# Then return the number of elements in nums which are not equal to val.

# Solution : given nums [ 1 , 3 , 5 , 2, 2 , 3] val = 3 , 
# we want to return count of nums that are not val in array num
# we want to modify the array in place .

class Solution:
    def removeElement(self, nums : list[int] , val : int) -> int :
        k = 0
        for i in range(len(nums)):
            if nums[i] != val:
                nums[k] = nums[i]
                k += 1
        return k

# Unique Email Addresses
# QUESTION --> Every valid email consists of a local name and a domain name, separated by the '@' sign. Besides lowercase letters, the email may contain one or more '.' or '+'.

# For example, in "alice@leetcode.com", "alice" is the local name, and "leetcode.com" is the domain name.
# If you add periods '.' between some characters in the local name part of an email address, mail sent there will be forwarded to the same address without dots in the local name. 
# Note that this rule does not apply to domain names.

# For example, "alice.z@leetcode.com" and "alicez@leetcode.com" forward to the same email address.
# If you add a plus '+' in the local name, everything after the first plus sign will be ignored. 
# This allows certain emails to be filtered. Note that this rule does not apply to domain names.

class Solution:
    def uniqueEmail(self , emails : list[str] ) -> int:
        unique_emails: set[str] = set()
        for email in emails:
            local_name, domain_name = email.split('@')
            local_name = local_name.split('+')[0]
            local_name = local_name.replace('.', '')
            email = local_name + '@' + domain_name
            unique_emails.add(email)
        return len(unique_emails)

# Isomorphic Strings
# Question --> Given two strings s and t, determine if they are isomorphic.

# Two strings s and t are isomorphic if the characters in s can be replaced to get t.

# All occurrences of a character must be replaced with another character while preserving the order of characters. 
# No two characters may map to the same character, but a character may map to itself.

# Solution : mapping the string simultaneously i.e S ---->  T and T -----> S which means we are mapping it 
# it both way


class Solution:
    def isIsomorphic(self, s: str, t: str) -> bool:
        mapST , mapTS = {} , {}
        
        for c1 , c2 in zip( s , t):
            if((c1 in mapST and mapST[c1] != c2) or c2 in mapTS and mapTS[c2] != c1):
                return False
            mapST[c1] = c2
            mapTS[c2] = c1
                      
        return True

# Can Place Flowers
# QUESTION --> You have a long flowerbed in which some of the plots are planted, and some are not. However, 
# flowers cannot be planted in adjacent plots.
# Given an integer array flowerbed containing 0's and 1's, where 0 means empty and 1 means not empty, and an integer n, 
# return true if n new flowers can be planted in the flowerbed without violating the no-adjacent-flowers rule and false otherwise.
# Solution --> [ 1 , 0 , 1]
            #    [ 1 , 0 , 0 , 1]
                # 0 [ 1 , 0 , 0 , 0 , 1] 0
                # 0 [  0 , 0 , 1] 0
                
# we are going to puting an Zero at the begining and end of the array and before we can plant
# a flower we are going to check if the spot is 0 and the spot before it is 0 and after it is 0 as well
# and we keep decreementing our flower and check before the loops run out
class Solution3:
    def canPlaceFlowers(self, flowerbed: list[int], n: int) -> bool:
       # Solution with O(n) space complexity
       #  adding 0 at the beginning and ending of the array
       f = [0] + flowerbed + [0]       
       for i in range(1, len(f) - 1):  # skip first & last because of added zero
           if f[i - 1] == 0 and f[i] == 0 and f[i + 1] == 0:
            #  plant flower if we have  3 zero
               f[i] = 1
               #  decreement flower
               n -= 1
       return n <= 0

# Majority Element
# Question --> Given an array nums of size n, return the majority element.

# The majority element is the element that appears more than ⌊n / 2⌋ times. You may assume that the majority element always exists in the array.

 

# Example 1:

# Input: nums = [3,2,3]
# Output: 3

# Solution : using hashmap to set {number - count} and keep track of the maximum value 
class Solution:
    def majorityElement(self, nums : list[int]) -> int :
        count = {}
        res , maxCount = 0 , 0
        for n in nums :
            count[n] = 1 + count.get(n, 0)
            res = n if count[n] > maxCount else res
            maxCount = max(count[n] , maxCount)
        return res
    
# Boyer moore Algorithms
#   keeping track of the most occurence in a single variable e.g
#     nums = [ 1 , 1 , 2 , 3 , 2 , 1]
#     res = 0
#     count = 0
    # we keep increementing count if the value is the same and decreementing count if the value is different
    class Solution:
        def majorityElem(self , nums : list[int]) -> int :
            res , count = 0 , 0
            for n in nums:
                # setting result to the current number if the res == 0
                if res == 0:
                    res = n
                count +=  ( 1 if n == res else -1)
            return res


# Next Greater Element
# Question --> The next greater element of some element x in an array is the first greater element that is to the right of x in the same array.
# You are given two distinct 0-indexed integer arrays nums1 and nums2, where nums1 is a subset of nums2.
# For each 0 <= i < nums1.length, find the index j such that nums1[i] == nums2[j] and determine the next greater element of nums2[j] in nums2. 
# If there is no next greater element, then the answer for this query is -1.
# Return an array ans of length nums1.length such that ans[i] is the next greater element as described above.

# Solution :

class Solution:
   def nextGreaterElement(self, nums1: list[int], nums2: list[int]) -> list[int]:
        # assigning hashmap with value --- index
        nums1Idx  = { n : i for  i , n in enumerate(nums1)}
        # initailizing res to -1
        res = [-1] * len(nums1)


        for i in range(len(nums2)):
            # we don't want to do anything if it doesn't appear in nums1 thenn we continue
            if nums2[i] not in nums1Idx:
                continue
            # loop starting frm the next value
            for j in range(i + 1 , len(nums2)):
                if nums2[j] > nums2[i]:


