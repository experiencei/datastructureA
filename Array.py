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

from collections import collections

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

# Solution  : we want to initial the 1st one since we knew it going to be 1.


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
                    #  get the index from nums1 since it greater than the current & appear as well
                    idx = nums1Idx[nums2[i]]
                    # assign it
                    res[idx] = nums2[j]
                    break
        return res


# Range Sum Query - Immutable
# Question ---> Given an integer array nums, handle multiple queries of the following type:

# Calculate the sum of the elements of nums between indices left and right inclusive where left <= right.
# Implement the NumArray class:

# NumArray(int[] nums) Initializes the object with the integer array nums.
# int sumRange(int left, int right) Returns the sum of the elements of nums between indices left and right inclusive 
# (i.e. nums[left] + nums[left + 1] + ... + nums[right]).

# Solution : to avoid heavy computational range which is 0(n2) we have to compute the prefix
# and just get the range of computed prefix inside the array
class Solution:

    def __init__(self, nums: list[int]):
        self.prefix = []
        cur = 0
        for n in nums:
            cur += n
            self.prefix.append(cur)
        
        
    def sumRange(self, left: int, right: int) -> int:
        r = self.prefix[right] 
        l = self.prefix[left - 1] if left > 0 else 0
        return r - l


#  Find Pivot Index
# Question --> Given an array of integers nums, calculate the pivot index of this array.

# The pivot index is the index where the sum of all the numbers strictly to the left of the 
# index is equal to the sum of all the numbers strictly to the index's right.

# If the index is on the left edge of the array, then the left sum is 0 because there are no elements to the left.
#  This also applies to the right edge of the array.

# Return the leftmost pivot index. If no such index exists, return -1.

class Solution:
    def pivotIndex(self, nums: list[int]) -> int:
        total = sum(nums)  # O(n)

        leftSum = 0
        for i in range(len(nums)):
            rightSum = total - nums[i] - leftSum
            if leftSum == rightSum:
                return i
            leftSum += nums[i]
        return -1
        

# Find All Numbers Disappeared in an Array

# Question ==> Given an array nums of n integers where nums[i] is in the range [1, n], 
#              return an array of all the integers in the range [1, n] that do not appear in nums.

# Solution : we will have an hashSet that contain value { 1 , n} and we loop 
#    through the nums array and pop from the from the HashSet which is 0(1)
    #   Input: nums = [4,3,2,7,8,2,3,1]
    #             Output: [5,6]  i.e we are creating { 1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 } and 
    #             cancel out any occurence from it
            # Space complexity -- 0(n)
            # Time complexity -- 0(n)
# Efficients -- since it"s one - one mapping how about if we see 1 we set the value
    #  at index 0 to -1 
    #  1 we set the value at index 0 to -1
    #  4 we set the value at index 3 to -4

    #         0   1  2   3   
    #       [ 1 , 4, 4 , 4,  ]

    #       0 <---> n - 1   == index
    #       1 <---> n       == value


class Solution:
    def findDisappearedNumbers(self, nums: list[int]) -> list[int]:
        for n in nums:
            i = abs(n) - 1
            nums[i] = -1 * abs(nums[i])

            res = []
        for i, n in enumerate(nums):
            if n > 0:
                res.append(i + 1)
        return res

# Maximum Number of Balloons
# Question --> Given a string text, you want to use the characters of text to 
# form as many instances of the word "balloon" as possible.
# You can use each character in text at most once. Return the maximum number of instances that can be formed

# Solution :  for "nlaebolko"
        # Counter({'l': 2, 'o': 2, 'n': 1, 'a': 1, 'e': 1, 'b': 1, 'k': 1})
        # Counter({'l': 2, 'o': 2, 'b': 1, 'a': 1, 'n': 1})
        
        # Idea is to check how many parts can be obtained from given string
        # for example. In parent we have 2 o's, we need 2 o's to make 1 baloon
        # so we divide those 2 and get 1. and we keep updating the minimum
from collections import Counter

class Solution:
    def maxNumberOfBalloons(self , text : str) -> int:
        countText = Counter(text)
        balloon = Counter("balloon")


        res = len(text)  # or float("inf")
        for c in balloon:
            res = min(res, countText[c] // balloon[c])
        return res


#  Word Pattern

# Question --> Given a pattern and a string s, find if s follows the same pattern.
# Here follow means a full match, such that there is a bijection 
# between a letter in pattern and a non-empty word in s.
# Example 1:
# Input: pattern = "abba", s = "dog cat cat dog"
# Output: true

# Solution --> more like ISOMORPHIC string we will have pair matching in hashmap
class Solution:
    def wordPattern(self, pattern: str, s: str) -> bool:
        words = s.split(" ")
        if len(pattern) != len(words):
            return False
        charToWord = {}
        wordToChar = {}

        for c, w in zip(pattern, words):
            if c in charToWord and charToWord[c] != w:
                return False
            if w in wordToChar and wordToChar[w] != c:
                return False
            charToWord[c] = w
            wordToChar[w] = c
        return True


# Design HashSet
# Question --> Design a HashSet without using any built-in hash table libraries.

# Implement MyHashSet class:

# void add(key) Inserts the value key into the HashSet.
# bool contains(key) Returns whether the value key exists in the HashSet or not.
# void remove(key) Removes the value key in the HashSet. If key does not exist in the HashSet, do nothing.

# solution
class MyHashSet:

    def __init__(self):
        self.bucket_size = 1
        self.hs = [[] for _ in range(self.bucket_size)]

    def hashFunc(self, num: int) -> int:
        return num % self.bucket_size
        
    def add(self, key: int) -> None:
        hash_value = self.hashFunc(key)
        # first item in the bucket, don't add duplicate keys
        if key not in self.hs[hash_value]:
            self.hs[hash_value].append(key)

    def remove(self, key: int) -> None:
        hash_value = self.hashFunc(key)
        # Check weather the hashed bucket exist
        if not self.hs[hash_value]:
            return False
        else:
            try:
                self.hs[hash_value].pop(self.hs[hash_value].index(key))
            # Not found in the bucket either
            except ValueError:
                return False
        
    def contains(self, key: int) -> bool:
        hash_value = self.hashFunc(key)
        return key in self.hs[hash_value]


# Design Hashmap
# Question --> 
# Design a HashMap without using any built-in hash table libraries.

# Implement the MyHashMap class:

# MyHashMap() initializes the object with an empty map.
# void put(int key, int value) inserts a (key, value) pair into the HashMap.
#  If the key already exists in the map, update the corresponding value.
# int get(int key) returns the value to which the specified key is mapped, or -1 if this map contains no mapping for the key.
# void remove(key) removes the key and its corresponding value if the map contains the mapping for the key.

# Solution --> efficient use a dummyNode to avoid edge cases of pointing to null 
#   after hasfunction of moding value with 1000 [ index | key : value]
#    after moding the key to be index we still want to keep tap of the real value of key

class ListNode:
    def __init__(self, key=-1, val=-1, next=None):
        self.key = key
        self.val = val
        self.next = next

class MyHashMap:
    def __init__(self):
        self.map = [ListNode() for i in range(1000)]

    def hashcode(self, key):
        return key % len(self.map)

    def put(self, key: int, value: int) -> None:
        cur = self.map[self.hashcode(key)]
        while cur.next:
            if cur.next.key == key:
                cur.next.val = value
                return
            cur = cur.next
        cur.next = ListNode(key, value)

    def get(self, key: int) -> int:
#         it make sense to add .next here because we don't want to start at dummy node 
        cur = self.map[self.hashcode(key)].next
        while cur and cur.key != key:
            cur = cur.next
        if cur:
            return cur.val
        return -1  

    def remove(self, key: int) -> None:
        cur = self.map[self.hashcode(key)]
        while cur.next and cur.next.key != key:
            cur = cur.next
        if cur and cur.next:
            cur.next = cur.next.next

# Sort an Array
# Question --> Given an array of integers nums, sort the array in ascending order and return it.
# You must solve the problem without using any built-in functions in O(nlog(n)) time complexity and 
# with the smallest space complexity possible.

# Solution --. we will be using MergeSort whcih is going to be 0(nlog(n)) and Divide and Conquer appraoch
                        #       |
                        #     [ 5 , 2 , 3 , 1]
                        #      /         \
                        #     /           \          n/2
                        #   [5 , 2]       [3 , 1]
                           
                        #    |             |
                        #    5 , 2         3 , 1
    # we recursively divide the array and use 3 pointer to sort inplace 
    # i.e no extra space complexity


class Solution:
    def sortArray(self, nums: list[int]) -> list[int]:
        def merge(arr, L, M, R):
            left, right = arr[L:M+1], arr[M+1:R+1]
            i, j, k = L, 0, 0
            while j < len(left) and k < len(right):
                if left[j] <= right[k]:
                    arr[i] = left[j]
                    j += 1
                else:
                    arr[i] = right[k]
                    k += 1
                i += 1
            while j < len(left):
                arr[i] = left[j]
                j += 1
                i += 1
            while k < len(right):
                arr[i] = right[k]
                k += 1
                i += 1

        def mergeSort(arr, l, r):
            if l == r:
                return arr
            m = (l + r) // 2
            mergeSort(arr, l, m)
            mergeSort(arr, m + 1, r)
            merge(arr, l, m, r)
            return arr
        
        return mergeSort(nums, 0, len(nums) - 1)

# Top K Frequent Elements
# Question --> Given an integer array nums and an integer k, return the k most frequent elements. 
# You may return the answer in any order.

# Example 1:
# Input: nums = [1,1,1,2,2,3 , 6], k = 2
# Output: [1,2]

# Solution : will be using hashmap and counting number of occurences of val
#   i.e [ 1 --> 3] which means 1 occur 3 times
#       [2 --> 2]
#    which is definitely ineffecient
    #    ----> Another Solution is to use heapity / max heap
                    #   (Bucket Sort)
    # Efficient--> i ( count) 0 , 1 , 2 , 3 , 4 , 5 , 6
    #               values       [3,6] [2] [1]

    # which means the maximum sizes of the bucket is bounded by number of the array
    # and keep loop from the back to check top K frequent element

class Solution:
    def topKFrequent(self, nums: list[int], k : int) -> list[int] :
        count = {}
        freq = [[] for i in range(nums) + 1]

        for n in nums:
            count[n] = 1 + count.get(n, 0)
        for n , c in count.items():
            freq[c].append(n)

        res = []
        for i in range(len(freq)-1 , 0 , -1):
            for n in freq[i]:
                res.append(n)
                if len(res) == k:
                    return res

            
# Product of Array Except Self
# Question --> Given an integer array nums, return an array answer such that answer[i] 
# is equal to the product of all the elements of nums except nums[i].
# The product of any prefix or suffix of nums is guaranteed to fit in a 32-bit integer.
# You must write an algorithm that runs in O(n) time and without using the division operation.

 
# Example 1:

# Input: nums = [1,2,3,4]
# Output: [24,12,8,6]

# Solution : we will be using prefix and postfix approach to solve this problem
#  and inittailize 1 at the beginning and end of the array
               
    #  pre = 1 , 1 , 2 , 6         post = 1 , 4 , 12 , 24
                #    --------------->
                #    <--------------
                #   [ 1 , 2 , 3 , 4 ] 
    # output =  [ 1  -> 24 , 1 -> 12 , 2 -> 8 , 6 ]


class Solution:
    def productExceptSelf(self, nums: list[int]) -> list[int]:
        res = [1] * (len(nums))

        prefix = 1
        for i in range(len(nums)):
            res[i] = prefix
            prefix *= nums[i]
        postfix = 1
        for i in range(len(nums) - 1, -1, -1):
            res[i] *= postfix
            postfix *= nums[i]
        return res


# Valid Sudoku
# Question --> Determine if a 9 x 9 Sudoku board is valid. Only the filled cells need to be validated according to the following rules:

# Each row must contain the digits 1-9 without repetition.
# Each column must contain the digits 1-9 without repetition.
# Each of the nine 3 x 3 sub-boxes of the grid must contain the digits 1-9 without repetition.
# Note:

# A Sudoku board (partially filled) could be valid but is not necessarily solvable.
# Only the filled cells need to be validated according to the mentioned rules.


# Solution : to validate any duplicate in row an column we're going to use hashset
#    and Timecomplexity and Spacecomplexity == 0(9^2)

#       0      1         2
#     0 1 2   3 4 5   6 7 8
#   0
# 0 1
#   2

#   3
# 1 4
#   5

#   6                  9 
# 2 7
#   8

#   to get the co-ordinate of 9 [6 / 3 row , 6  / 3 column]  3 is the number sub-boxes
# remember to round it down 

# key = (r / 3 , c / 3)
# val = hashset to avoid duplicate

class Solution:
    def isValidSudoku(self, board: list[list[str]]) -> bool:
        cols = collections.defaultdict(set)
        rows = collections.defaultdict(set)
        squares = collections.defaultdict(set)  # key = (r /3, c /3)

        for r in range(9):
            for c in range(9):
                if board[r][c] == ".":
                    continue
                if (
                    board[r][c] in rows[r]
                    or board[r][c] in cols[c]
                    or board[r][c] in squares[(r // 3, c // 3)]
                ):
                    return False
                cols[c].add(board[r][c])
                rows[r].add(board[r][c])
                squares[(r // 3, c // 3)].add(board[r][c])

        return True


# Longest Consecutive Sequence
# Question  --> Given an unsorted array of integers nums, return the length of the longest consecutive elements sequence.

# You must write an algorithm that runs in O(n) time.

# Example 1:

# Input: nums = [100,4,200,1,3,2]
# Output: 4
# Explanation: The longest consecutive elements sequence is [1, 2, 3, 4]. Therefore its length is 4.

# Solution :     1,2,3,4      100    200
   #             1            2      3 
# to get the start of each sequence of set we need to check 
# if it doesn't have a neighbour i.e 1 doesn't have which 0 and 100 which is 99 & 200 which is 199

# iterate through the array of set check if they have left neighbours if yes then
# they are start of the sequence and we check the right neighbours as well to keep increasing the left neigbours

class Solution:
    def longestConsecutive(self, nums : list[int]) -> int:
        numsSet = set(nums)
        longest = 0

        for n in nums:
            # check if its the start of a sequence
            if (n - 1) not in numsSet:
                length = 1
                while( n + 1) in numsSet:
                    length += 1
                longest = max(length , longest)
        return longest


# Sort colors

# Question ---> Given an array nums with n objects colored red, white, or blue, sort them in-place so that objects of the same color are adjacent,
#  with the colors in the order red, white, and blue.
# We will use the integers 0, 1, and 2 to represent the color red, white, and blue, respectively.
# You must solve this problem without using the library's sort function.
 
# Example 1:

# Input: nums = [2,0,2,1,1,0]
# Output: [0,0,1,1,2,2]

# Solution: 
    # Efficient is Quicksort with 2 pointer , with 1 pointer will be 2pass algorithms.
    # we are going to have left and right pointer and i pointer & if i == 1 we don't want to swap

class Solution:
    def sortColors(self , nums: list[int]) -> None :
        """
        do  not return anything , modify nums in-place instead.
        """
        l , r = 0 , len(nums) - 1
        i = 0
        def swap( i , j):
            tmp = nums[i]
            nums[i] = nums[j]
            nums[j] = tmp

        while i <= r:
            if nums[i] == 0:
                swap(l , i)
                i += 1
            elif nums[i] == 2:
                swap(i , r)
                r -= 1
                i -= 1
            i += 1


# Encode and Decode TinyURL

# Question : Note: This is a companion problem to the System Design problem: Design TinyURL.
# TinyURL is a URL shortening service where you enter a URL such as https://leetcode.com/problems/design-tinyurl and 
# it returns a short URL such as http://tinyurl.com/4e9iAk. Design a class to encode a URL and decode a tiny URL.

# There is no restriction on how your encode/decode algorithm should work.
#  You just need to ensure that a URL can be encoded to a tiny URL and the tiny URL can be decoded to the original URL.

# Implement the Solution class:

# Solution() Initializes the object of the system.
# String encode(String longUrl) Returns a tiny URL for the given longUrl.
# String decode(String shortUrl) Returns the original long URL for the given shortUrl. It is guaranteed that the given shortUrl was encoded by the same object.
 
# Example 1:

# Input: url = "https://leetcode.com/problems/design-tinyurl"
# Output: "https://leetcode.com/problems/design-tinyurl"



# Solution : we are having 2 hashmap 1 for encode and other for decode 
#  and we're encoding with figure which is efficient and Time complexity of 0(1)
class Codec:
    def __init__(self):
        self.encodeMap = {}
        self.decodeMap = {}
        self.base = "http://tinyurl.com/"

    def encode(self, longUrl: str) -> str:
        """Encodes a URL to a shortened URL.
        """
        if longUrl not in self.encodeMap: 
            shortUrl = self.base + str(len(self.encodeMap) + 1)
            self.encodeMap[longUrl] = shortUrl
            self.decodeMap[shortUrl] = longUrl
        return self.encodeMap[longUrl]

    def decode(self, shortUrl: str) -> str:
        """Decodes a shortened URL to its original URL.
        """
        return self.decodeMap[shortUrl]

# Solid wall

# Question : There is a rectangular brick wall in front of you with n rows of bricks. 
# The ith row has some number of bricks each of the same height (i.e., one unit) but they can be of different widths. 
# The total width of each row is the same.
# Draw a vertical line from the top to the bottom and cross the least bricks. 
# If your line goes through the edge of a brick, then the brick is not considered as crossed.
#  You cannot draw a line just along one of the two vertical edges of the wall, in which case the line will obviously cross no bricks.
# Given the 2D array wall that contains the information about the wall, return the minimum number of crossed bricks after drawing such a vertical line.

# Solution : we will be counting the number of space in each brick horizontally instead of vertically
    #  we're are going to use a hashmap instead 

    #  key = number of rows 
    #  value = how many space we're going to

# Note i.e the maximum number of value(space we have) = min number of cut through
#        result = Total rows - max(gaps)
# the main reason we initializecountGap = { 0 : 0 } is because when we're taking our max(countGap.values())
# if there's no value there it will throw an error
class Solution:
    def leastBricks(self, wall: list[list[int]]) -> int:
        countGap = { 0 : 0 }    # { Position : Gap count }

        for r in wall:
            total = 0   # Position we at in the row
            # [:-1] we don't want to count the last most edge of the brick
            for b in r[:-1]:
                total += b
                countGap[total] = 1 + countGap.get(total, 0)

        return len(wall) - max(countGap.values())    # Total number of rows - Max gap


# Best Time to Buy and Sell Stock II
# Question --> You are given an integer array prices where prices[i] is the price of a given stock on the ith day.
# On each day, you may decide to buy and/or sell the stock. You can only hold at most one share of the stock at any time. 
# However, you can buy it then immediately sell it on the same day.
# Find and return the maximum profit you can achieve.

# Input: prices = [7,1,5,3,6,4]
# Output: 7
# Explanation: Buy on day 2 (price = 1) and sell on day 3 (price = 5), profit = 5-1 = 4.
# Then buy on day 4 (price = 3) and sell on day 5 (price = 6), profit = 6-3 = 3.
# Total profit is 4 + 3 = 7.

# Solution : we are adding every increase to our total profit . 
#            that means buy low and sell high


class Solution:
    def maxProfit(self, prices: list[int]) -> int:
        max_profit = 0
        # we're going to be skipping the first index since we can't compare it to any previous position
        for i in range(1, len(prices)):
            if prices[i] > prices[i-1]:
                max_profit += prices[i] - prices[i-1]
        return max_profit


# Subarray Sum Equals K
# Question --> 
# Given an array of integers nums and an integer k, return the total number of subarrays whose sum equals to k.
# A subarray is a contiguous non-empty sequence of elements within an array.
# Example 1:

# Input: nums = [1,1,1], k = 2
# Output: 2

Solution --> 
#   sliding window is ineffective because it is 0(n^2)