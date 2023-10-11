# Best Time to Buy and Sell Stock II=

# Question --> You are given an array prices where prices[i] is the price of a given stock on the ith day.
# You want to maximize your profit by choosing a single day to buy one stock and choosing a different day in the future to sell that stock.
# Return the maximum profit you can achieve from this transaction. If you cannot achieve any profit, return 0.


# Example 1:

# Input: prices = [7,1,5,3,6,4]
# Output: 5
# Explanation: Buy on day 2 (price = 1) and sell on day 5 (price = 6), profit = 6-1 = 5.
# Note that buying on day 2 and selling on day 1 is not allowed because you must buy before you sell.


class Solution:
    def maxProfit(self, prices: list[int]) -> int:
        res = 0
        
        lowest = prices[0]
        for price in prices:
            if price < lowest:
                lowest = price
            res = max(res, price - lowest)
        return res

class Solution:
  def maxProfit(self , nums: list[int] ) -> int:

    profit = 0
    # we're are not starting at first index because no previous price to compare it to
    for n in range( 1 , len(nums)):
      if nums[n] > nums[ n -1]:
        profit += (nums[n] - nums[n - 1])

    return profit


# Contains Duplicate II
# Question --> Given an integer array nums and an integer k, 
#             return true if there are two distinct indices i and j in the array such that nums[i] == nums[j] and abs(i - j) <= k.


# Example 1:

# Input: nums = [1,2,3,1], k = 3
# Output: true

class Solution:
    def containsNearbyDuplicate(self, nums: list[int], k: int) -> bool:
        window = set()
        L = 0

        for R in range(len(nums)):
            if R - L > k:
                window.remove(nums[L])
                L += 1
            if nums[R] in window:
                return True
            window.add(nums[R])
        return False



# Number of Sub-arrays of Size K and Average Greater than or Equal to Threshold


# Question --> Given an array of integers arr and two integers k and threshold, return the number of sub-arrays of size k and average greater than or equal to threshold
 

# Example 1:

# Input: arr = [2,2,2,2,5,5,5,8], k = 3, threshold = 4
# Output: 3
# Explanation: Sub-arrays [2,5,5],[5,5,5] and [5,5,8] have averages 4, 5 and 6 respectively. All other sub-arrays of size 3 have averages less than 4 (the threshold).



class Solution:
    def numOfSubarrays(self, arr: list[int], k: int, threshold: int) -> int:
        #initializing res to count the number of results we have.
        res = 0
        # summing up the value up until k without k inclusive
        curSum = sum(arr[:k-1])

        for L in range(len(arr) - k + 1):
            # L + k - 1 --> getting the right index
            curSum += arr[L + k - 1]
            if (curSum / k) >= threshold:
                res += 1
                # we are subtracting the left index from array
            curSum -= arr[L]
        return res


# Longest Substring Without Repeating Characters

#Question --> Given a string s, find the length of the longest substring without repeating characters.



# Example 1:

# Input: s = "abcabcbb"
# Output: 3
# Explanation: The answer is "abc", with the length of 3.



class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        charSet = set()
        l = 0
        res = 0

        for r in range(len(s)):
            while s[r] in charSet:
                charSet.remove(s[l])
                l += 1
            charSet.add(s[r])
            res = max(res, r - l + 1)
        return res


# Longest Repeating Character Replacement

#Question --> You are given a string s and an integer k. You can choose any character of the string and change it to any other uppercase English character.
#  You can perform this operation at most k times.

# Return the length of the longest substring containing the same letter you can get after performing the above operations.

 

# Example 1:

# Input: s = "ABAB", k = 2
# Output: 4
# Explanation: Replace the two 'A's with two 'B's or vice versa.

# solution : we're going to be having a hashmap counting the number of letter
# difference btw windowLen - length of max char is "Number of replacement to be made"
# windowLen - length of Max character <= k


# Solution -->
class Solution:
    def characterReplacement(self, s: str, k: int) -> int:
      #hashmap for maintaining the count of character
        count = {}
        
        l = 0
        # maxF to keep track of maximum value we've encounter
        maxf = 0
        for r in range(len(s)):
            count[s[r]] = 1 + count.get(s[r], 0)
            # track the maximum value 
            maxf = max(maxf, count[s[r]])

            # shift the left pointer if the number of replacement is greater than K
            if (r - l + 1) - maxf > k:
                count[s[l]] -= 1
                l += 1

        return (r - l + 1)





# 567. Permutation in String

#Question --> Given two strings s1 and s2, return true if s2 contains a permutation of s1, or false otherwise.

# In other words, return true if one of s1's permutations is the substring of s2.

 #solution -->  permutaion is also like anagram , we will be using to hashmap or array
#  to compute the value and compare it , we will keep adjusting the window and compare

# Example 1:

# Input: s1 = "ab", s2 = "eidbaooo"
# Output: true
# Explanation: s2 contains one permutation of s1 ("ba").


class Solution:
    def checkInclusion(self, s1: str, s2: str) -> bool:
        # if the length of s1 is greater than s2 will be returning false 
        if len(s1) > len(s2):
            return False

        #initailize the array to empty zero and assign the letter in it
        s1Count, s2Count = [0] * 26, [0] * 26
        for i in range(len(s1)):
            # Ascii value as index and +1 if it's present
            s1Count[ord(s1[i]) - ord("a")] += 1
            s2Count[ord(s2[i]) - ord("a")] += 1

        # matches variable to count number of matches if it's 26 it means correct
        matches = 0
        for i in range(26):
            #if s2 and s1 is the same add 1 else 0
            matches += 1 if s1Count[i] == s2Count[i] else 0

        
        l = 0
        for r in range(len(s1), len(s2)):
            if matches == 26:
                return True

            index = ord(s2[r]) - ord("a")
            s2Count[index] += 1
            # if by incrementing s2Count the array/hashmap are equal then we want to increment the matches
            if s1Count[index] == s2Count[index]:
                matches += 1
            # if by incrementing s2Count the array/hashmap of s2count **over** increment then we want to reduce the matches 
            elif s1Count[index] + 1 == s2Count[index]:
                matches -= 1

            index = ord(s2[l]) - ord("a")
            s2Count[index] -= 1
            if s1Count[index] == s2Count[index]:
                matches += 1
            elif s1Count[index] - 1 == s2Count[index]:
                matches -= 1
            l += 1
            # which will return true or false
        return matches == 26




#  Frequency of the Most Frequent Element



# Question --> The frequency of an element is the number of times it occurs in an array.
# You are given an integer array nums and an integer k. 
# In one operation, you can choose an index of nums and increment the element at that index by 1.

# Return the maximum possible frequency of an element after performing at most k operations.

 

# Example 1:

# Input: nums = [1,2,4], k = 5
# Output: 3
# Explanation: Increment the first element three times and the second element two times to make nums = [4,4,4].
# 4 has a frequency of 3.



# Solution :  we have to sort cause it is easier to change (increment) value on the left side of it than random value in the array
#  the formular is expand the window while (num[r] * windowLength < totalSumInWindow + k)
# while because [ 1 , 1 , 1 , 2 , 2 , 4 ] k = 2
# total sum of the window + k (what we have) is greater than num[r] (number we are trying to make most frequent) * windowLength(all what are trying to make frequent)

class Solution:
    def maxFrequency(self, nums: list[int], k: int) -> int:
        nums.sort()
        
        l , r = 0 , 0
        res , total = 0 , 0
        
        while r < len(nums):
            # we want to keep adding to total
            total += nums[r]
            
            # while the condition is not met
            while nums[r] * ( r - l + 1) > total + k:
                # we want to decrement total by left point in nums 
                total -= nums[l]
                l += 1
                
            # maximum window we have as long as condition is met 
            res = max(res , r - l + 1)
            r += 1
            
        return res
            





# class Solution:
#     def maxFrequency(self, nums: List[int], k: int) -> int:
        
#         nums.sort()
        
#         start=0
#         sm=0
#         ans=1
        
#         for i,val in enumerate(nums):
            
#             op=val*(i-start)-sm
            
#             while op>k:
#                 op-=val-nums[start]
#                 sm-=nums[start]
#                 start+=1
            
#             sm+=val
#             ans=max(ans,i-start+1)
                    
#         return ans


#Fruit Into Baskets

#Question -->  You are visiting a farm that has a single row of fruit trees arranged from left to right. 
# The trees are represented by an integer array fruits where fruits[i] is the type of fruit the ith tree produces.
# You want to collect as much fruit as possible. However, the owner has some strict rules that you must follow:

# You only have two baskets, and each basket can only hold a single type of fruit. There is no limit on the amount of fruit each basket can hold.
# Starting from any tree of your choice, you must pick exactly one fruit from every tree (including the start tree) while moving to the right. The picked fruits must fit in one of your baskets.
# Once you reach a tree with fruit that cannot fit in your baskets, you must stop.
# Given the integer array fruits, return the maximum number of fruits you can pick.


# Example 1:

# Input: fruits = [1,2,1]
# Output: 3
# Explanation: We can pick from all 3 trees.

# Solution : we want a consecutive of subsequence of the 2 different fruit.
# we need an hashmap to count the fruit and its occurences  

class Solution:
    def totalFruit(self , fruits: list[int]) -> int:
        count = collections.defaultdict(int) #fruitType -> countInBasket
        l , total , res = 0 , 0 , 0
        for r in range(len(fruits)):
            #counting the occurence of the fuit in hashmap 
            count[fruits[r]] += 1
            total += 1
            
            # if the count is more than 2 , since we're only allow to pick 2 fruits
            while len(count) > 2:
                f = fruits[l]
                # reduce the count and deduct from the total
                count[f] -= 1
                total -= 1
                l += 1

                #
                if not count[f]:
                    count.pop(f)

            res = max(res , total)

            return res



# class Solution:
#     def totalFruit(self, fruits: list[int]) -> int:
#         max_length = 0
#         window_start = 0
#         fruit_frequencies = {}

#         for window_end in range(len(fruits)):
#             right_fruit = fruits[window_end]
#             if right_fruit not in fruit_frequencies:
#                 fruit_frequencies[right_fruit] = 0
#             fruit_frequencies[right_fruit] += 1

#             while len(fruit_frequencies) > 2:
#                 left_fruit = fruits[window_start]
#                 fruit_frequencies[left_fruit] -= 1
#                 if fruit_frequencies[left_fruit] == 0:
#                     del fruit_frequencies[left_fruit]
#                 window_start += 1
#             max_length = max(max_length, window_end - window_start + 1)
#         return max_length


# Find K Closest Elements
# Questions --> Given a sorted integer array arr, two integers k and x, return the k closest integers to x in the array. 
# The result should also be sorted in ascending order.

# An integer a is closer to x than an integer b if:

# |a - x| < |b - x|, or
# |a - x| == |b - x| and a < b

# Example 1:

# Input: arr = [1,2,3,4,5], k = 4, x = 3
# Output: [1,2,3,4]

# Log(n) + k
# More code but also more intuitive
class Solution:
    def findClosestElements(self, arr: list[int], k: int, x: int) -> list[int]:
        l, r = 0, len(arr) - 1

        # Find index of x or the closest val to x
        val, idx = arr[0], 0
        while l <= r:
            m = (l + r) // 2
            curDiff, resDiff = abs(arr[m] - x), abs(val - x)
            if curDiff < resDiff or (curDiff == resDiff and arr[m] < val):
                val, idx = arr[m], m

            if arr[m] < x:
                l = m + 1
            elif arr[m] > x:
                r = m - 1
            else:
                break

        l = r = idx
        for i in range(k - 1):
            if l == 0:
                r += 1
            elif r == len(arr) - 1 or x - arr[l - 1] <= arr[r + 1] - x:
                l -= 1
            else:
                r += 1
        return arr[l : r + 1]


# Log(n-k) + k
# Elegant but very difficult to understand
class Solution:
    def findClosestElements(self, arr: list[int], k: int, x: int) -> list[int]:
        l, r = 0, len(arr) - k

        while l < r:
            m = (l + r) // 2
            if x - arr[m] > arr[m + k] - x:
                l = m + 1
            else:
                r = m
        return arr[l : l + k]
