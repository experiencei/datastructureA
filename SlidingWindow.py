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

 

Example 1:

Input: s = "ABAB", k = 2
Output: 4
Explanation: Replace the two 'A's with two 'B's or vice versa.




class Solution:
    def characterReplacement(self, s: str, k: int) -> int:
        count = {}
        
        l = 0
        maxf = 0
        for r in range(len(s)):
            count[s[r]] = 1 + count.get(s[r], 0)
            maxf = max(maxf, count[s[r]])

            if (r - l + 1) - maxf > k:
                count[s[l]] -= 1
                l += 1

        return (r - l + 1)
