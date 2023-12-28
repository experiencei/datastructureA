# Binary Search

# Question ---> Given an array of integers nums which is sorted in ascending order, and an integer target, 
# write a function to search target in nums. If target exists, then return its index. Otherwise, return -1.
# You must write an algorithm with O(log n) runtime complexity.


# Example 1:

# Input: nums = [-1,0,3,5,9,12], target = 9
# Output: 4
# Explanation: 9 exists in nums and its index is 4

# solution --> we're going to be using divide and conquer method
# [-1,0,3,5,9,12], since they are already in sorted order
# if the target 9 is greater than the middle of the array (then we eliminate the other half of the array)

class Solution:
  def binarySearch(self, nums : list[int] , target : int) -> int:

    # initialize the left and right pointer
    l , r = 0 , len(nums) - 1

    # as long as r is greater than l
    while l <= r:

      m = l + ((r - l) // 2) 

      # re assign right index if middle value is greater than target
      if nums[m] > target:
        r = m - 1

      # re assign left index if middle value is greater than target
      elif nums[m] < target:
        l = m + 1
        
      else : 
         return m

    return -1
  
  

# Search Insert Position

# Question --> Given a sorted array of distinct integers and a target value, return the index if the target is found. If not, return the index where it would be if it were inserted in order.
#  You must write an algorithm with O(log n) runtime complexity.

# Example 1:

# Input: nums = [1,3,5,6], target = 5
# Output: 2

# Example 3:

# Input: nums = [1,3,5,6], target = 7
# Output: 4

# Solution --> we want to run a binary search on the array given . if target not found
# return the index where it would be if it were inserted in order
 
class Solution:
    def searchInsert(self, nums: list[int], target: int) -> int:
        # O(log n) and O(1)   
        # initialize the left and right pointer
        low, high = 0, len(nums)

        # as long as r is greater than l
        while low<high:
            

            mid = low +(high - low) // 2

            # re assign left index if middle value is greater than target
            if target > nums[mid]:
                low = mid + 1

            # re assign right index if middle value is greater than target
            else:
                high = mid

        # left index will always be the fall through index if target not found
        # [2] target = 1 , L will fall to zero and r = 1
        # [2] target = 3 , L will fall to since we will be searching to right of array
        return low



# Guess Number Higher or Lower

# Question --> We are playing the Guess Game. The game is as follows:
# I pick a number from 1 to n. You have to guess which number I picked.
# Every time you guess wrong, I will tell you whether the number I picked is higher or lower than your guess.

# You call a pre-defined API int guess(int num), which returns three possible results:

# -1: Your guess is higher than the number I picked (i.e. num > pick).
# 1: Your guess is lower than the number I picked (i.e. num < pick).
# 0: your guess is equal to the number I picked (i.e. num == pick).


# Return the number that I picked.

# Example 1:

# Input: n = 10, pick = 6
# Output: 6


# Solution ---> we are going to be using a binary search to find the guess number according to the rules and
#               API given

class Solution:
   def guessNumber(self, nums : int , guess) -> int:
      
      l , r = 0 , nums

      mid = l + (r - l) // 2

      while True:

        guessNum = guess(mid)

        if guessNum == -1:
           r = mid - 1
        elif guessNum == 1:
           l = mid + 1
        else:
           return mid    




# Arranging Coins

# Question --> You have n coins and you want to build a staircase with these coins. 
# The staircase consists of k rows where the ith row has exactly i coins. 
# The last row of the staircase may be incomplete.
# Given the integer n, return the number of complete rows of the staircase you will build.

# Esample 1 : Input: n = 5
# Output: 2
# Explanation: Because the 3rd row is incomplete, we return 2.

# Solution --> we can sequencially add up everything together which will be 0(n)
# 10 - 1 = 9 - 2 = 7 - 3 = 4 - 4 = 0 which means with 10 we can make 4 steps of coins ---> bruteForce

# we can use gauss formular of suming up a large value in 0(log n) 
# 1.............100 
# 100 + 1 = 101
# 99 + 2 = 101
# meaning -----> lowest + n * (n/2)

# left boundary = 1 ----> we're guaranted to have 1
# right boundary = n ----> this is going to be our upper boundaries

# suppose we have 1,2,3,4,5

# mid = n = 3

# how many coins do we need to complete n
# 1 + n * (n/2) = 6 and we have 5 , so we cross out everything to the right of it.
# reassign r = 2 and n = 1 

class Solution:
    def arrangeCoins(self, n: int) -> int:
        # assign pointers
        l, r = 1, n

        # assign results
        res = 0

        # while right pointer is greater than left pointer
        while l <=r:
            
            # getting the mid number
            mid = (l+r)//2

            # number of coins needed to complete the mid value (with our formular)
            coins = (mid /2) * (mid+1)

            # if coins is greater than n adjust the right pointer
            if coins > n:
                r = mid - 1

            # if coins is lesser than n adjust the left pointer
            else:
                l = mid + 1

                # maximum of result and mid value the coins can completed is taking
                res = max(mid, res)

        return res

# Squares of a Sorted Array

# Question ----> Given an integer array nums sorted in non-decreasing order, 
# return an array of the squares of each number sorted in non-decreasing order.

# Example 1:

# Input: nums = [-4,-1,0,3,10]
# Output: [0,1,9,16,100]
# Explanation: After squaring, the array becomes [16,1,0,9,100].
# After sorting, it becomes [0,1,9,16,100].


# Solution --> we will be using binary search on the array to get efficient algorithm 0(n)
# we will have left and right pointer and have result array . we will be comparing the square of right and left pointers together
# and building the result in reverse order i.e from largest ---> lowest and later reverse the array back to non-decreasing element


class Solution:
   def sortedSquares(self, nums : list[int]) -> list[int]:
      # assign result array
      res = []

      # assign left and right pointer
      l , r = 0 , len(nums) - 1

      while l <r:
            
            # if square of left pointer is greater than square of right pointer
            if nums[l] * nums[l] > nums[r] * nums[r]:
                res.append(nums[l] * nums[l])
                l += 1
            
            # if square of right pointer is greater than square of left pointer
            else:
               res.append(nums[r] * nums[r])
               r += 1

      # reverse the array back
      return res[::-1]

      

# Valid Perfect Square


# Question --> Given a positive integer num, return true if num is a perfect square or false otherwise.
# A perfect square is an integer that is the square of an integer. In other words, it is the product of some integer with itself.
# You must not use any built-in library function, such as sqrt.


# Example 1:

# Input: num = 16
# Output: true
# Explanation: We return true because 4 * 4 = 16 and 4 is an integer


class Solution:
     def isPerfectSquare(self, num: int) -> bool:
        # initializing the left and right pointer
        l ,r = 1, num

        # as long as right pointer is not less than left
        while l <= r:

            mid = (l +r) // 2

            # if Square of mid is greater than nums readjust the right pointer
            if mid * mid > num:
                r = mid - 1
              
            # if Square of mid is less than nums readjust the left pointer
            elif mid * mid < num:
                l = mid + 1
              
            # else if it doesn't fall into the two categories return true
            else:
                return True
            
        # return false if it exist the while loops without returning true
        return False


# Sqrt(x)
# Question --> Given a non-negative integer x, return the square root of x rounded down to the nearest integer.
# The returned integer should be non-negative as well.

# You must not use any built-in exponent function or operator.
# For example, do not use pow(x, 0.5) in c++ or x ** 0.5 in python.


# Example 1:

# Input: x = 4
# Output: 2
# Explanation: The square root of 4 is 2, so we return 2.


# solution ---> we will be using binary search on it which is more efficient than
# 0(sqrt(n)) ... log n > 0(sqrt(n))

class Solution(object):
    def mySqrt(self, x):
        l, r = 0, x
        while l <= r:
            mid = l + (r-l)//2
            if mid * mid <= x < (mid+1)*(mid+1):
                return mid
            elif x < mid * mid:
                r = mid - 1
            else:
                l = mid + 1


# Single Element in a Sorted Array

# Question ---> You are given a sorted array consisting of only integers where every element appears exactly twice,
# except for one element which appears exactly once.
# Return the single element that appears only once.
# Your solution must run in O(log n) time and O(1) space.


# Example 1:

# Input: nums = [1,1,2,3,3,4,4,8,8]
# Output: 2

# Solution ---> we are going to be using binary search on it , to get the single element both side are not going to be the same
# [1,1,2,3,3,4,4,8,8] like the both side of 2 is totally different and to know the side we need to search for 
# (since our length of array is always going to be odd) hence we are going to the odd side



# If every element in the sorted array were to appear exactly twice, they would occur in pairs at indices i, i+1 for all even i.

# Equivalently, nums[i] = nums[i+1] and nums[i+1] != nums[i+2] for all even i.

# When we insert the unique element into this list, the indices of all the pairs following it will be shifted by one, negating the above relationship.

# So, for any even index i, we can compare nums[i] to nums[i+1].

# If they are equal, the unique element must occur somewhere after index i+1
# If they aren't equal, the unique element must occur somewhere before index i+1
# Using this knowledge, we can use binary search to find the unique element.

# We just have to make sure that our pivot index is always even, so we can use mid = 2 * ((lo + hi) // 4) instead of the usual mid = (lo + hi) // 2.

class Solution:
    def singleNonDuplicate(self, nums: list[int]) -> int:
        l=0
        h=len(nums)-1
        while l<h:
            m=2*((l+h)//4)
            if nums[m]==nums[m+1]:
                l=m+2
            else:
                h=m
        return nums[l]        


# Capacity To Ship Packages Within D Days

# Question ---> A conveyor belt has packages that must be shipped from one port to another within days days.
# The ith package on the conveyor belt has a weight of weights[i]. Each day, we load the ship with packages on the conveyor belt (in the order given by weights).
#  We may not load more weight than the maximum weight capacity of the ship.
# Return the least weight capacity of the ship that will result in all the packages on the conveyor belt being shipped within days days.

# Example 1:

# Input: weights = [1,2,3,4,5,6,7,8,9,10], days = 5
# Output: 15
# Explanation: A ship capacity of 15 is the minimum to ship all the packages in 5 days like this:
# 1st day: 1, 2, 3, 4, 5
# 2nd day: 6, 7
# 3rd day: 8
# 4th day: 9
# 5th day: 10

# Note that the cargo must be shipped in the order given, 
# so using a ship of capacity 14 and splitting the packages into parts like (2, 3, 4, 5), (1, 6, 7), (8), (9), (10) is not allowed.

# Solution --> let assumes days to be number of ships
# we want a minimum numbers of capacity that can carry the weights in that number of ships

# so we will run a binary search on the weights ( which the lower boundary is the highest value of weights) and the upper boundaries is the sum of all weights
# our binary search will give us the minimun number of capacity 

class Solution:
    def shipWithinDays(self, weights: list[int], days: int) -> int:
        # initialize the left and right pointer
        l , r = max(weights) , sum(weights)

        # result initialize to right pointer
        res = r

        def canShip(cap):
            # initialize the ships to 1 and the current capacity to cap passed in
            ships , currCap = 1 , cap
            for w in weights:
                # if capacity trips to - , it means we need to increment the ships
                if currCap - w < 0:
                    ships += 1
                    currCap = cap
                currCap -= w
            return ships <= days


        # running a normal binary search with helper function
        while l <= r:
            cap = l + ((r - l) // 2)
            if canShip(cap):
                res = min(res, cap)
                r = cap - 1 

            else:
                l = cap + 1
        return res

# Find Peak Element

# Question --> A peak element is an element that is strictly greater than its neighbors.
# Given a 0-indexed integer array nums, find a peak element, and return its index. 
# If the array contains multiple peaks, return the index to any of the peaks.
# You may imagine that nums[-1] = nums[n] = -∞. In other words,
#  an element is always considered to be strictly greater than a neighbor that is outside the array.
# You must write an algorithm that runs in O(log n) time.

# Example 1:

# Input: nums = [1,2,3,1]
# Output: 2
# Explanation: 3 is a peak element and your function should return the index number 2.


# Solution ---> it's always guaranted we are going to be having a greater neighbour
# and we can run our modify binary search on it to get the index of the peak element

class Solution:
    def findPeakElement(self, nums: list[int]) -> int:
        start, end = 0, len(nums) - 1
        nums.append(float('-inf')) # index -1 and n will both get this in python
        while start <= end: # binary search to eliminate half each time
            mid = (start + end) // 2
            if nums[mid - 1] < nums[mid] and nums[mid] > nums[mid + 1]:
                return mid
            if nums[mid - 1] > nums[mid]:
                end = mid - 1
            else:
                start = mid + 1


# Successful Pairs of Spells and Potions

# Question ---> You are given two positive integer arrays spells and potions, of length n and m respectively, 
# where spells[i] represents the strength of the ith spell and potions[j] represents the strength of the jth potion.

# You are also given an integer success. A spell and potion pair is considered successful if the product of their strengths is at least success.
# Return an integer array pairs of length n where pairs[i] is the number of potions that will form a successful pair with the ith spell.


# Solution --> we will be running a binary search on the potions to know where it will be least than the success number
# which will mlogn Time complexity

class Solution:
    def successfulPairs(self, spells: list[int], potions: list[int], s: int) -> list[int]:
        q=[]
        potions.sort()  
        
        # assigning the len(potions)  so as not to get highvalue if we are doing subtraction in end                                #Sort the potion array
        a=len(potions)


        for i in spells:
            count=0
            l=0                                   #We have to find a value which is less than (success/i) in sorted array  
            r=len(potions)                # binary seach will give index of that point and onwards that index all are 
            x=s/i                                #greater values 
            while l<r:
                mid=l+(r-l)//2
                if potions[mid]>=x:
                    r=mid
                else:
                    l=mid+1
            
            count=(a-l)  
                                                #Last - index that came with binary search
            q.append(count)
        return q


#  Search a 2D Matrix

# Question --> You are given an m x n integer matrix matrix with the following two properties:
# Each row is sorted in non-decreasing order.
# The first integer of each row is greater than the last integer of the previous row.
# Given an integer target, return true if target is in matrix or false otherwise.
# You must write a solution in O(log(m * n)) time complexity.

# Example 1:

# Input: matrix = [[1,3,5,7],[10,11,16,20],[23,30,34,60]], target = 3
# Output: true


# Solution ---> run a binary search on the matrix since it is non decreasing order to
# to get the actual row the target can fall into
# and run another binary search on the exact row to get the target value

# Time Complexity == mlogN + nlogN

class Solution:
    def searchMatrix(self, matrix: list[list[int]], target: int) -> bool:
        ROWS, COLS = len(matrix), len(matrix[0])

        top, bot = 0, ROWS - 1

        #running the first binary search on the matrix
        while top <= bot:
            row = (top + bot) // 2
            if target > matrix[row][-1]: 
                top = row + 1
            elif target < matrix[row][0]:
                bot = row - 1
            else:
                break

        if not (top <= bot):
            return False
        
        #running the second binary search on the row found
        row = (top + bot) // 2
        l, r = 0, COLS - 1
        while l <= r:
            m = (l + r) // 2
            if target > matrix[row][m]:
                l = m + 1
            elif target < matrix[row][m]:
                r = m - 1
            else:
                return True
        return False



# Koko Eating Bananas

# Question ---> Koko loves to eat bananas. There are n piles of bananas, the ith pile has piles[i] bananas. The guards have gone and will come back in h hours.
# Koko can decide her bananas-per-hour eating speed of k. Each hour, she chooses some pile of bananas and eats k bananas from that pile. If the pile has less than k bananas, 
# she eats all of them instead and will not eat any more bananas during this hour.
# Koko likes to eat slowly but still wants to finish eating all the bananas before the guards return.

# Return the minimum integer k such that she can eat all the bananas within h hours.

# Example 1:

# Input: piles = [3,6,7,11], h = 8
# Output: 4

# Solution ---> we want to minimize what koko can consume in specific amount of time
# the lowest k can can be is 1 and the highest it can be is (highest value in piles)
# then run a binary search on it and middle value of the [lowest , highest] is K
import math
class Solution:
    def minEatingSpeed(self, piles: list[int], h: int) -> int:
        l, r = 1, max(piles)

        # initialize result to be r (since we will be minimizing it)
        res = r

        while l <= r:
            k = (l + r) // 2

            totalTime = 0
            for p in piles:
                totalTime += math.ceil(float(p) / k)
            if totalTime <= h:
                res = min(k , res)
                r = k - 1
            else:
                l = k + 1
        return res

# Find Minimum in Rotated Sorted Array
# Question --> Suppose an array of length n sorted in ascending order is rotated between 1 and n times.
#  For example, the array nums = [0,1,2,4,5,6,7] might become:

# [4,5,6,7,0,1,2] if it was rotated 4 times.
# [0,1,2,4,5,6,7] if it was rotated 7 times.

# Notice that rotating an array [a[0], a[1], a[2], ..., a[n-1]] 1 time results in the array [a[n-1], a[0], a[1], a[2], ..., a[n-2]].

# Given the sorted rotated array nums of unique elements, return the minimum element of this array.

# You must write an algorithm that runs in O(log n) time.

# Example 1:

# Input: nums = [3,4,5,1,2]
# Output: 1
# Explanation: The original array was [1,2,3,4,5] rotated 3 times.



# solution ---> we are having two sorted array [3,4,5,1,2] one from 3 -> 5 and second from 1 -> 2
# if the middle index is 5 then we know we are searching to right of it (if left most index is < 5)

# and if reverse is the case we have to search to the left of it

class Solution:
    def findMin(self, nums: list[int]) -> int:
        start , end = 0, len(nums) - 1 
        curr_min = float("inf")
        
        while start  <  end :
            mid = (start + end ) // 2
            curr_min = min(curr_min,nums[mid])
            
            # right has the min 
            if nums[mid] > nums[end]:
                start = mid + 1
                
            # left has the  min 
            else:
                end = mid - 1 
                
        return min(curr_min,nums[start])


# Search in Rotated Sorted Array

# Question --> There is an integer array nums sorted in ascending order (with distinct values).

# Prior to being passed to your function, nums is possibly rotated at an unknown pivot index k (1 <= k < nums.length)
#  such that the resulting array is [nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]] (0-indexed). 
# For example, [0,1,2,4,5,6,7] might be rotated at pivot index 3 and become [4,5,6,7,0,1,2].
# Given the array nums after the possible rotation and an integer target, return the index of target if it is in nums, or -1 if it is not in nums.
# You must write an algorithm with O(log n) runtime complexity.

# Example 1:

# Input: nums = [4,5,6,7,0,1,2], target = 0
# Output: 4

class Solution:
    def search(self, nums: list[int], target: int) -> int:
        l, r = 0, len(nums) - 1

        while l <= r:
            mid = (l + r) // 2
            if target == nums[mid]:
                return mid

            # left sorted portion
            if nums[l] <= nums[mid]:
                if target > nums[mid] or target < nums[l]:
                    l = mid + 1
                else:
                    r = mid - 1
            # right sorted portion
            else:
                if target < nums[mid] or target > nums[r]:
                    r = mid - 1
                else:
                    l = mid + 1
        return -1



# Search in Rotated Sorted Array II

# Question --> There is an integer array nums sorted in non-decreasing order (not necessarily with distinct values).

# Before being passed to your function, nums is rotated at an unknown pivot index k (0 <= k < nums.length) 
# such that the resulting array is [nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]] (0-indexed). 
# For example, [0,1,2,4,4,4,5,6,6,7] might be rotated at pivot index 5 and become [4,5,6,6,7,0,1,2,4,4].

# Given the array nums after the rotation and an integer target, return true if target is in nums, or false if it is not in nums.
# You must decrease the overall operation steps as much as possible.

# Example 1:

# Input: nums = [2,5,6,0,0,1,2], target = 0
# Output: true


# Solution ---> we can have linear value in non decreasing array like  2 , 2, 3 , 4 , 1 , 2 , 2
# like we can have the same value on both side ( right & left)

class Solution:
    def search(self, nums: list[int], target: int) -> bool:
        low, high = 0, len(nums) - 1

        while low <= high:
            mid = (low + high) // 2

            # if the mid value is exactly equal to target return true
            if nums[mid] == target:
                return True
         
            # we want to keep adjusting till mid is no longer the same as left most pointer
            if nums[low] == nums[mid]:
                low += 1
                continue
            
            # left sorted non decreasing portion
            if nums[low] <= nums[mid]:
                if nums[low] <= target <= nums[mid]:
                    high = mid - 1
                else:
                    low = mid + 1
            
            # right sorted non decreasing portion
            else:
                if nums[mid] <= target <= nums[high]:
                    low = mid + 1
                else:
                    high = mid - 1
        
        # return false if no target found
        return False


# Time Based Key-Value Store

# Question ---> Design a time-based key-value data structure that can store multiple values for the same key at different time stamps and retrieve the key's value at a certain timestamp.

# Implement the TimeMap class:

# TimeMap() Initializes the object of the data structure.
# void set(String key, String value, int timestamp) Stores the key key with the value value at the given time timestamp.
# String get(String key, int timestamp) Returns a value such that set was called previously, with timestamp_prev <= timestamp. If there are multiple such values, it returns the value associated with the largest timestamp_prev.
# If there are no values, it returns "".

# Example 1:

# Input
# ["TimeMap", "set", "get", "get", "set", "get", "get"]
# [[], ["foo", "bar", 1], ["foo", 1], ["foo", 3], ["foo", "bar2", 4], ["foo", 4], ["foo", 5]]
# Output
# [null, null, "bar", "bar", null, "bar2", "bar2"]

# Explanation
# TimeMap timeMap = new TimeMap();
# timeMap.set("foo", "bar", 1);  // store the key "foo" and value "bar" along with timestamp = 1.
# timeMap.get("foo", 1);         // return "bar"
# timeMap.get("foo", 3);         // return "bar", since there is no value corresponding to foo at timestamp 3 and timestamp 2, then the only value is at timestamp 1 is "bar".
# timeMap.set("foo", "bar2", 4); // store the key "foo" and value "bar2" along with timestamp = 4.
# timeMap.get("foo", 4);         // return "bar2"
# timeMap.get("foo", 5);         // return "bar2"


class TimeMap:

    def __init__(self):
        # timeMap structure:
        #   Key - key passed by user
        #   Value - List of [value, timestamp] passed by user
        self.timeMap = {}

    def set(self, key: str, value: str, timestamp: int) -> None:
        if key not in self.timeMap: self.timeMap[key] = []
        self.timeMap[key].append([value, timestamp])

    def get(self, key: str, timestamp: int) -> str:
        # If key is not timeMap, we return empty string
        if key not in self.timeMap: return ""
        # Getting map for that key
        mapForKey = self.timeMap[key]
        # Since, timestamps are already in sorted order according to question,
        # if time stamp of first entry > given time stamp, we can't find 
        # a value with <= timestamp. So, we return ""
        if mapForKey[0][1] > timestamp: return ""
        # If time stamp of last entry <= given time stamp, that is the latest
        # value. So we return it
        if mapForKey[-1][1] <= timestamp: return mapForKey[-1][0]
        # Binary Search
        left = 0
        right = len(mapForKey) - 1
        while left <= right:
            mid = (left + right) // 2
            if mapForKey[mid][1] == timestamp: return mapForKey[mid][0]
            elif mapForKey[mid][1] > timestamp: right = mid - 1
            else: left = mid + 1
        return mapForKey[right][0]



# Find First and Last Position of Element in Sorted Array

# Question --> Given an array of integers nums sorted in non-decreasing order,
# find the starting and ending position of a given target value.
# If target is not found in the array, return [-1, -1].
# You must write an algorithm with O(log n) runtime complexity.

# Example 1:

# Input: nums = [5,7,7,8,8,10], target = 8
# Output: [3,4]

# Solution  ---> we want to run the binary search on the list till we get the extreme left target value
# and like wise on the right side as well till we get the extreme right target value
class Solution:
    def searchRange(self, nums: list[int], target: int) -> list[int]:
        
        def binary_search(nums, target, is_searching_left):
            left = 0
            right = len(nums) - 1
            idx = -1
            
            while left <= right:
                mid = (left + right) // 2
                
                if nums[mid] > target:
                    right = mid - 1
                elif nums[mid] < target:
                    left = mid + 1
                else:
                    idx = mid
                    if is_searching_left:
                        right = mid - 1
                    else:
                        left = mid + 1
            
            return idx
        
        left = binary_search(nums, target, True)
        right = binary_search(nums, target, False)
        
        return [left, right]