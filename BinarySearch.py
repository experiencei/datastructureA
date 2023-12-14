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

Example 1:

Input: matrix = [[1,3,5,7],[10,11,16,20],[23,30,34,60]], target = 3
Output: true