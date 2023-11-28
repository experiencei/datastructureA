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


