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
