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