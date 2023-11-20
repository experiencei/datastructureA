# Binary Search

# Question ---> Given an array of integers nums which is sorted in ascending order, and an integer target, 
# write a function to search target in nums. If target exists, then return its index. Otherwise, return -1.
# You must write an algorithm with O(log n) runtime complexity.


# Example 1:

# Input: nums = [-1,0,3,5,9,12], target = 9
# Output: 4
# Explanation: 9 exists in nums and its index is 4

solution --> we're going to be using divide and conquer method
[-1,0,3,5,9,12], since they are already in sorted order
if the target 9 is greater than the middle of the array (then we eliminate the other half of the array)