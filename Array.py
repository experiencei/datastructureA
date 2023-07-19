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


class Solution:
    def isAnagram(self, s: str, t: str) -> bool:
        
        
        if len(s) != len(t):
            return False

        countS , countT = {} , {}

        for i in range(len(s)):
            countS[s[i]] = 1 + countS.get(s[i] ,0)
            countT[s[i]] = 1 + countT.get(t[i] ,0)
        return countS == countT