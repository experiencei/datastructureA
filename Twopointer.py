#  Valid Palindrome

# Question -->  A phrase is a palindrome if, after converting all uppercase letters into lowercase letters and removing all non-alphanumeric characters, it reads the same forward and backward. 
# Alphanumeric characters include letters and numbers.
# Given a string s, return true if it is a palindrome, or false otherwise.

# Example 1:

# Input: s = "A man, a plan, a canal: Panama"
# Output: true
# Explanation: "amanaplanacanalpanama" is a palindrome.

# Solution --> 
class Solution:
    def isPalindrome(self, s: str) -> bool:
        l, r = 0, len(s) - 1
        while l < r:
            while l < r and not self.alphanum(s[l]):
                l += 1
            while l < r and not self.alphanum(s[r]):
                r -= 1
            if s[l].lower() != s[r].lower():
                return False
            l += 1
            r -= 1
        return True

    # Could write own alpha-numeric function
    def alphanum(self, c):
        return (
            ord("A") <= ord(c) <= ord("Z")
            or ord("a") <= ord(c) <= ord("z")
            or ord("0") <= ord(c) <= ord("9")
        )

# Valid Palindrome II

# Question --> 
#     Given a string s, return true if the s can be palindrome after deleting at most one character from it.

# Example 2:

# Input: s = "abca"
# Output: true
# Explanation: You could delete the character 'c'.

class Solution:
    def validPalindrome(self, s: str) -> bool:
        
        # if empty return True
        if not s:
            return True
        
        start = 0
        end = len(s)-1
        
        #starts scanning from both ends inwards break at unmatched characters
        while start <= end and s[start]==s[end]:
            start += 1
            end -= 1
        
        #if no unmatched characters found
        if end <= start:
            return True
        
        # function to check if string is palindrome
        def isPalindrome(start,end):
            while start <= end:
                if s[start] != s[end]:
                    return False
                start += 1
                end   -=1
            return True
        
        #deleting either characters can result in palindrome hence checking both
        if isPalindrome(start+1,end) or isPalindrome(start,end-1):
            return True
        
        return False 