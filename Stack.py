# Valid Parentheses

# Question --> Given a string s containing just the characters '(', ')', '{', '}', '[' and ']', determine if the input string is valid.

# An input string is valid if:

# Open brackets must be closed by the same type of brackets.
# Open brackets must be closed in the correct order.
# Every close bracket has a corresponding open bracket of the same type.

# Solution : we can only have as many as open bracket to begin with {{[()]}} we can start with close )]}}{{[(
#  we will be having a hashmap to check wether the closing match the key to know the opening before popping from stack
# Map = {")": "(", "]": "[", "}": "{"} to check the key to the closing bracket in stack

class Solution:
    def isValid(self, s: str) -> bool:
        #hashmap to check for key of parentheses
        Map = {")": "(", "]": "[", "}": "{"}
        stack = []

        for c in s:
            #means if the parentheses is not a key(closing bracket) we want to add to stack
            if c not in Map:
                #adding it to stack
                stack.append(c)
                continue
            if the top of the stack is not equal to closing bracket then return false or nothing in stack
            if not stack or stack[-1] != Map[c]:
                return False
            stack.pop()

        return not stack