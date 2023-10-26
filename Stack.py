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
            # if the top of the stack is not equal to closing bracket then return false or nothing in stack
            if not stack or stack[-1] != Map[c]:
                return False
            # else we want to pop if reverse is the case
            stack.pop()
        # return True if stack is empty
        return not stack



# Baseball Game

# Question --> You are keeping the scores for a baseball game with strange rules. At the beginning of the game, you start with an empty record.

# You are given a list of strings operations, where operations[i] is the ith operation you must apply to the record and is one of the following:

# An integer x.
# Record a new score of x.
# '+'.
# Record a new score that is the sum of the previous two scores.
# 'D'.
# Record a new score that is the double of the previous score.
# 'C'.
# Invalidate the previous score, removing it from the record.
# Return the sum of all the scores on the record after applying all the operations.

# The test cases are generated such that the answer and all intermediate calculations fit in a 32-bit integer and that all operations are valid.

# Example 1:

# Input: ops = ["5","2","C","D","+"]
# Output: 30
# Explanation:
# "5" - Add 5 to the record, record is now [5].
# "2" - Add 2 to the record, record is now [5, 2].
# "C" - Invalidate and remove the previous score, record is now [5].
# "D" - Add 2 * 5 = 10 to the record, record is now [5, 10].
# "+" - Add 5 + 10 = 15 to the record, record is now [5, 10, 15].
# The total sum is 5 + 10 + 15 = 30.

class Solution:
    def calPoints(self, operations: list[str]) -> int:
        
        score_stack = []
        
        for o in operations:
            
            # it is +, D, or C
            # if stack isn't of sufficient length, then operation is voided
            if o == "+" and len(score_stack) >= 2:
                summed = score_stack[-2] + score_stack[-1]
                score_stack.append(summed)
                
            elif o == "D" and len(score_stack) >= 1:
                doubled = score_stack[-1] * 2
                score_stack.append(doubled)
                
            elif o == "C" and len(score_stack) >= 1:
                score_stack.pop() 
                
            else: 
                score_stack.append(int(o))

        return sum(score_stack)


Implement Stack using Queues