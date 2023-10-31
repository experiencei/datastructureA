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


# Implement Stack using Queues
# Question --> Implement a last-in-first-out (LIFO) stack using only two queues. The implemented stack should support all the functions of a normal stack (push, top, pop, and empty).

# Implement the MyStack class:

# void push(int x) Pushes element x to the top of the stack.
# int pop() Removes the element on the top of the stack and returns it.
# int top() Returns the element on the top of the stack.
# boolean empty() Returns true if the stack is empty, false otherwise.


# Example 1:

# Input
# ["MyStack", "push", "push", "top", "pop", "empty"]
# [[], [1], [2], [], [], []]
# Output
# [null, null, null, 2, 2, false]

# Explanation
# MyStack myStack = new MyStack();
# myStack.push(1);
# myStack.push(2);
# myStack.top(); // return 2
# myStack.pop(); // return 2
# myStack.empty(); // return False


# Solution --> we want use queue to build stack, with queue we can only pop from the left(FIFO) unlike stacks(LIFO)
#  so to pop we need to loop through every value apart from the last value in the queue and add it back in the queue

class MyStack:
    def __init__(self):
        self.q = deque()

    def push(self, x: int) -> None:
        self.q.append(x)

    def pop(self) -> int:
        # loop through everything apart from the last value
        for i in range(len(self.q) - 1):
            # add everything in the queue again by push the popleft value
            self.push(self.q.popleft())
        return self.q.popleft()

    def top(self) -> int:
        for i in range(len(self.q) - 1):
            self.push(self.q.popleft())
        res = self.q[0]
        self.push(self.q.popleft())
        return res

    def empty(self) -> bool:
        return len(self.q) == 0


# Min Stack
# Question --> Design a stack that supports push, pop, top, and retrieving the minimum element in constant time.

# Implement the MinStack class:

# MinStack() initializes the stack object.
# void push(int val) pushes the element val onto the stack.
# void pop() removes the element on the top of the stack.
# int top() gets the top element of the stack.
# int getMin() retrieves the minimum element in the stack.
# You must implement a solution with O(1) time complexity for each function.


# Example 1:

# Input
# ["MinStack","push","push","push","getMin","pop","top","getMin"]
# [[],[-2],[0],[-3],[],[],[],[]]

# Output
# [null,null,null,null,-3,null,0,-2]

# Explanation
# MinStack minStack = new MinStack();
# minStack.push(-2);
# minStack.push(0);
# minStack.push(-3);
# minStack.getMin(); // return -3
# minStack.pop();
# minStack.top();    // return 0
# minStack.getMin(); // return -2


# Solution --> Consider each node in the stack having a minimum value.  then we will be using 2 stacks
         # one for minstack and the other for stack itself
class MinStack:
    def __init__(self):
        self.stack = []
        self.minStack = []

    def push(self, val: int) -> None:
        self.stack.append(val)
        # take the top of the minstack if present and else minimum of val and new val
        val = min(val, self.minStack[-1] if self.minStack else val)
        self.minStack.append(val)

    def pop(self) -> None:
        self.stack.pop()
        self.minStack.pop()

    def top(self) -> int:
        return self.stack[-1]

    def getMin(self) -> int:
        return self.minStack[-1]


# Evaluate Reverse Polish Notation

# Question --> You are given an array of strings tokens that represents an arithmetic expression in a Reverse Polish Notation.
# Evaluate the expression. Return an integer that represents the value of the expression.

# Note that:

# The valid operators are '+', '-', '*', and '/'.
# Each operand may be an integer or another expression.
# The division between two integers always truncates toward zero.
# There will not be any division by zero.
# The input represents a valid arithmetic expression in a reverse polish notation.
# The answer and all the intermediate calculations can be represented in a 32-bit integer.

# Example 1:


# Input: tokens = ["2","1","+","3","*"]
# Output: 9
# Explanation: ((2 + 1) * 3) = 9


# Solution --> 
class Solution:
    def evalRPN(self, tokens: list[str]) -> int:
        stack = []
        for c in tokens:
            if c == "+":
                stack.append(stack.pop() + stack.pop())
            elif c == "-":
                a, b = stack.pop(), stack.pop()
                # subtracting the second value in stack from the first
                stack.append(b - a)
            elif c == "*":
                stack.append(stack.pop() * stack.pop())
            elif c == "/":
                a, b = stack.pop(), stack.pop()
                # dividing the second value in stack from the first and to interger division instead of decimal
                stack.append(int(float(b) / a))
            else:
                # convert to interger and append to stack if it is numbers
                stack.append(int(c))
        return stack[0]


# Generate Parentheses
# Question --> Given n pairs of parentheses, write a function to generate all combinations of well-formed parentheses.

# Example 1:

# Input: n = 3
# Output: ["((()))","(()())","(())()","()(())","()()()"]

# Solution --> we are going to be using backtracking and the condition which needed 
# to be hold 
#only add open parenthesis if open < n
#only add a closing parenthesis if close < open
#valid IIF open == closed == n , then we will stop adding 
class Solution:
    def generateParenthesis(self, n: int) -> list[str]:
        # stack which is going to hold the parenthesis
        stack = []
        res = []

        def backtrack(openN, closedN):
             # this is the base case 
            if openN == closedN == n:
                # we want to join everything in the stack and append it to result
                res.append("".join(stack))
                return

            if openN < n:
                # if openStack is less than n we want to add open parentheses to the stack
                stack.append("(")
                # we want to increase our open count by 1 and closed count remain the same
                backtrack(openN + 1, closedN)
                # we have to pop the single character we have from the stack after backtracking
                stack.pop()

            if closedN < openN:
                # if closeStack is less than openStack we want to add close parentheses to the stack
                stack.append(")")
                # we want to increase our closed count by 1 and open count remain the same
                backtrack(openN, closedN + 1)
                # cleanup as well
                stack.pop()
                
        # call the backtrack function and start with initial zero
        backtrack(0, 0)
        return res

# Removing Stars From a String
# Question --> You are given a string s, which contains stars *.

# In one operation, you can:

# Choose a star in s.
# Remove the closest non-star character to its left, as well as remove the star itself.
# Return the string after all stars have been removed.

# Note:
# The input will be generated such that the operation is always possible.
# It can be shown that the resulting string will always be unique.