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