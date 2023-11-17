# Valid Parentheses

# Question --> Given a string s containing just the characters '(', ')', '{', '}', '[' and ']', determine if the input string is valid.

# An input string is valid if:

# Open brackets must be closed by the same type of brackets.
# Open brackets must be closed in the correct order.
# Every close bracket has a corresponding open bracket of the same type.

# Solution : we can only have as many as open bracket to begin with {{[()]}} we can't start with close )]}}{{[(
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
            
            # if it is +, D, or C
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
        # loop through everything apart from the last value
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

# Example 1:

# Input: s = "leet**cod*e"
# Output: "lecoe"
# Explanation: Performing the removals from left to right:
# - The closest character to the 1st star is 't' in "leet**cod*e". s becomes "lee*cod*e".
# - The closest character to the 2nd star is 'e' in "lee*cod*e". s becomes "lecod*e".
# - The closest character to the 3rd star is 'd' in "lecod*e". s becomes "lecoe".
# There are no more stars, so we return "lecoe".

# Solution --> a stack will be efficient for the solution since will be poping from left

class Solution:
    def removeStar(self, s: str) -> str:
        stack = []

        for c in s:
            # if the character is star we want to pop the last character from the stack
            if c == "*":
                stack and stack.pop()
            else:
                stack.append(c)

        return "".join(stack)

# Validate Stack Sequences

# Question --> Given two integer arrays pushed and popped each with distinct values, 
#     return true if this could have been the result of a sequence of push and pop operations on an initially empty stack, or false otherwise.

# Example 1:

# Input: pushed = [1,2,3,4,5], popped = [4,5,3,2,1]
# Output: true
# Explanation: We might do the following sequence:
# push(1), push(2), push(3), push(4),
# pop() -> 4,
# push(5),
# pop() -> 5, pop() -> 3, pop() -> 2, pop() -> 1


class Solution:
    def validateStackSequences(self, popped : list[int] , pushed : list[int]) -> bool:
        stack = []
        i = 0
        for n in pushed : 
            stack.append(n)
            #if popped at index of i is the same as the top of the stack we wan to pop it.
            #it is possible to pop consecutively so we will change if to **while** .
            #edge cases --> if by incrementing i it goes out of balance ( i < len(popped)).
            #               if stack is empty it will throw an error as well ( sta)
            if i < len(popped) and stack and stack[-1] == popped[i]:
                stack.pop()
                i += 1

        return not stack


# Asteroid Collision

# Question --> We are given an array asteroids of integers representing asteroids in a row.
# For each asteroid, the absolute value represents its size, and the sign represents its direction (positive meaning right, negative meaning left).
# Each asteroid moves at the same speed.

# Find out the state of the asteroids after all collisions. If two asteroids meet, the smaller one will explode.
# If both are the same size, both will explode. Two asteroids moving in the same direction will never meet.

# Example 1:

# Input: asteroids = [10,2,-5]
# Output: [10]
# Explanation: The 2 and -5 collide resulting in -5. The 10 and -5 collide resulting in 10.


# solution

class Solution:
    def asteroidCollision(self, asteroids: list[int]) -> list[int]:
        stack = []

        for a in asteroids:
            # while there's asteroid n stack and a < 0 ( meaning negative) and top of the stack > 0 ( meaning positive)
            while stack and a < 0 and stack[-1] > 0:
                # we want to know the difference between the top of the stack and the the asteroids
                diff = a + stack[-1]
                # if it's positive (it means the top of the stack win and we set asteroid to zero)
                if diff > 0:
                    a = 0
                # if it's negative (it means the asteroid win and we set the top of the stack to zero)
                elif diff < 0:
                    stack.pop()
                # if both are equal , we set it to zero (top of the stack and asteroids)
                else:
                    a = 0
                    stack.pop()
            # append the remaining a if there are remaining
            if a:
                stack.append(a)

        return stack

#  Daily Temperatures

# Question --> Given an array of integers temperatures represents the daily temperatures, 
# return an array answer such that answer[i] is the number of days you have to wait after the ith day to get a warmer temperature.
#  If there is no future day for which this is possible, keep answer[i] == 0 instead.


# Example 1:

# Input: temperatures = [73,74,75,71,69,72,76,73]
# Output: [1,1,4,2,1,1,0,0]

# Solution --> we will be getting the index and temperature

class Solution:
    def dailyTemperatures(self, temperatures: list[int]) -> list[int]:
        # initaialize all the array result with zero
        res = [0] * len(temperatures)

        stack = []  # pair: [temp, index]
        # enumerate through the temperatures to get both the index and temps
        for i, t in enumerate(temperatures):
            
            # as long a s there's value in stack and the top of the stack value is less than temps
            # we wan to pop it (monothonic decreasing stack)
            while stack and t > stack[-1][0]:

                # get the stackIndex and stackTemp pop
                stackT, stackInd = stack.pop()

                # current I - pop stackInd (the differences between them inicate the next greater temperatures)
                res[stackInd] = i - stackInd

            # append to the stack both temp and index
            stack.append((t, i))

            # return the results
        return res


#  Online Stock Span

#Question--> Design an algorithm that collects daily price quotes for some stock and returns the span of that stock's price for the current day.
# The span of the stock's price in one day is the maximum number of consecutive days (starting from that day and going backward) for which the stock price was less than or equal to the price of that day.

# For example, if the prices of the stock in the last four days is [7,2,1,2] and the price of the stock today is 2, then the span of today is 4 because starting from today, the price of the stock was less than or equal 2 for 4 consecutive days.
# Also, if the prices of the stock in the last four days is [7,34,1,2] and the price of the stock today is 8, then the span of today is 3 because starting from today, the price of the stock was less than or equal 8 for 3 consecutive days.


# Implement the StockSpanner class:

# StockSpanner() Initializes the object of the class.
# int next(int price) Returns the span of the stock's price given that today's price is price.


# Example 1:

# Input
# ["StockSpanner", "next", "next", "next", "next", "next", "next", "next"]
# [[], [100], [80], [60], [70], [60], [75], [85]]
# Output
# [null, 1, 1, 1, 2, 1, 4, 6]

# Explanation
# StockSpanner stockSpanner = new StockSpanner();
# stockSpanner.next(100); // return 1
# stockSpanner.next(80);  // return 1
# stockSpanner.next(60);  // return 1
# stockSpanner.next(70);  // return 2
# stockSpanner.next(60);  // return 1
# stockSpanner.next(75);  // return 4, because the last 4 prices (including today's price of 75) were less than or equal to today's price.
# stockSpanner.next(85);  // return 6

# solution --> 
#  we will be using monotic decreasing stack for efficiency
# for stack created we want to have (price and span) and the span will be recorded and only pop
# when the span is greater than the previous one

class StockSpanner:
    def __init__(self):
        self.stack = []  # pair: (price, span)

    def next(self, price: int) -> int:

        # by default every span will be 1
        span = 1

        # we only want to make it bigger if theres stack and the current price is less than the top of the stack
        while self.stack and self.stack[-1][0] <= price:

            # add to span the already span of top of the stack
            span += self.stack[-1][1]

            # and pop from the stack or until the price is no longer greater than the top of stack (which makes the loop invalid)
            self.stack.pop()

        # update the stack with price and span
        self.stack.append((price, span))
        return span


# Car Fleet

# Question --> There are n cars going to the same destination along a one-lane road. The destination is target miles away.

# You are given two integer array position and speed, both of length n, where position[i] is the position of the ith car and speed[i] is the speed of the ith car (in miles per hour).
# A car can never pass another car ahead of it, but it can catch up to it and drive bumper to bumper at the same speed. The faster car will slow down to match the slower car's speed.
#  The distance between these two cars is ignored (i.e., they are assumed to have the same position).

# A car fleet is some non-empty set of cars driving at the same position and same speed. Note that a single car is also a car fleet.

# If a car catches up to a car fleet right at the destination point, it will still be considered as one car fleet.

# Return the number of car fleets that will arrive at the destination.


# Input: target = 12, position = [10,8,0,5,3], speed = [2,4,1,1,3]
# Output: 3
# Explanation:
# The cars starting at 10 (speed 2) and 8 (speed 4) become a fleet, meeting each other at 12.
# The car starting at 0 does not catch up to any other car, so it is a fleet by itself.
# The cars starting at 5 (speed 1) and 3 (speed 3) become a fleet, meeting each other at 6. The fleet moves at speed 1 until it reaches target.
# Note that no other cars meet these fleets before the destination, so the answer is 3.


# Solution --> let merge the 2 array together [(position , speed)] and we have a single target
# [(10 , 2),(8 , 4),(0 , 1),(5 , 1),(3 , 3)] they are just a system of linear equation and plot of graph

# position
#    |
# 10 |
#  9 |
#  8 |
#  7 |
#  6 |
#  5 |
#  4 |
#  3 |
#  2 |
#  1 |
#    |_______________________________________>   Time

# (3 , 3) means for every one minute of time it reaches 3 point ahead starting at position 3.
# 6 position in 1 minutes
# 9 position in 2 minutes
# 12 position in 3 minutes

# i j k ------------------> destination

# if j reaches the destination point before k that means there must be a car fleet

# time = destination - k / speed

# remember we will be iterating through in reverse order(rigth to left)
# length of the stack is going to tell us the car fleet


class Solution:
    def carFleet(self, target: int, position: list[int], speed: list[int]) -> int:
        # zip the position and speed in array 
        pair = [(p, s) for p, s in zip(position, speed)]

        # sort and reverse the array of position and speed
        pair.sort(reverse=True)

        stack = []
        for p, s in pair:  # Reverse Sorted Order

            # append the time it tooks to reach the target point into the stack
            stack.append((target - p) / s)

            # if the length of the stack is more than two and the the top of the stack is less than the previous time(meaning there's a collision)
            if len(stack) >= 2 and stack[-1] <= stack[-2]:
                stack.pop()

        return len(stack)


# Simplify Path

# Question --->  Given a string path, which is an absolute path (starting with a slash '/') to a file or directory in a Unix-style file system, convert it to the simplified canonical path.
# In a Unix-style file system, a period '.' refers to the current directory, a double period '..' refers to the directory up a level, and any multiple consecutive slashes (i.e. '//') are treated as a single slash '/'.
#  For this problem, any other format of periods such as '...' are treated as file/directory names.

# The canonical path should have the following format:

# The path starts with a single slash '/'.
# Any two directories are separated by a single slash '/'.
# The path does not end with a trailing '/'.
# The path only contains the directories on the path from the root directory to the target file or directory (i.e., no period '.' or double period '..')
# Return the simplified canonical path.

# Example 1:

# Input: path = "/home/"
# Output: "/home"
# Explanation: Note that there is no trailing slash after the last directory name.

# Solution --> because of the .. we will be using stack and carry out each operation on the stack


class Solution:
    def simplifyPath(self, path: str) -> str:

        stack = []

        # looping through the path and having it seperated by the /
        for i in path.split("/"):
            #  if i == "/" or i == '//', it becomes '' (empty string)

            # if i == .. and there's value in there that whan we want to pop
            if i == "..":
                if stack:
                    stack.pop()


            # continue the loop if it . or nothing in i
            elif i == "." or i == '':
                # skip "." or an empty string
                continue

            else:
                stack.append(i)
                
        # start the result with / and join value in stack with /
        res = "/" + "/".join(stack)
        return res
    


#  Decode String

# Question --> Given an encoded string, return its decoded string.

# The encoding rule is: k[encoded_string], where the encoded_string inside the square brackets is being repeated exactly k times. 
# Note that k is guaranteed to be a positive integer.

# You may assume that the input string is always valid; there are no extra white spaces, square brackets are well-formed, etc. 
# Furthermore, you may assume that the original data does not contain any digits and that digits are only for those repeat numbers, k.
# For example, there will not be input like 3a or 2[4].

# The test cases are generated so that the length of the output will never exceed 105

# Example 2:

# Input: s = "3[a2[c]]"
# Output: "accaccacc"


# Example 3:

# Input: s = "2[abc]3[cd]ef"
# Output: "abcabccdcdcdef"

# Solution --> converting this 3[a2[c]] -------> accaccacc  require you to solve inner 
# problem first i.e we want to approach it by looping through the character and appending to stack until we see 
# a ] --- that when we start popping from stack until we see a [
#we still want to pop the [ and it is guaranted there will always be a number at the begining of the [
# and multiply the number with the character

class Solution:
    def decodeString(self, s: str) -> str:
        stack = []

        # looping through s
        for char in s:

            # add every character except ]
            if char is not "]":
                stack.append(char)

            # "else" is a case where there is every character apart from ]
            else:
                sub_str = ""

                # while the top of the stack is not [
                while stack[-1] is not "[":

                    # add from the back the pop character
                    sub_str = stack.pop() + sub_str

                # and we want to pop the [ itself
                stack.pop()

                multiplier = ""

                # while there's character in stack and the top of the stack is digit
                while stack and stack[-1].isdigit():

                    # add from the back the digit character
                    multiplier = stack.pop() + multiplier

                # append to the stack the multiplication of int digit and sub_str
                stack.append(int(multiplier) * sub_str)


        return "".join(stack)


# Remove All Adjacent Duplicates in String II

# Question --->  You are given a string s and an integer k, a k duplicate removal consists of choosing k adjacent and equal letters from s and removing them, 
# causing the left and the right side of the deleted substring to concatenate together.
# We repeatedly make k duplicate removals on s until we no longer can.
# Return the final string after all such duplicate removals have been made. It is guaranteed that the answer is unique.

# Example 1:

# Input: s = "abcd", k = 2
# Output: "abcd"
# Explanation: There's nothing to delete.

# Solution --> we will be removing k consecutive element in the string the ideal way is to use a stack
# we will be looping through the string and adding to stack [char | count]
# whenever the count of the top of the stack equivalent to k we will pop it and return the left char or empty string if nothing remain

class Solution:
    def removeDuplicates(self, s: str, k: int) -> str:
        # create a stack with char and count
        stack = []  # [char, count]

        # looping through the strings
        for c in s:

            # if there's value in stack and the top of the stack character == to the current character
            if stack and stack[-1][0] == c:

                # we want to increement the stack count by 1
                stack[-1][1] += 1

            # if there's no stack yet or the current character doesn't match the top of the stack
            else:

                # append to it the character with the count of 1
                stack.append([c, 1])

            # if the top of stack ever reches k that when we want to pop from the stack
            if stack[-1][1] == k:
                stack.pop()

        # we want to convert the rest of the char to string and return as result 
        res = ""
        for char, count in stack:
            res += char * count

        return res


# Remove K Digits

# Question --> Given string num representing a non-negative integer num, and an integer k, 
# return the smallest possible integer after removing k digits from num.

# Example 1:

# Input: num = "1432219", k = 3
# Output: "1219"
# Explanation: Remove the three digits 4, 3, and 2 to form the new number 1219 which is the smallest.


# Solution --> i) we want to probably remove the larger digit and ii) we're biased towards left i.e ( removing the larger digit )
# iii) if we're in decresing order 54321 we want to remove from the left iii) if we're in increasing order 12345 we want to remove the larger value from the right
# MONOTONIC INCREASING ORDER algorithm 
# we want to let the stack be in an increasing order

class Solution:
    def removeKdigits(self , num: str , k: int) -> str:
        # create an empty stack 
        stack = []

        for c in num:
            # as long as we still have k left and there's value in our stack and top of the stack is greater than the current char
            # we want to pop it
            while k > 0 and stack and stack[-1] > c:
                # decreement the k
                k -= 1
                # pop from the stack
                stack.pop()
            stack.append(c)

            # what happens if there's still k left which means they are all in increasing order but left with "k value"
            # then we want to pop from the rigth as condition iii)

        # slice out the length of the k from stack
        stack = stack[:len(stack) - k]

        # convert the stack to string before returning it
        res = "".join(stack)

        # if there's result we to remove the leading zero if they are present by converting it to int first and string later
        # and "0" if the stack is empty
        return str(int(res)) if res else "0"



# 132 Pattern

# Question --> Given an array of n integers nums, a 132 pattern is a subsequence of three integers nums[i], 
# nums[j] and nums[k] such that i < j < k and nums[i] < nums[k] < nums[j].
# Return true if there is a 132 pattern in nums, otherwise, return false.

# Example 1:

# Input: nums = [1,2,3,4]
# Output: false
# Explanation: There is no 132 pattern in the sequence.

# Solution --> to look for 132 pattern efficiently we're going to be using stack
# monothonic decreasing stack 
# stack [3,1,4,2]
# we're going to be taking minimum of every single value in stack before poping
# the minimum value will be equivalent to 1
# the begining of the stack will be equivalent to 3 because it is bigger
# and the top of the stack will be equivalent to 2 because it's smaller than 3 and bigger than 1


class Solution:
    def find132pattern(self, nums: list[int]) -> bool:
        stack = [] # pair [num, curLeftMin], mono-decreasing stack

        # assign the current minimum to nums at index 0 or -infinity
        curMin = nums[0]

        for n in nums:

            # as long as there's stack and n is greater than top of stack we want to pop
            # so as to keep our stack in decreasing order
            while stack and n >= stack[-1][0]:
                stack.pop()


            #     assume top of the stack = 3
            #     assume n = 2
            #     assume currentMinimum = 1

            # as long as there's value in stack and n is less than top of the stack and
            # n is greater than currentMinimum we want to return true
            if stack and n < stack[-1][0] and n > stack[-1][1]:
                return True

            # append to n and current minimum to stack in pairs
            stack.append([n, curMin]) 
            curMin = min(n, curMin)

        return False

# Maximum Frequency Stack

# Question--> Design a stack-like data structure to push elements to the stack 
# and pop the most frequent element from the stack.
# Implement the FreqStack class:

# FreqStack() constructs an empty frequency stack.
# void push(int val) pushes an integer val onto the top of the stack.
# int pop() removes and returns the most frequent element in the stack.
# If there is a tie for the most frequent element, the element closest to the stack's top is removed and returned.


# Example 1:

# Input
# ["FreqStack", "push", "push", "push", "push", "push", "push", "pop", "pop", "pop", "pop"]
# [[], [5], [7], [5], [7], [4], [5], [], [], [], []]
# Output
# [null, null, null, null, null, null, null, 5, 7, 5, 4]


# Explanation
# FreqStack freqStack = new FreqStack();
# freqStack.push(5); // The stack is [5]
# freqStack.push(7); // The stack is [5,7]
# freqStack.push(5); // The stack is [5,7,5]
# freqStack.push(7); // The stack is [5,7,5,7]
# freqStack.push(4); // The stack is [5,7,5,7,4]
# freqStack.push(5); // The stack is [5,7,5,7,4,5]
# freqStack.pop();   // return 5, as 5 is the most frequent. The stack becomes [5,7,5,7,4].
# freqStack.pop();   // return 7, as 5 and 7 is the most frequent, but 7 is closest to the top. The stack becomes [5,7,5,4].
# freqStack.pop();   // return 5, as 5 is the most frequent. The stack becomes [5,7,4].
# freqStack.pop();   // return 4, as 4, 5 and 7 is the most frequent, but 4 is closest to the top. The stack becomes [5,7].


# Solution --->  we are going to be having an hashmap to count the frequent value
#  and maxVal variable to keep track of the frequency
#   1    2   1   2   1   3    ----> groups of frequency   
# [5,4],[5],[3],[4],[2],[5]  we are going to be mapping count with values
# i.e sepparating based on the frequency

# Count | Group
# 1      [5 , 4 , 3 , 2 ] 

# 2      [5 , 4 , 3]

# 3      [5]  


# if we want to add a 5 to the stack we will be having another group 4 -- > [5]
# we pop from the most frequent and last value added to stack

class FreqStack:

    def __init__(self):
        self.cnt = {}
        self.maxCnt = 0
        self.stack = {}
        

    def push(self, val: int) -> None:
        valCnt = 1 + self.cnt.get(val , 0)

        # if the valueCount is greater than the maximum frequency it means we will have a new count and stack
        if valCnt > self.maxCnt:

            # update the new value count to the maximum frequency
            self.maxCnt = valCnt

            # add the value count with the array
            self.stack[valCnt] = []

        # append the value now the to the stack created 
        self.stack[valCnt].append(val)

    def pop(self) -> int:
        # pop from the most frequent stack
        res = self.stack[self.maxCnt].pop()

        # reduce the count of the res from stack
        self.cnt[res] -= 1

        # if after poppping from the stack it happens to be empty then we want to reduce the count of the maxCount
        if not self.stack[self.maxCnt]:
            self.maxCnt -= 1
        return res
        


# class FreqStack:

#     def __init__(self):
#         self.freq = defaultdict(int)
#         self.groups = defaultdict(list)
#         self.maxFreq = 0
        

#     def push(self, val: int) -> None:
#         self.freq[val] += 1

#         if self.freq[val] > self.maxFreq: self.maxFreq = self.freq[val]

#         self.groups[self.freq[val]].append(val)

#     def pop(self) -> int:
#         first = self.groups[self.maxFreq].pop()
#         self.freq[first] -= 1

#         if not self.groups[self.maxFreq]:
#             self.maxFreq -= 1

#         return first


# Largest Rectangle in Histogram

# Question ---> Given an array of integers heights representing the histogram's bar height where the width of each bar is 1, 
# return the area of the largest rectangle in the histogram.

# Example 1:

# Input: heights = [2,1,5,6,2,3]
# Output: 10
# Explanation: The above is a histogram where width of each bar is 1.
# The largest rectangle is shown in the red area, which has an area = 10 units




# Solution --> we want to keep the stack in increaing order (because we can create more rectangle for smaller value on the right by doing something so)
# before popping we want to know how many area they covered by multiplying there height * (index - starting point) and keeping track of the maximum value
# and after looping through the height the remaining value in stack , the area will be calculated as well to know the maximum value

class Solution:
    def largestRectangleArea(self, heights: list[int]) -> int:
        maxArea = 0
        stack = []  # pair: (index, height)

        for i, h in enumerate(heights):
            start = i

            # as long as there's value in the stack and the top value is grater than current h we want to pop the heigher value
            while stack and stack[-1][1] > h:

                # get the index and height before popping to calculate the maxValue
                index, height = stack.pop()

                # areaa = height * distance between the current index from begining
                maxArea = max(maxArea, height * (i - index))
                start = index

            # then append the index and height to the stack
            stack.append((start, h))


        # leftover after popping all
        for i, h in stack:
            # calculate the maximum area as well
            maxArea = max(maxArea, h * (len(heights) - i))
        return maxArea
