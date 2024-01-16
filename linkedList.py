# Reverse Linked List

# Question --> Given the head of a singly linked list, reverse the list, and return the reversed list.

# Example 1
# Input: head = [1,2,3,4,5]
# Output: [5,4,3,2,1]

# Solution --> we want to swing the current with previous value and previous with current value 

class Solution:
    def reverseList(self, head: ListNode) -> ListNode:
        prev, curr = None, head

        while curr:
            temp = curr.next
            curr.next = prev
            prev = curr
            curr = temp
        return prev
    

# Merge Two Sorted Lists

# Question ---> You are given the heads of two sorted linked lists list1 and list2.
# Merge the two lists into one sorted list. The list should be made by splicing together the nodes of the first two lists.

# Return the head of the merged linked list.

# Example 1 :

# Input: list1 = [1,2,4], list2 = [1,3,4]
# Output: [1,1,2,3,4,4]

# Solution ---> we want to compare the listNode and assign into output the smaller value of both list


class Solution:
    def mergeTwoLists(self, list1: ListNode, list2: ListNode) -> ListNode:
        dummy = node = ListNode()

        # as long as there's value in both list1 and list2
        while list1 and list2:
            if list1.val < list2.val:
                
                # append list one value to the node if it happens to be smaller one
                node.next = list1
                # assign the next value to be list1
                list1 = list1.next
            else:
                node.next = list2
                list2 = list2.next

            # adjust the tail/node regardless of which value we move to the output
            node = node.next

        # join the rest of the remaining value in the remaining list
        node.next = list1 or list2

        return dummy.next


# Palindrome Linked List

# Question --> Given the head of a singly linked list, return true if it is a palindrome or false otherwise.
# Example 1:

# Input: head = [1,2,2,1]
# Output: true

# Solution --> we want to loop through all linked list and get the middle of the list with slow pointer
# and reverse the second half of it ... [1 --> 2 --> 2 --> 1] = [1 --> 2 <-- 2 <-- 1]
# and later check the palindrome list

class Solution:
    def isPalindrome(self, head: ListNode) -> bool:
        fast = head
        slow = head
        
        # find the middle (slow)
        while fast and fast.next:
            fast = fast.next.next
            slow = slow.next
            
        # reverse second half
        prev = None
        while slow:
            tmp = slow.next
            slow.next = prev
            prev = slow
            slow = tmp
        
        # check palindrome
        left, right = head, prev
        while right:
            if left.val != right.val:
                return False
            left = left.next
            right = right.next
        return True
    

# Remove Linked List Elements

# Question --> Given the head of a linked list and an integer val, 
# remove all the nodes of the linked list that has Node.val == val, and return the new head.

# Example 1:

# Input: head = [1,2,6,3,4,5,6], val = 6
# Output: [1,2,3,4,5]


# Solution -->  we are using dummyNode to avoid any bug that may occur and to return the node we want to return dummy.next 
# which is equal to head

class Solution:
    def removeElements(self, head: ListNode, val: int) -> ListNode:
        dummy = ListNode(next=head)
        prev, curr = dummy, head
        
        while curr:
            nxt = curr.next
            
            if curr.val == val:
                prev.next = nxt
            else:
                prev = curr
            
            curr = nxt
        return dummy.next


# Remove Duplicates from Sorted List

# Question ---> Given the head of a sorted linked list, delete all duplicates such that each element appears only once. 
# Return the linked list sorted as well.

# Example 1:
# Input: head = [1,1,2]
# Output: [1,2]

class Solution:
    def deleteDuplicates(self, head: Optional[ListNode]) -> Optional[ListNode]:
        cur = head

        # as long as there's value we want to keep looping
        while cur:
            
            # while there's value and previous value and next value to previous are the same
            # we want to remove
            while cur.next and cur.next.val == cur.val:
                cur.next = cur.next.next

            # else update the pointer
            cur = cur.next
            # and return the head
        return head
    

  # Middle of the Linked List

#  Question --> Given the head of a singly linked list, return the middle node of the linked list.
# If there are two middle nodes, return the second middle node.

# Example 1:

# Input: head = [1,2,3,4,5]
# Output: [3,4,5]
# Explanation: The middle node of the list is node 3.

# Solution --> we will be using 2 pointer approach of (fast & slow pointer)
# and return the slow pointer when the first iteration is reach ( the end)

class Solution:
    def middleNode(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if not head or not head.next:
            return head
        

        # they both begins at the head
        slow = fast = head

        # while there's value for fast pointer and slow pointer to reach
        while fast and fast.next:
            
            slow, fast = slow.next, fast.next.next

        return slow

# Intersection of Two Linked Lists
# Question --> Given the heads of two singly linked-lists headA and headB, return the node at which the two lists intersect.
# If the two linked lists have no intersection at all, return null.

# For example, the following two linked lists begin to intersect at node c1:

# The test cases are generated such that there are no cycles anywhere in the entire linked structure.

# Note that the linked lists must retain their original structure after the function returns.

# Custom Judge:

# The inputs to the judge are given as follows (your program is not given these inputs):

# intersectVal - The value of the node where the intersection occurs. This is 0 if there is no intersected node.
# listA - The first linked list.
# listB - The second linked list.
# skipA - The number of nodes to skip ahead in listA (starting from the head) to get to the intersected node.
# skipB - The number of nodes to skip ahead in listB (starting from the head) to get to the intersected node.
# The judge will then create the linked structure based on these inputs and pass the two heads, headA and headB to your program. 
# If you correctly return the intersected node, then your solution will be accepted.


# Solution -->  we are looking for intersection and to know that there will be 
# pointer starting from both head of the list , if they're not equal that when we set the
# other pointer that is shorter length to the second one

class Solution:
    def getIntersectionNode(
        self, headA: ListNode, headB: ListNode
    ) -> Optional[ListNode]:

        # set the headA and headB pointer
        l1, l2 = headA, headB

        # as long as they are not equal we need to keep looping 
         
        while l1 != l2:
            # if there's node  add l1 else set l1 to l2 (head)
            l1 = l1.next if l1 else headB
            # if there's node add l2 else set l2 to l1 (head)
            l2 = l2.next if l2 else headA
        return l1

  
# Reorder List

# Question --> You are given the head of a singly linked-list. The list can be represented as:

# L0 → L1 → … → Ln - 1 → Ln

# Reorder the list to be on the following form:

# L0 → Ln → L1 → Ln - 1 → L2 → Ln - 2 → …
# You may not modify the values in the list's nodes. Only nodes themselves may be changed.


# Example 1:

# Input: head = [1,2,3,4]
# Output: [1,4,2,3]

# Solution --> find the middle of the list and reverse the second & 
# alternate the first list and the second list (reverse order)

class Solution:
    def reorderList(self, head: ListNode) -> None:
        # find middle
        slow, fast = head, head.next
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next

        # reverse second half
        second = slow.next
        prev = slow.next = None
        while second:
            tmp = second.next
            second.next = prev
            prev = second
            second = tmp

        # merge two halfs
        first, second = head, prev
        while second:
            tmp1, tmp2 = first.next, second.next
            first.next = second
            second.next = tmp1
            first, second = tmp1, tmp2


# Maximum Twin Sum of a Linked List

# Question ---> In a linked list of size n, where n is even, the ith node (0-indexed) of the linked list is known as the twin of the (n-1-i)th node, if 0 <= i <= (n / 2) - 1.

# For example, if n = 4, then node 0 is the twin of node 3, and node 1 is the twin of node 2. These are the only nodes with twins for n = 4.
# The twin sum is defined as the sum of a node and its twin.

# Given the head of a linked list with even length, return the maximum twin sum of the linked list.

# Example 1:

# Input: head = [5,4,2,1]
# Output: 6
# Explanation:
# Nodes 0 and 1 are the twins of nodes 3 and 2, respectively. All have twin sum = 6.
# There are no other nodes with twins in the linked list.
# Thus, the maximum twin sum of the linked list is 6. 

# Solution --> we want to keep looping using fast & slow pointer and  [5-> 4 -> 2 -> 1] == [5 <- 4 <- 2 -> 1]
# after breaking and reversing the link start your comparison according to the 

class Solution:
    def pairSum(self, head: Optional[ListNode]) -> int:
        slow, fast = head, head
        prev = None

        # loop and reverse the links of the first half
        while fast and fast.next:
            fast = fast.next.next
            tmp = slow.next
            slow.next = prev
            prev = slow
            slow = tmp

        res = 0
        while slow:
            # check the max
            res = max(res, prev.val + slow.val)
            # update the pointer
            prev = prev.next
            slow = slow.next
        return res

# remove-nth-node-from-end-of-list

# Question -->  Given the head of a linked list,
#  remove the nth node from the end of the list and return its head.


# Example 1:


# Input: head = [1,2,3,4,5], n = 2
# Output: [1,2,3,5]

# Solution --> to remove the n node from the end we will be using to pointer L(left) & R(right)
# with the interval of n in between ... so by the time Right is at null Left pointer will
# be exactly at preceeding Left of it (because left will be starting at dummy node)

class Solution:
    def removeNthFromEnd(self, head: ListNode, n: int) -> ListNode:
        # dummy node to avoid edge case
        dummy = ListNode(0, head)
        # left pointer starting at dummy node
        left = dummy
        # right pointer starting at head 
        right = head

        # as long as n > 0 (to keep right pointer at the interval of n)
        while n > 0:
            right = right.next
            n -= 1

        # while there's right keep increementing both pointers
        while right:
            left = left.next
            right = right.next

        # delete ( re assign the pointer to next.next)
        left.next = left.next.next
        return dummy.next


# Swapping Nodes in a Linked List


# Question ---> You are given the head of a linked list, and an integer k.
# Return the head of the linked list after swapping the values of the kth node from the beginning 
# and the kth node from the end (the list is 1-indexed).

# Example 1:

# Input: head = [1,2,3,4,5], k = 2
        #    |   ||    
# Output: [1,   4,3,2,5]

# Solution --> we want to maintain interval between right and left pointer and make clone of the right pointer 
# when the clone right pointer hit the ends of the list the left pointer will be exactly at the interval of k from end needed

class Solution:
    def swapNodes(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        right_pointer = head
        for _ in range(1, k):
            right_pointer = right_pointer.next
        left_kth_node = right_pointer

        left_pointer = head
        while right_pointer is not None:
            right_kth_node = left_pointer
            right_pointer = right_pointer.next
            left_pointer = left_pointer.next

        left_kth_node.val, right_kth_node.val = right_kth_node.val, left_kth_node.val
        return head


# Copy List with Random Pointer

# Questions --> A linked list of length n is given such that each node contains an additional random pointer, which could point to any node in the list, or null.

# Construct a deep copy of the list. The deep copy should consist of exactly n brand new nodes, where each new node has its value set to the value of its corresponding original node. 
# Both the next and random pointer of the new nodes should point to new nodes in the copied list such that the pointers in the original list and copied list represent the same list state.
# None of the pointers in the new list should point to nodes in the original list.


# For example, if there are two nodes X and Y in the original list, where X.random --> Y, then for the corresponding two nodes x and y in the copied list, x.random --> y.

# Return the head of the copied linked list.

# The linked list is represented in the input/output as a list of n nodes. Each node is represented as a pair of [val, random_index] where:

# val: an integer representing Node.val
# random_index: the index of the node (range from 0 to n-1) that the random pointer points to, or null if it does not point to any node.
# Your code will only be given the head of the original linked list.


# Example 1:

# Input: head = [[7,null],[13,0],[11,4],[10,2],[1,0]]
# Output: [[7,null],[13,0],[11,4],[10,2],[1,0]]

class Solution:
    def copyRandomList(self, head: "Node") -> "Node":
        # assign None should incase the loop reach pointer
        oldToCopy = {None: None}

        cur = head
        # first loop to make copy of the node
        while cur:
            copy = Node(cur.val)
            oldToCopy[cur] = copy
            cur = cur.next
        cur = head

        # second loop to assign the pointer to each pointer
        while cur:
            copy = oldToCopy[cur]
            copy.next = oldToCopy[cur.next]
            copy.random = oldToCopy[cur.random]
            cur = cur.next
        return oldToCopy[head]

# Design Linked List

# Question ---> Design your implementation of the linked list. You can choose to use a singly or doubly linked list.
# A node in a singly linked list should have two attributes: val and next. val is the value of the current node, and next is a pointer/reference to the next node.
# If you want to use the doubly linked list, you will need one more attribute prev to indicate the previous node in the linked list. Assume all nodes in the linked list are 0-indexed.