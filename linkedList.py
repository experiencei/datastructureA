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
  
