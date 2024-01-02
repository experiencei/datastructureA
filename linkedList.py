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