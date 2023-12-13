"""
Let's talk about sorting algorithms.
Sorts basically do 'what it says on the box' - they arrange data in a specified sorted order.
There are many different sorting algorithms, with different strengths, efficiencies, and use cases.
"""


def selection_sort(array: list):
    """
    This is an implementation of a selection sort.
    Selection sort iterates over each element in an array, finds the minimum element in the unsorted part of the array,
    and then swaps it with the iterated element.
    Complexity: O(n^2), where n is the array length. (The quadratic complexity is because of the two nested for loops.)
    """
    array_length = len(array)
    for i in range(array_length):  # For each index value in the array
        min_idx = i  # Initialize the minimum index value as the iterated element

        for j in range(i+1, array_length):  # For each element in the unsorted portion of the array
            if array[j] < array[min_idx]:  # If the iterated element is less than the current value of min_idx
                min_idx = j  # Assign the lower value to min_idx

        array[i], array[min_idx] = array[min_idx], array[i]  # Swap the new minimum element with the element at index i

    return array  # Return the sorted array


def insertion_sort(array: list):
    """
    This is an implementation of an insertion sort.
    Insertion sort iteratively inserts each element into its correct position in the sorted part of the array.
    Complexity: O(n^2), where n is the array length. (Quadratic complexity comes from the nested for and while loops.)
    """
    array_length = len(array)
    for i in range(1, array_length):
        key = array[i]  # Initialize a 'key' value as the iterated element, which will be 'inserted' into sorted order

        j = i-1  # Initialize a value j representing the last element of the sorted section of the array
        while j >= 0 and key < array[j]:  # Boundary check and key comparison
            array[j + 1] = array[j]  # Move elements that are greater than the key one position 'ahead' in the array
            j -= 1  # Assess the next element

        array[j + 1] = key  # Insert the key at the appropriate position

    return array  # Return the sorted array


def bubble_sort(array: list):
    """
    This is an implementation of a bubble sort.
    Bubble sort iterates through an array, compares adjacent elements, and swaps them if they are in the wrong order.
    This process is repeated until the list is fully sorted.
    Complexity: O(n^2), where n is the array length. (The quadratic complexity is because of the two nested for loops.)
    """
    array_length = len(array)

    for i in range(array_length):  # For each element index in the array

        for j in range(0, array_length-i-1):  # The last 'i' elements are already sorted, so we avoid them
            if array[j] > array[j+1]:  # If the iterated element is greater than the next element
                array[j], array[j+1] = array[j+1], array[j]  # Swap those two elements

    return array  # Return the sorted array


def merge_sort(array: list):
    """
    This is an implementation of a merge sort, which requires a recursive solution.
    A merge sort divides the input array into two halves, calls itself recursively for those halves, and then merges
    the two sorted halves. Note that this sort is input order-agnostic.
    Complexity: O(n log n), where n is the array length.
    """
    def merge(left, right):
        """
        This 'helper' function merges two sorted sub-arrays into a single sorted array.
        """
        merged = []  # Initialize an empty list
        i = j = 0  # Initialize index positions to zero

        # Iterate through both arrays and append the smallest of both elements to 'merged'
        while i < len(left) and j < len(right):
            if left[i] < right[j]:
                merged.append(left[i])
                i += 1

            else:
                merged.append(right[j])
                j += 1

        # When we run out of elements in the sub-arrays, add any remaining elements to 'merged'
        merged.extend(left[i:])
        merged.extend(right[j:])

        return merged

    # Base Case
    # If the array is of length 1 or less, it's already fully sorted
    if len(array) <= 1:
        return array

    # Recursive Case
    # Begin by dividing the array elements into 2 halves
    midpoint = len(array) // 2

    # Make the recursive calls
    left = merge_sort(array[:midpoint])
    right = merge_sort(array[midpoint:])

    # Merge the sorted sub-arrays by calling the 'merge' helper function and return the sorted list
    return merge(left, right)


def quick_sort(array: list):
    """
    This is an implementation of a quick sort.
    Quick sort works by selecting a 'pivot' element from the array and partitioning the other elements into
    two sub-arrays, according to whether they are less than or greater than the pivot.
    The sub-arrays are then sorted recursively. This sort therefore requires a recursive solution.
    This implementation uses random pivot selection. This method helps improve performance over the traditional choice
    of selecting the last element as the pivot value, especially in worst-case scenarios for the sort.
    Complexity: O(n log n) on average, O(n^2) in the worst case.
    """
    import random  # This library is required for random pivot selection

    def partition(low, high):
        """
        This 'helper' function partitions the array into two parts around a randomly chosen pivot.
        """
        # Randomly select a pivot and swap it with the last element
        pivot_index = random.randint(low, high)  # Choose a random index value from within the appropriate range
        array[pivot_index], array[high] = array[high], array[pivot_index]
        pivot = array[high]

        i = low - 1  # Initialize a pointer to track the position where the next smaller element should be placed

        # Iterate over the array and compare all elements with the pivot
        for j in range(low, high):
            if array[j] <= pivot:  # If current element is less than or equal to the pivot value
                i += 1  # Increment the index of the smaller element
                array[i], array[j] = array[j], array[i]  # And perform the swap using multiple assignment

        # Place the pivot in the correct position by swapping it with the element at index i+1
        array[i + 1], array[high] = array[high], array[i + 1]  # Perform the swap using multiple assignment
        return i + 1  # And return the partition index

    def quick_sort_main(low, high):
        """
        The main function that implements the quick sort by recursively sorting the partitions
        """
        if low < high:
            part_idx = partition(low, high)  # This is the index of the partition value

            # Recursive calls: sort elements before and after the partition value
            quick_sort_main(low, part_idx - 1)
            quick_sort_main(part_idx + 1, high)

    quick_sort_main(0, len(array) - 1)  # Initiate the quick sort
    return array  # Finally, return the sorted array


def heap_sort(array: list):
    """
    This is an implementation of a heap sort.
    This function first builds a max-heap and then repeatedly extracts the maximum element from the heap.
    It then rebuilds the heap until the entire array is sorted.
    Complexity: O(n log n), where n is the array length.
    """

    def heapify(heap_size, root_index):
        """
        Transforms a subtree rooted with a node at root_index into a max heap
        Note that heap_size is simply the number of elements in the heap
        """
        largest = root_index  # Initialize the largest value as the root index
        left_child = 2 * root_index + 1  # Create the left 'child' node
        right_child = 2 * root_index + 2  # Create the right 'child' node

        # Check if the left child exists and is greater than the root value
        if left_child < heap_size and array[left_child] > array[largest]:
            largest = left_child  # If so, reassign the largest value

        # Check if the right child exists and is greater than the largest value assessed thus far
        if right_child < heap_size and array[right_child] > array[largest]:
            largest = right_child  # If so, reassign the largest value

        if largest != root_index:  # If the largest value is not the root value,
            array[root_index], array[largest] = array[largest], array[root_index]  # Swap it with the root value
            heapify(heap_size, largest)  # And continue calling the function recursively

    array_length = len(array)

    # Build a max-heap iteratively
    for i in range(array_length // 2 - 1, -1, -1):  # Iterate over the *non-leaf* nodes in reverse order
        heapify(array_length, i)  # Ensure that the subtree rooted at index 'i' satisfies the max heap property

    # Extract elements one by one from the max-heap
    for i in range(array_length - 1, 0, -1):  # Iterate backwards through the array
        # Swap the maximum of the heap (at index 0) with the last element of the unsorted part of the array (at index i)
        array[i], array[0] = array[0], array[i]

        heapify(i, 0)  # Restore the max-heap property in the reduced heap

    return array  # Finally, return the sorted array


def tim_sort(array: list):
    """
    This is an implementation of a more advanced sort, the TimSort algorithm.
    This is the default algorithm behind Python's native .sort() method.
    A TimSort is a hybrid sort derived from combining the merge sort and insertion sort algorithms. 
    It is designed to perform well on many kinds of real-world data. It divides the array into segments known as 'runs'
    and first sorts these runs using insertion sort, then merges the runs using a merge sort.
    Complexity: O(n log n) where n is the array length.
    """
    min_run_size = 32  # Defining the minimum size of the 'runs' - 32 is a well-tested by-convention value

    def tim_insertion_sort(start, end):
        """
        Sorts the elements between the start and end values (exclusive) using an insertion sort
        """
        for i in range(start + 1, end):  # Iterate from the second element to the end of the sub-array
            element = array[i]  # Set element to be repositioned equal to the element at the iterated index
            j = i - 1  # Find the position where the element should be inserted
            while j >= start and element < array[j]:  # Move elements greater than the iterated 'element'
                array[j + 1] = array[j]  # By shifting those elements to the right
                j -= 1  # Assess the next element
            array[j + 1] = element  # And place the iterated element in its correct sorted position

    def tim_merge(left, right):
        """
        Merges two sorted lists into a single sorted list
        """
        if not left or not right:  # If either of the two lists is empty
            return left or right  # Return the other list

        result = []  # Initialize the merged list
        i = j = 0  # Initialize pointers for the left and right lists, both equal to zero

        # Iterate until the merged list contains all the elements from both lists
        while len(result) < len(left) + len(right):
            # Append smaller elements to the result list
            if left[i] < right[j]:
                result.append(left[i])
                i += 1

            else:
                result.append(right[j])
                j += 1

            # Append the remaining elements if the end of one of the lists has been reached
            if i == len(left) or j == len(right):
                result.extend(left[i:] or right[j:])
                break  # Exit the loop if the condition is satisfied
        return result

    # Step 1 of the TimSort: Sort the individual 'runs' of size 'min_run_size' using an insertion sort
    array_length = len(array)
    for start in range(0, array_length, min_run_size):  # Iterate over the array in steps of size 'min_run_size'
        end = min(start + min_run_size, array_length)  # Define the end of each iterated 'run'
        tim_insertion_sort(start, end)  # Perform the insertion sort on each run

    # Step 2 of the TimSort: Merge the sorted 'runs' iteratively, doubling the size in each iteration
    size = min_run_size
    while size < array_length:  # Loop as long as the size of the run is less than the total array length
        for left in range(0, array_length, size * 2):  # Iterate over array in steps of 'size * 2'
            middle = min(array_length, left + size)  # Calculate the middle index value
            right = min(array_length, left + size * 2)  # Calculate the end index value
            if middle < right:  # If we can legitimately merge the two segments
                merged_list = tim_merge(array[left:middle], array[middle:right])  # Merge the two halves
                array[left:left + len(merged_list)] = merged_list  # Update the array with the newly-merged list
        size *= 2  # And double the value of 'size' for the next iteration

    return array  # Finally, return the sorted array


def radix_sort(array: list):
    """
    This is an implementation of a Radix sort that sorts integers.
    Radix sorts are non-comparative sorts insofar as they don't directly compare elements with each other. Instead, they
    process each 'digit' of the elements to be sorted, from the least significant to the most significant digit.
    Radix sort is very efficient for sorting large integers or strings of characters where the length of the keys is
    relatively small compared to the range of the key values.
    Example: sorting an unsorted array of 1,000,000 eight-digit social security numbers.
    Complexity: O(n*k) where n is array length (number of elements) and k is the number of digits in the maximum value.
    """
    # The first step is to define a 'counting sort' subroutine which will serve to sort the individual digits
    def counting_sort(count_array, exp):
        """
        A subroutine to perform counting sort on an array according to the digit represented by exp (exponent).
        """
        n = len(count_array)
        output = [0] * n  # Initialize the output array
        count = [0] * 10  # Count array for storing the count of occurrences of each possible digit

        # Store the count of occurrences of each digit in the count array
        for i in range(n):  # For each index value in the array
            index = (count_array[i] // exp) % 10  # Calculate the digit at each position of each element in the array
            count[index] += 1  # Update the count array

        # Change count[i] so that count[i] contains actual position of that digit in the output array
        for i in range(1, 10):  # For each digit from 1 to 9
            count[i] += count[i - 1]  # Update value at count[i] to accumulate the count

        # Build the output array - i.e. perform the actual sort
        i = n - 1  # Initialize i to the last index of the array
        while i >= 0:  # Iterate backward through the array, starting from the last element, as long as values remain
            index = (count_array[i] // exp) % 10  # For each element, isolate the digit at the current sorting position
            output[count[index] - 1] = count_array[i]  # Place each element into the correct position in output array
            count[index] -= 1  # Then decrement the count at count[index]
            i -= 1  # And decrement i

        # Copy the output array back to count_array
        for i in range(n):
            count_array[i] = output[i]

    # Find the maximum number to know the number of digits
    max_num = max(array)

    # Run counting sort for every digit. Instead of passing the digit number, pass 'exp.'
    # Note that exp is 10^i where i is current digit number
    exp = 1
    while max_num // exp > 0:
        counting_sort(array, exp)
        exp *= 10
