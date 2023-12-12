"""
Algorithmic paradigms may be understood as the core conceptualizations or 'methods' according to which algorithms
are implemented to solve problems. They represent a variety of computational 'approaches' and each have strengths,
weaknesses, and problem archetypes to which they are well-suited.

In this file, I will discuss the fundamentals of each 'core' algorithmic paradigm and provide an example implementation.
"""

"""
PARADIGM 1: INCREMENTAL
The incremental paradigm refers to an approach in which a solution to a problem is built up incrementally.

To implement an incremental approach, you take a problem, solve for some piece of it in whichever manner you choose,
and 'extend' that solution to solve the next piece of the problem. You repeat the process until the problem is solved.

This approach of course requires an iterable solution that can be universally applied to all pieces of a problem.
You use this broadly-applicable solve process on one piece of the problem and then 'increment' the solution one piece
at a time.

Incremental approaches are usually simple both to understand and to implement. Any fundamentally simple problem can
often be solved with an incremental approach.

Let's look at two simple implementations of the incremental paradigm: adding up a list of numbers and finding the 
longest word in a list of words.
"""


def add_integers(integer_list: list):
    """
    This function takes a list of integers as input and uses an incremental approach to return their sum.
    The incremental solution is that each number is individually added to a running sum which is re-computed at each
    step. When no more numbers remain to be added to the running sum, the final sum can be returned.
    """
    running_sum = 0  # We begin by initializing the running sum to zero

    # A for loop will address each value in the list in turn, and terminate when the end of the list has been reached
    for value in integer_list:  # Now, for each value in the list of integers
        running_sum += value  # We use addition assignment to increment the value of the running sum

    return running_sum  # We return the final value of the running sum, which is the sum of all integers in the list


int_list = [1, 2, 3, 4, 5]  # The sum of this list of integers is 15
print(add_integers(int_list))  # The function's incremental approach returns the correct sum


def longest_word(word_list: list):
    """
    This function takes a list of strings as input and returns the longest string and its character count.
    The incremental solution is that each word is separately assessed and compared to a 'current' maximum.
    Each time the algorithm encounters a word longer than the current maximum, the maximum is changed appropriately.
    """
    largest_char_count = 0  # We initialize the longest word character count to zero
    longest_current_word = ''  # We initialize the text of the longest word to an empty string

    # Next, we use a for loop to assess each word in the list of words
    for word in word_list:  # For each word in the list
        if len(word) > largest_char_count:  # If the length of the iterated string is larger than the current maximum
            largest_char_count = len(word)  # Update the value of the maximum character count
            longest_current_word = word  # And update the value of the longest current word

    return longest_current_word, largest_char_count  # We return the longest word in the list and its character count


string_list = ['Bob', 'Cedar', 'Hippopotamus', 'Child', 'Grapefruit']  # The longest word is Hippopotamus - 12 letters
print(longest_word(string_list))  # The function returns a tuple containing the correct word and its character count


"""
PARADIGM 2: DIVIDE AND CONQUER
The divide and conquer paradigm (DAC) refers to an approach in which a problem is broken down into smaller sub-problems,
solutions to the sub-problems are computed, and the component solutions are recombined to form an overall solution to
the original large-scale problem.

Divide and conquer approaches are frequently implemented using recursion. This is because recursion provides a natural 
and straightforward way to handle the 'breaking down and recombining' processes. Recursion inherently supports the idea 
of solving a given large problem by solving smaller instances/segments of the same problem.

As a result, many divide and conquer approaches have a first step of finding a recursive solution to the problem.

Common use cases for a DAC approach are sorting, (the merge sort and quick sort algorithms are common and powerful,) 
multiplying large numbers together, or solving the 'closest pair of points' problem.

We'll look at two DAC implementations: we'll sum a list of integers (as we did above) and provide code for a merge sort.
"""


def add_integers_dac(integer_list: list):
    """
    This function takes a list of integers as input and uses a recursive DAC approach to calculate their sum.
    """
    # We start with the first recursive 'base case'
    if len(integer_list) == 0:  # If the list of integers is empty
        return 0  # Return zero

    # Then, we move to the second recursive base case
    elif len(integer_list) == 1:  # If the list of integers has only one value
        return integer_list[0]  # Return that value, at index position 1

    # Now, we implement our recursive case
    else:
        midpoint = len(integer_list) // 2  # Use floored division to calculate a midpoint of the list

        # This is the 'divide' portion of the algorithm, where we recursively sum each half of the list
        left_sum = add_integers_dac(integer_list[:midpoint])  # Recursive call on the left 'half' of the list
        right_sum = add_integers_dac(integer_list[midpoint:])  # Recursive call on the right 'half' of the list

        # This is the 'conquer' portion of the algorithm, where we combine the two sub-solutions
        return left_sum + right_sum  # Combine the two component sums and return the overall sum


int_list = [1, 2, 3, 4, 5]  # The sum of this list of integers is 15
print(add_integers_dac(int_list))  # The function's DAC approach returns the correct sum


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


unsorted_array = [1, 7, 3, 19, 8, 7, 9, 2, 12, 11]  # An unsorted array of integers
print(merge_sort(unsorted_array))  # The array has been sorted in the correct order


"""
PARADIGM 3: DECREASE AND CONQUER
Decrease and conquer is an algorithmic paradigm which involves reducing a problem to a smaller instance of the same 
problem, solving the smaller instance, and then extending that solution to the original problem. 

Unlike divide and conquer, which breaks the problem into multiple sub-problems, decrease and conquer focuses on 
a *single* sub-problem at each step. However, similarly to divide and conquer, decrease and conquer approaches are often 
(though not always) implemented using recursion.

Common use cases for a decrease and conquer approach are binary searches, computing powers of a number, and graph-search
algorithms, such as a breadth-first search.

The implementation of decrease and conquer that I'll show here is the binary search, which finds an element in 
a *sorted* list by reducing the size of the problem by half at each step, focusing *only* on the segment where the 
element could potentially be found. This process continues until the element is found or the list segment becomes empty.
Note that this implementation of a binary search is *iterative* rather than recursive.
"""


def binary_search(sorted_list: list, target_value):
    """
    This function performs an iterative binary search to find a target value in an already-sorted list.
    It takes a list of sorted elements and the target element as parameters and returns the index of the
    target element (if it can be found) or an error message if the target element cannot be found.
    """

    # We begin by initialize the start and end indices
    start_index = 0
    end_index = len(sorted_list) - 1

    # We continue to search as long as the start index is less than or equal to the end index
    while start_index <= end_index:
        middle_index = (start_index + end_index) // 2  # Calculate the middle index

        if sorted_list[middle_index] == target_value:  # If the middle element *is* the target
            return middle_index  # The target value has been found, and we can return its index

        elif sorted_list[middle_index] > target_value:  # If the target is smaller than the middle element
            end_index = middle_index - 1  # We can ignore the right half of the list

        else:  # Otherwise, if the target is larger than the middle element
            start_index = middle_index + 1  # We can ignore the left half of the list

    # Target not found in the list
    else:
        return 'Target value not found in list'


sorted_integers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
print(binary_search(sorted_integers, 6))  # Returns 5, because the number 6 is at index position 5 in the sorted list
print(binary_search(sorted_integers, 50))  # Returns the error message because the number 50 is not in the list


"""
PARADIGM 4: RANDOMIZATION
The randomization paradigm involves incorporating randomness into the logic of an algorithm to improve performance, 
either in terms of speed, simplicity, or both. 

Unlike the deterministic algorithms shown above, which follow a strict set of rules and produce the same output for 
a given input every time they're run, randomized algorithms can exhibit different behaviors on different runs, 
even if provided with the same input.

Randomized algorithms are particularly useful in situations where a deterministic approach would be too slow, 
too complex, or computationally not feasible. They often offer simpler solutions and can average better performance 
over multiple runs, although they might not guarantee the best possible solution in every single execution. 

Example use cases for the randomization paradigm include algorithms for sorting, searching, optimization, and 
computational geometry, as well as some problems in cryptography and machine learning.

The implementation I'll show is a sorting algorithm called quicksort, which randomly chooses a 'pivot' point in an
unsorted array and sorts the sub-arrays on either side of the pivot.
"""


def quick_sort(array: list):
    """
    Quick sort works by selecting a 'pivot' element from the array and partitioning the other elements into
    two sub-arrays, according to whether they are less than or greater than the pivot. The sub-arrays are then
    sorted recursively.

    This implementation uses random pivot selection. This improves performance over the traditional choice of
    selecting the last element as the pivot value, especially in worst-case scenarios for the sort.
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


unsorted_array = [1, 7, 3, 19, 8, 7, 9, 2, 12, 11]  # An unsorted array of integers
print(quick_sort(unsorted_array))  # The array has been sorted in the correct order


"""
PARADIGM 5: BACKTRACKING
The backtracking paradigm is used to find solutions to problems incrementally (one step at a time) by removing 
any solutions which fail to satisfy the problem's constraints at any point. 

Fundamentally, it is a use of recursion which explores all potential solutions in a highly-structured way.

The key design principle is that we build a solution 'piece-by-piece,' and backtrack as soon as we asses that the 
current path does not lead to a valid solution. 

This paradigm is particularly useful in solving combinatorial problems like puzzles and games (like Sudoku or chess)
where all possible configurations cannot be practically tested. 

Backtracking ensures that *only potential solutions* are explored and eliminates them as soon as a constraint violation 
is found. This significantly reduces both search space and computation time.

Note that in certain kinds of problems, such as those that can be solved with DAC, a backtracking approach can be 
very inefficient. Backtracking approaches should only be applied to appropriate problems such as puzzle/game solving,
certain graph-based problems, constraint-satisfaction problems (such as scheduling or resource allocation,) or
pathfinding problems.

Because backtracking implementations are often complex and hard to follow, I'll solve a conceptually simple problem 
using backtracking: finding all possible permutations of a string.    
"""


def string_permutations(string: str):
    """
    This algorithm uses backtracking to produce all possible permutations of a given string.
    It takes a string to 'permutate' as input and uses recursion to return all possible permutations.
    """

    def permute(prefix, remaining_chars):
        """
        Recursively builds permutations by choosing each character as a 'prefix' then permuting all other characters.
        There's no return statement because the print statement in the base case prints all possible permutations.
        """
        # Base case:
        if len(remaining_chars) == 0:  # If no characters are remaining
            print(prefix)  # Print the current prefix

        # Recursive case:
        # Iterate over each remaining character, build a new prefix by adding the character, and create a
        # new remaining string *without* using that character
        else:
            for i in range(len(remaining_chars)):  # Iterate over each character
                new_prefix = prefix + remaining_chars[i]  # Add the iterated character to the prefix
                # print(new_prefix)  # Print debugging - this helps the recursion make more sense

                # Remove the iterated character from the remaining string
                new_remaining = remaining_chars[:i] + remaining_chars[i+1:]
                # print(new_remaining)  # Print debugging - this helps the recursion make more sense

                # Recursive calls using the new prefixes and new remaining strings
                # This step is the 'backtrack' piece in that we're exploring all potential paths
                permute(new_prefix, new_remaining)

    # Start the permutation process using an empty prefix and the full string as the remaining characters
    permute('', string)


# Now we'll call the function on a sample string
string_permutations('abc')  # Note that there are n! permutations of a string - this returns six possible permutations


"""
PARADIGM 6: DYNAMIC PROGRAMMING
Dynamic Programming (DP) is a method for solving complex problems by breaking them down into simpler sub-problems. 

It's most applicable to problems exhibiting two main characteristics: optimal substructure and overlapping sub-problems.

A problem has optimal substructure if an optimal solution to the problem contains optimal solutions to its sub-problems. 
This means that a problem can be solved optimally by solving its sub-problems optimally.

A problem has overlapping sub-problems if a recursive algorithm solves the same sub-problem repeatedly. 
In such cases, the problem is said to contain 'repeated work.' DP allows us to avoid 'repeated work' because partial 
solutions can be stored in memory rather than being repeatedly recomputed.

As a result, DP approaches, in some situations, can be much more efficient than DAC approaches.

An important note: there are two foundational DP approaches.

One approach involves writing a recursive algorithm and using a data structure (e.g. a dictionary or array) to store 
and retrieve the solutions to the sub-problems. This is a recursive 'top down' approach that is considered by some
people to not be true dynamic programming, but rather a DAC approach that makes use of memory to store solutions.
As a result, this approach is often called 'Memoized DAC.'

The other approach is the bottom-up 'tabulation' approach which involves finding and storing optimal solutions to 
the simplest sub-problems first, and subsequently building up solutions to more complex sub-problems.

Dynamic Programming is widely used to solve problems like the shortest path in a graph, the 'knapsack problem,' 
Fibonacci number series, and any other problem where a 'naive' recursive approach performs repeated work.

To demonstrate both the top-down and bottom-up approaches, we can implement algorithms to compute Fibonacci numbers.
"""


def fibonacci_dp_top_down(target_fib: int, memo=None):
    """
    This function computes a target Fibonacci number using top-down memoized DAC.
    It uses a dictionary to store previously-computed numbers to avoid redundant calculations during recursion.
    """

    # Initialize the memoized dictionary during the first function call
    if memo is None:  # As per the function definition
        memo = {}  # Initialize an empty dictionary

    # Base cases:
    # The first and second Fibonacci numbers are both 1
    if target_fib == 1 or target_fib == 2:  # So, if the target Fibonacci number is 1 or 2
        return 1  # Return 1

    # Memoization:
    # Check if the Fibonacci number for the target position has already been computed
    if target_fib not in memo:
        # If not already stored, calculate it by recursively calling the function for the two preceding numbers
        # Store the result in the memoized dictionary
        memo[target_fib] = fibonacci_dp_top_down(target_fib - 1, memo) + fibonacci_dp_top_down(target_fib - 2, memo)

    # Return the computed Fibonacci number for the target position
    return memo[target_fib]


print(fibonacci_dp_top_down(10))  # Outputs the 10th Fibonacci number, which is 55


def fibonacci_dp_bottom_up(target_fib: int):
    """
    This function computes a target Fibonacci number using bottom-up dynamic programming.
    This method iteratively builds up the solution *from the base cases,* storing each computed number in an array.
    """

    # Base cases:
    # The first and second Fibonacci numbers are both 1
    # If the target number is 1 or 2, return 1 as the Fibonacci number
    if target_fib == 1 or target_fib == 2:  # So, if the target Fibonacci number is 1 or 2
        return 1  # Return 1

    # Initialize an array to store Fibonacci numbers up to the target value
    # We need 'target + 1' spaces, because the list is zero-indexed
    fib_array = [0] * (target_fib + 1)
    fib_array[1], fib_array[2] = 1, 1  # Set the first two Fibonacci numbers using multiple assignment

    # Iteratively compute each Fibonacci number from 3 (we already have the first two) to the target
    for i in range(3, target_fib + 1):  # For each integer in the appropriate range
        # The 'ith' Fibonacci number is the sum of the two preceding Fibonacci numbers
        fib_array[i] = fib_array[i - 1] + fib_array[i - 2]

    # Return the target Fibonacci number
    return fib_array[target_fib]


print(fibonacci_dp_bottom_up(10))  # Outputs the 10th Fibonacci number, which is 55


"""
PARADIGM 7: GREEDY
The greedy paradigm is a method for solving problems by making a series of choices, each of which is the most optimal 
*at that moment,* without regard for future consequences. 

This approach assumes that by choosing a local optimum at each step, a global optimum will be reached.

Greedy algorithms are used when a problem can be broken down into stages such that a decision is required at each stage. 
At each stage, the algorithm selects the option that looks the best at that moment, hence the term "greedy". 

This approach is easy to conceptualize and can be fast and efficient. However, greedy algorithms do not always provide 
the best solution for all problems. They are most effective when they can guarantee an optimal solution, which 
requires a problem which satisfies the 'greedy choice property,' which is the condition that making a *locally* optimal 
choice at every step will lead to a globally optimal solution. This property is rarely satisfied in real-world 
applications. 

It is also necessary for the problem to exhibit optimal substructure for a greedy approach to be valid.

While applications of the greedy paradigm are limited compared to other algorithmic designs, common applications of 
greedy algorithms include finding the minimum spanning tree in a graph (e.g. Prim's algorithms), the shortest path in 
a graph (e.g. Dijkstra's algorithm), and in real-world scenarios like coin change problems or scheduling problems.

Below is an implementation of a simple solution to the 'n-coins' problem, where we make up a given amount of change 
using the smallest total number of coins possible. More advanced implementations are possible, including versions
using DP-compliant optimal and traceback tables.
"""


def make_change(target_amount: float, coin_denoms: list):
    """
    This function takes a target amount of money and a list of coin denominations as parameters.
    It uses a greedy approach to compute the way to make up the target amount of change using the fewest possible coins.
    We assume all values are in cents to avoid floating-point issues.
    """

    # Convert target amount to cents
    target_amount_cents = int(round(target_amount * 100))

    # Convert coin denominations to cents
    coin_denoms_cents = [int(coin * 100) for coin in coin_denoms]  # List comprehension

    # Sort the coin denominations in reverse order
    coin_denoms_cents.sort(reverse=True)

    coins_used = []  # Initialize an empty list of the coins used

    for coin in coin_denoms_cents:  # For each coin in the cent-converted coins available
        while target_amount_cents >= coin:  # While the target amount is greater than the iterated denomination
            target_amount_cents -= coin  # Reduce the remaining amount by the iterated coin's value
            coins_used.append(coin / 100)  # Convert back to the original units and add to the list of coins used

    return coins_used  # Return the list of coins used


US_coins = [0.01, 0.05, 0.10, 0.25]  # These are the modern US coins expressed in cents
print(make_change(1.44, US_coins))  # Change used to make up $1.44: 5 quarters, 1 dime, 1 nickel, and 4 pennies


"""
PARADIGM 8: BRANCH AND BOUND
The branch and bound paradigm is a method used to solve optimization problems. 
It provides a systematic way of considering various candidate solutions in order to find the optimal solution.

Conceptually, it works as follows:
Branching: Dividing the problem into smaller sub-problems (branches) that are easier to solve. 
The process is recursive and continues until the sub-problems become simple enough to solve directly.

Bounding: For each sub-problem, an upper or lower bound on the 'objective function' is calculated. 
This helps to estimate the best possible solution that can be obtained from a given sub-problem. 
If a sub-problem's bound is worse than the best already-known solution, that sub-problem can be discarded.

Various search strategies like depth-first, breadth-first, or best-first can be used to traverse a tree of sub-problems. 
This traversal is crucial in determining the efficiency of the algorithm.
 
Note that depth-first, breadth-first, and best-first searches can be implemented almost interchangeably simply by 
changing the core data structure being used by the search. Depth-first searches use a stack, breadth-first searches use
a queue, and best-first searches use a priority queue.

Below is a generalized implementation of a breadth-first search which can be modified to use either of the other search
approaches by modifying its core data structure.
"""


class myQueue:
    """Implementing the queue class to use in the BFS implementation"""
    def __init__(self):
        self.queue = []  # Initializing the queue as an empty list. Python lists are dynamic arrays, which we need here.

    # Defining the enqueue method
    def enqueue(self, element):
        self.queue.append(element)  # This method adds an element to the end of the queue

    # Defining the dequeue method
    def dequeue(self):
        if not self.empty():  # If the queue is not empty
            return self.queue.pop(0)  # Remove and return the first element in the list - this is FIFO compliant

        else:  # If the queue is in fact empty
            raise IndexError('Dequeue has been called on an empty queue.')  # Raise an index error

    # Defining the empty method
    def empty(self):
        return len(self.queue) == 0  # The queue is empty if its length is zero. This 'state' can be repeatedly checked.

    # Defining the string method
    def __str__(self):
        return str(self.queue)  # Return a string representation of the current 'state' of the queue


def BFS(G, s):
    """This function runs a breadth-first search on an adjacency list representing a graph"""
    # Initialize a list to store distances from a source vertex 's'
    distance = [float('inf')] * len(G)  # Initialize as many 'infinities' as there are vertices in the graph

    visited = set()  # Initialize a set to keep track of the vertices that the BFS has visited

    myQueue_instance = myQueue()  # Initialize an instance of my 'myQueue' class

    distance[s] = 0  # Set the distance from the source vertex 'to itself' as 0

    visited.add(s)  # Mark the source vertex as 'visited'

    myQueue_instance.enqueue(s)  # Add the source vertex to the queue and start doing a BFS

    while not myQueue_instance.empty():  # As long as the queue is not empty
        current_vertex = myQueue_instance.dequeue()  # Iterate to the next vertex, pop it out, and 'visit' its neighbors

        for neighbor_vertex in G[current_vertex]:  # For each neighboring vertex of the current vertex
            if neighbor_vertex not in visited:  # If this vertex has not been listed as 'visited'
                visited.add(neighbor_vertex)  # Add it to the 'visited' set

                distance[neighbor_vertex] = distance[current_vertex] + 1  # Update the shortest distance to the vertex

                myQueue_instance.enqueue(neighbor_vertex)  # And add it to the queue to 'visit' its neighboring nodes

    return distance  # Finally, return the list of updated distances
