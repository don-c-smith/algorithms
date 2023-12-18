import pandas as pd
import csv  # I need this for the QUOTE_MINIMAL method in the data-loading function


def load_investments(file_name):
    """This function uses the Pandas library to load and process the investment data from the CSV file.
    It needs to handle data with and without quotation marks."""
    # Using the 'quoting' parameter from the csv library to handle different quoting conventions
    # Using the 'skiprows' parameter to skip the first non-header row (at index 1) as per the prompt
    # Pandas is reading the first row as the header automatically
    data = pd.read_csv(file_name, quoting=csv.QUOTE_MINIMAL, skiprows=[1])

    # Using Pandas, I can prevent errors by converting any relevant fields incorrectly read as strings to numeric types
    # If the conversion encounters errors, errors='coerce' will convert those errors to NaN, Pandas for 'not a number'
    # This will prevent bad values from messing up my calculation of ROI
    data['Zhvi'] = pd.to_numeric(data['Zhvi'], errors='coerce')  # Handling the needed 'Zhvi' field
    data['10Year'] = pd.to_numeric(data['10Year'], errors='coerce')  # Handling the needed '10Year' field

    # Calculating the estimated return on investment for each available investment option
    data['ROI'] = (data['Zhvi'] * data['10Year'])

    # Print debugging
    # print('Columns in DataFrame:', data.columns)
    # print('Data types:', data.dtypes)
    # print('First rows:', data.head())

    # Converting the values in the dataframe into a list of investment options, organized as (name, cost, estimated ROI)
    # I will use Pandas' .itertuples() method to iterate over rows as named tuples
    # The result is a list of tuples, each of which represents a single investment option
    investments = list(data[['RegionName', 'Zhvi', 'ROI']].itertuples(index=False, name=None))

    return investments  # Return the list of tuples


def optimize_investments(investment_options, budget):
    """This function uses dynamic programming to optimize investment ROI given a set of possible investments and
    a stated investment budget."""
    investment_count = len(investment_options)  # The length of the list of tuples is the number of investment options

    # Initialize a 2D 'optimal' table for implementing dynamic programming and fill with zeros
    optimal_table = [[0 for _ in range(budget + 1)] for _ in range(investment_count + 1)]  # List comprehension

    # Initialize a traceback table to track the actual investments in the optimal solution and fill with empty lists
    traceback_table = [[[] for _ in range(budget + 1)] for _ in range(investment_count + 1)]  # More list comprehension

    # Iterate over each available investment option
    # There's an excellent use case here for tuple-unpacking. One-step processing of the data.
    for investment_index in range(1, investment_count + 1):
        inv_name, inv_cost, inv_roi = investment_options[investment_index - 1]
        print(f'Checking investment: {inv_name}, Cost: {inv_cost}, ROI: {inv_roi}')  # Print debugging

        # Then, at the next iterative level, iterate over each budget level/value
        for iter_budget in range(1, budget + 1):
            # First, check if the iterated investment can be included/bought with the current budget value
            if inv_cost <= iter_budget:
                # We include the current investment *if* it leads to a higher ROI and update both tables
                if inv_roi + optimal_table[investment_index - 1][iter_budget - inv_cost] > \
                        optimal_table[investment_index - 1][iter_budget]:

                    # Update optimal table with improved value
                    optimal_table[investment_index][iter_budget] = inv_roi + \
                        optimal_table[investment_index - 1][iter_budget - inv_cost]

                    # Update traceback table with relevant investment name
                    traceback_table[investment_index][iter_budget] = \
                        traceback_table[investment_index - 1][iter_budget - inv_cost] + [inv_name]

                    # Print debugging
                    # print(f"At budget {iter_budget}, optimal ROI: {optimal_table[investment_index][iter_budget]}")

                else:  # Otherwise, if the investment does *not* lead to a higher ROI, we exclude it
                    optimal_table[investment_index][iter_budget] = optimal_table[investment_index - 1][iter_budget]
                    traceback_table[investment_index][iter_budget] = traceback_table[investment_index - 1][iter_budget]

            else:
                # If the investment cost is more than the iterated budget value, it cannot be purchased/included
                optimal_table[investment_index][iter_budget] = optimal_table[investment_index - 1][iter_budget]
                traceback_table[investment_index][iter_budget] = traceback_table[investment_index - 1][iter_budget]

    # Now locate the optimal ROI and the corresponding investments in the tables
    best_roi = optimal_table[investment_count][budget]  # Fetch ROI from optimal table
    best_investments = traceback_table[investment_count][budget]  # Fetch investment names from traceback table

    return best_roi, best_investments  # Return both results


# Testing code
# file_path = 'zhvi_short.csv'  # Location of 'short' investment options file
file_path = 'state_zhvi_summary_allhomes.csv'  # Location of 'full' investment options file

investments_list = load_investments(file_path)  # Create list of tuples to pass to the optimize_investments function

budget_value = 1000000  # Declare available budget

selected_roi, selected_investments = optimize_investments(investments_list, budget_value)  # Multiple assignment

# Print results of analysis
print(f'Optimal ROI: {selected_roi}')
print(f'Optimal Investments: {selected_investments}')
