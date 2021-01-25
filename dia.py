import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import scipy.sparse as sparse
import random

ddx=pd.read_csv('syditriage.csv')
ddx.columns=['sym','symptom','dis','diagnose','dg','wei']
cleaned_retail = ddx[['wei', 'symptom', 'diagnose']] # Get rid of unnecessary info

grouped_cleaned = cleaned_retail.groupby(['diagnose', 'symptom']).sum().reset_index() # Group together
grouped_cleaned.wei.loc[grouped_cleaned.wei == 0] = 1 # Replace a sum of zero purchases with a one to
# indicate purchased
grouped_purchased = grouped_cleaned.query('wei > 50') # Only get customers where purchase totals were positive
customers = list(np.sort(grouped_purchased.symptom.unique())) # Get our unique customers
products = list(grouped_purchased.diagnose.unique()) # Get our unique products that were purchased
quantity = list(grouped_purchased.wei) # All of our purchases
rows = grouped_purchased['symptom'].astype('category').cat.codes
# # # Get the associated row indices
cols = grouped_purchased['diagnose'].astype('category').cat.codes
# # # Get the associated column indices
purchases_sparse = sparse.csr_matrix((quantity, (rows, cols)), shape=(len(customers), len(products)))
print(np.shape(purchases_sparse))
ddx.groupby(['diagnose']).mean(),pd.DataFrame(customers)
basket_sets = pd.pivot_table(ddx, index='symptom',columns='diagnose',values='wei')
item_lookup = pd.DataFrame( list( basket_sets.columns ) ,columns=['StockCode'])# Only get unique item/description pairs

# matrix_size = purchases_sparse.shape[0]*purchases_sparse.shape[1] # Number of possible interactions in the matrix
# num_purchases = len(purchases_sparse.nonzero()[0]) # Number of items interacted with
# sparsity = 100*(1 - (num_purchases/matrix_size))

# def make_train(ratings, pct_test = 0.2):
#     test_set = ratings.copy()  # Make a copy of the original set to be the test set.
#     test_set[test_set != 0] = 1  # Store the test set as a binary preference matrix
#     training_set = ratings.copy()  # Make a copy of the original data we can alter as our training set.
#     nonzero_inds = training_set.nonzero()  # Find the indices in the ratings data where an interaction exists
#     nonzero_pairs = list(zip(nonzero_inds[0], nonzero_inds[1]))  # Zip these pairs together of user,item index into list
#     random.seed(0)  # Set the random seed to zero for reproducibility
#     num_samples = int(
#         np.ceil(pct_test * len(nonzero_pairs)))  # Round the number of samples needed to the nearest integer
#     samples = random.sample(nonzero_pairs, num_samples)  # Sample a random number of user-item pairs without replacement
#     user_inds = [index[0] for index in samples]  # Get the user row indices
#     item_inds = [index[1] for index in samples]  # Get the item column indices
#     training_set[user_inds, item_inds] = 0  # Assign all of the randomly chosen user-item pairs to zero
#     training_set.eliminate_zeros()  # Get rid of zeros in sparse array storage after update to save space
#     return training_set, test_set, list(set(user_inds))  # Output the unique list of user rows that were altered
#
# product_train, product_test, product_users_altered = make_train(purchases_sparse, pct_test = 0.01)
# product_train