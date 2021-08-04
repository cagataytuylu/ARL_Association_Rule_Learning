###################################################
# Associotion Rule Learning Recommender
###################################################

# !pip install xlrd
from builtins import int

import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
from mlxtend.frequent_patterns import apriori, association_rules

########################
# Data Pre Processing
########################


df_ = pd.read_excel("datasets/online_retail_II.xlsx", sheet_name="Year 2010-2011")
df = df_.copy()
from helpers.helpers import check_df, retail_data_prep

# checking the df
check_df(df)

# data cleaning
df = retail_data_prep(df)



# data of Germany selected
df_ger = df[df["Country"] == "Germany"]

df_ger = df_ger[df_ger["StockCode"] != "POST"]  # delecting "post" from stock codes


# ilk olarak ARL işleyecek şekilde matrix oluşturmamız gerekiyor onu hazırlayalım.

############## invoice ve product df


from helpers.helpers import create_invoice_product_df, check_id

df_ger_inv_prod = create_invoice_product_df(df_ger, id=True)

# Lets check these id's
# user 1 product id: 21987
# user 2 product id: 23235
# user 3 product id: 22747

liste = [21987, 23235, 22747]
for i in liste:
    check_id(df_ger, i)

########################################
# Product reccomenttion for user who has a product on their Cart
########################################


frequent_itemsets = apriori(df_ger_inv_prod, min_support=0.01, use_colnames=True)

frequent_itemsets.sort_values("support", ascending=False).head(20)

rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)
rules.sort_values("support", ascending=False).head(100)

rules.sort_values("lift", ascending=False).head(100)

# Notes :
# support : probability of x and y occurring together
# confidence : probability of selling y when x is bought
# lift : When x is bought, the probability of buying y increases as much as the lift.


# recomendation for these id's
sorted_rules = rules.sort_values("lift", ascending=False)

product_id = 21987
recommendation_list = []

for i, product in enumerate(sorted_rules["antecedents"]):
    for j in list(product):
        if j == product_id:
            recommendation_list.append(list(sorted_rules.iloc[i]["consequents"])[0])

recommendation_list[0:2]

########################################
# Recommended product names
########################################


check_id(df_ger, recommendation_list[0])

rec_list = recommendation_list[0:3]
for i in rec_list:
    check_id(df_ger, i)


def check_id(dataframe, stock_code):
    product_name = dataframe[dataframe["StockCode"] == stock_code][["Description"]].values[0].tolist()
    print(product_name)



