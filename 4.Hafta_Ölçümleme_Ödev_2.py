########################################################
# Rating Product & Sorting Reviews in Amazon
########################################################

##################
# Business Problem
##################
"""
One of the most important problems in e-commerce is the correct calculation of the points given to the products after sales.
The solution to this problem means providing greater customer satisfaction for the e-commerce site, prominence of the product
for the sellers and a seamless shopping experience for the buyers.
Another problem is the correct ordering of the comments given to the products.
Since misleading comments will directly affect the sale of the product, it will cause both financial loss and loss of customers.
In the solution of these 2 basic problems, e-commerce site and sellers will increase their sales,
while customers will complete their purchasing journey without any problems.
"""

##################
# Dataset Story
##################
"""
This dataset, which includes Amazon product data, includes product categories and various metadata. 
The product with the most reviews in the electronics category has user ratings and reviews.
"""

#   Total Features : 12
#   Total Row : 4915
#   CSV File Size : 71.9MB

#	reviewerID  :   User Id
#	asin    	:   Product Id
#	reviewerName:	User Name
#	helpful     :	Useful Evaluation Degree
#	reviewText  :	Evaluation
#	overall     :	Product Rating
#	summary     :	Evaluation Summary
#	unixReviewTime:	Evaluation Time
#	reviewTime  :	Evaluation Time {RAW}
#	days - day_diff:	Number of days since assessment
#	helpful_yes :	The number of times the evaluation was found useful
#	total_vote  :	Number of votes given to the evaluation


###############################
# Preparing and Analyzing Data
###############################
import pandas as pd
import math
import scipy.stats as st

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

df_= pd.read_csv(r"C:\Users\mustafapolat\Desktop\Muzzpol\VBO\Python\Ölçümleme Problemleri\amazon_review.csv")
df = df_.copy()
df.head()
df.shape
df.info()
df.describe().T
#############################################################################################
# 1. Calculate the Average Rating according to the current comments and compare it with the existing average rating.
#############################################################################################

"""
In the shared data set, users gave scores and comments to a product. 
Our aim in this task is to evaluate the scores given by weighting them by date.
It is necessary to compare the first average score with the weighted score according to the date to be obtained.
"""

#############################################################################################
# 1.1 Calculate the average score of the product.
#############################################################################################
df['overall'].mean()

#############################################################################################
# 1.2 Calculate the weighted average score by date
#############################################################################################
"""
--> You need to declare the reviewTime variable as a date variable.
--> accepting the max value of reviewTime as current_date,
--> You need to create a new variable by expressing the difference of each score-comment date and current_date in days, 
and divide the variable expressed in days by 4 with the quantile function (3 quarters, 4 parts are given) and 
weight according to the values from the quarters.
For example, if q1 = 12, when weighting, averaging comments made less than 12 days ago and giving them a higher weight.
"""
df["reviewTime"]=pd.to_datetime(df["reviewTime"])
df['reviewTime'].max()
current_date=pd.to_datetime('2014-12-07 00:00:00')
df["days"]=(current_date-df["reviewTime"]).dt.days
df.head(3)


print("--------------------- Quantiles ---------------------")
    print(df["days"].quantile([0.25, 0.50, 0.75, 1]).T)

Q_1 = df["days"].quantile(0.25)
Q_2 = df["days"].quantile(0.50)
Q_3 = df["days"].quantile(0.75)

print(df.loc[(df["days"] <= 283), "overall"].mean())
# OUTPUT : 4.6957928802588995
print(df.loc[(df["days"] > 283) & (df["day_diff"] <= 433), "overall"].mean())
# OUTPUT : 4.634677419354839
print(df.loc[(df["days"] > 433) & (df["day_diff"] <= 603), "overall"].mean())
# OUTPUT : 4.573042776432607
print(df.loc[(df["days"] > 603), "overall"].mean())
# OUTPUT : 4.4462540716612375

#############################################################################################
# 2. Specifying 20 reviews for the product that will be displayed on the product detail page
#############################################################################################

#############################################################################################
# 2.1 Generate the helpful_no variable
#############################################################################################
"""
* total_vote is the total number of up-downs given to a comment.
* up means helpful.
* There is no helpful_no variable in the data set, it must be generated over existing variables.
* Find the number of votes that are not helpful (helpful_no) by subtracting the number of 
helpful votes (helpful_yes) from the total number of votes (total_vote).
"""
df["helpful_no"] = df["total_vote"] - df["helpful_yes"]
df.head()

#############################################################################################
# 2.2 Calculate score_pos_neg_diff, score_average_rating and wilson_lower_bound scores and add them to the data
#############################################################################################
"""
To calculate the "score_pos_neg_diff", "score_average_rating" and "wilson_lower_bound" scores, 
define the score_pos_neg_diff, score_average_rating and wilson_lower_bound functions.

• Create scores based on score_pos_neg_diff.
Next; Save it as score_pos_neg_diff in df.

• Create scores based on score_average_rating.
Next; Save it as score_average_rating in df.

• Create scores according to wilson_lower_bound.
Next; Save it as wilson_lower_bound in df.
"""

def wilson_lower_bound(up, down, confidence=0.95):
    """
    Wilson Lower Bound Score hesapla
    - Bernoulli parametresi p için hesaplanacak güven aralığının alt sınırı WLB skoru olarak kabul edilir.
    - Hesaplanacak skor ürün sıralaması için kullanılır.
    - Not:
    Eğer skorlar 1-5 arasıdaysa 1-3 negatif, 4-5 pozitif olarak işaretlenir ve bernoulli'ye uygun hale getirilebilir.
    Bu beraberinde bazı problemleri de getirir. Bu sebeple bayesian average rating yapmak gerekir.
    Parameters
    ----------
    up: int
        up count
    down: int
        down count
    confidence: float
        confidence
    Returns
    -------
    wilson score: float
    """
    n = up + down
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * up / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)


def score_up_down_diff(up, down):
    return up - down

def score_average_rating(up, down):
    if up + down == 0:
        return 0
    return up / (up + down)

# score_pos_neg_diff
df["score_pos_neg_diff"] = df.apply(lambda x: score_up_down_diff(x["helpful_yes"], x["helpful_no"]), axis=1)

# score_average_rating
df["score_average_rating"] = df.apply(lambda x: score_average_rating(x["helpful_yes"], x["helpful_no"]), axis=1)

# wilson_lower_bound
df["wilson_lower_bound"] = df.apply(lambda x: wilson_lower_bound(x["helpful_yes"], x["helpful_no"]), axis=1)
df.head(10)

#############################################################################################
# 2.3 # Identify the top 20 Comments and Comment the results: (Identify and rank the first 20 comments by wilson_lower_bound)
#############################################################################################

df.sort_values("wilson_lower_bound", ascending=False).head(20)
