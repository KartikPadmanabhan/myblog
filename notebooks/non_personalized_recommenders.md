---
layout: post
title: "non_personalized_recommenders"
tags:
    - python
    - notebook
---
# NON-PERSONALIZED RECOMMENDER SYSTEMS USING MOVIE-LENS DATASET

Non-personalized recommenders systems is a system where everyones gets to see
the same set of recommendation. The site doesn't logs in user details and
transaction info, and the site assumes that every user is same and displays a
list of general recommendations.

To explain this I am going to use the MovieLens data sets that were collected by
the GroupLens Research Project at the University of Minnesota.

This data set consists of:
        * 100,000 ratings (1-5) from 943 users on 1682 movies.
        * Each user has rated at least 20 movies.
    * Simple demographic info for the users (age, gender, occupation, zip)

I have made available their dataset and the ipython notebook used for this
exercise in my github page.

**In []:**

{% highlight python %}

{% endhighlight %}

**In []:**

{% highlight python %}

{% endhighlight %}

**In []:**

{% highlight python %}

{% endhighlight %}

The fields in this dataset are the userid, itemid (movie) that a paerticular
user has rating, his rating and the corresponding time at which the user has
rated the movie. Please note that the timestamps are unix seconds since 1/1/1970
UTC. Lets create a fundtion that converts this information into actual datetime
object

**In []:**

{% highlight python %}

{% endhighlight %}

Converting the timestamps into the corresponing datetime object

**In []:**

{% highlight python %}

{% endhighlight %}

**In []:**

{% highlight python %}

{% endhighlight %}

**In []:**

{% highlight python %}

{% endhighlight %}

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 100000 entries, 0 to 99999
    Data columns (total 4 columns):
    userid       100000 non-null int64
    itemid       100000 non-null int64
    rating       100000 non-null int64
    timestamp    100000 non-null datetime64[ns]
    dtypes: datetime64[ns](1), int64(3)
    memory usage: 3.8 MB


Now lets read the u.item data that contains the corresponding movie names for
the movie id (itemid)

Information about the items (movies); this is a tab separated list of
        movie id | movie title | release date | video release date |
        IMDb URL | unknown | Action | Adventure | Animation |
        Children's | Comedy | Crime | Documentary | Drama | Fantasy |
        Film-Noir | Horror | Musical | Mystery | Romance | Sci-Fi |
        Thriller | War | Western |

The last 19 fields are the genres, a 1 indicates the movie is of that genre, a 0
indicates it is not; movies can be in several genres at once. The movie ids are
the ones used in the u.data data set.

**In []:**

{% highlight python %}

{% endhighlight %}

**In []:**

{% highlight python %}

{% endhighlight %}

For the sake of this exercise I am just using the movie title and itemid (movie
id), lets ignore other columns

**In []:**

{% highlight python %}

{% endhighlight %}

**In []:**

{% highlight python %}

{% endhighlight %}

Lets merge all these data into our original dataframe so that we also have all
the movie information alongside.

**In []:**

{% highlight python %}

{% endhighlight %}

**In []:**

{% highlight python %}

{% endhighlight %}

Now say we are entering a website like Netflix without logging in and not
providing any user-specific information. For non-personalized recommendation,
the question is what movies do we recommend for the users to watch. There are
many ways to approach this, lets take a stab at it one by one.

## MEAN RATING

In this approach we calculate the mean mean rating for each movie, order with
the highest rating listed first, and submit the top rated movies.

**In []:**

{% highlight python %}

{% endhighlight %}

**In []:**

{% highlight python %}

{% endhighlight %}

**In []:**

{% highlight python %}

{% endhighlight %}

**In []:**

{% highlight python %}

{% endhighlight %}

Say I want to recommend all the users in the front landing page the movies with
the average rating of 5 stars, I would recommend the following movies.

**In []:**

{% highlight python %}

{% endhighlight %}

Something seems obviously wrong (apologies if your faviourite movies is among in
this list), but most of these movies seems not that popular compared to several
other good movies I have seen.

To dig more closer, lets get an histogram of how many people have rated these
movies that show the average rating of 5 stars.

**In []:**

{% highlight python %}

{% endhighlight %}

    [1, 3, 2, 3, 1, 1, 2, 1, 1, 1]


This is exactly where our problem is. The problem is that all of these movies
are rated by either 1, 2 or 3 users and they have all given 5 stars to it. That
doesnt make these movies great and we would rather shows the movies that have
been rated by many number of users and also with highest ratings on our
recommendations. This is the issue with a lot of non-personalized recommenders.

## SCALES AND NORMALIZATION

Take a look at the following link : https://janav.wordpress.com/2013/09/21/non-
personalized-recommenders/

The author explains amazon.com's non-personalized recommendations that appear in
section "customers who bought this item also bought" when users tries to look
for a product.

How would it look like when we want to recommend movies instead of amazon
product.

To illustrates this, I have taken out the following 5 movies in the movielens
dataset.

**In []:**

{% highlight python %}

{% endhighlight %}

**In []:**

{% highlight python %}

{% endhighlight %}

**In []:**

{% highlight python %}

{% endhighlight %}

**In []:**

{% highlight python %}

{% endhighlight %}

**In []:**

{% highlight python %}

{% endhighlight %}

Lets convert all nans to 0 and all other ratings to 1s.

**In []:**

{% highlight python %}

{% endhighlight %}

Say I like "12 Angry Men" and I want to watch this movie. In order for the non-
personalized recommender system to recommender other movies that people have
watched based on this movie, we use the following formulae.

Score(X, Y) =  Total People who watched X and Y / Total People who watched X

Total People who watched X are: (non-null entries)

**In []:**

{% highlight python %}

{% endhighlight %}

Total People who watched X and Y:

**In []:**

{% highlight python %}

{% endhighlight %}

**In []:**

{% highlight python %}

{% endhighlight %}

**In []:**

{% highlight python %}

{% endhighlight %}

Hence based on the highest scores, we would recommend 'Star Wars (1977)' and
'Toy Story (1995)' for the user to watch.

### BANANA TRAP

We want to make sure we aren't in banana trap here. What is a banana trap?

In a grocery store most of the customers will buy bananas. If someone buys a
razor and a banana then you cannot tell that the purchase of a razor influenced
the purchase of banana. Hence we need to adjust the formula to handle this case
as well. The modified version is

(Total People who watched X and Y / Total People who watched X) /
(Total People who did not watch X but watched Y / Total People who did not watch
X)

We have the numberator, we just need to calculate the denominator. Lets create a
seperate column for people who did not watch 12 Angry Men.

**In []:**

{% highlight python %}

{% endhighlight %}

**In []:**

{% highlight python %}

{% endhighlight %}

**In []:**

{% highlight python %}

{% endhighlight %}

This completely changes the game. Our normalized recommendation after adjusting
for banana trap are GoodFellas (1990) and Alien 3 (1992) . This clearly shows
that almost a lot of users had rated on the Toy Story and Star Wars.

# DAMPED MEANS

**In []:**

{% highlight python %}

{% endhighlight %}

**In []:**

{% highlight python %}

{% endhighlight %}

**In []:**

{% highlight python %}

{% endhighlight %}

**In []:**

{% highlight python %}

{% endhighlight %}

**In []:**

{% highlight python %}

{% endhighlight %}

**In []:**

{% highlight python %}

{% endhighlight %}

**In []:**

{% highlight python %}

{% endhighlight %}
