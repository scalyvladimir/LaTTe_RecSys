# LaTTe_RecSys
# Collaborative Filtering with smooth ratings scale, Skoltech RecSys Course 2022

[Vladimir Chernyy](https://github.com/scalyvladimir), [Elizaveta Makhneva](https://github.com/elizacc)

Problem statement:

In the CoFFee model [Frolov and Oseledets, 2016](https://arxiv.org/pdf/1807.10634.pdf), ratings are represented as a categorical variables and
interactions are encoded into a third-order user-item-rating tensor. On lectures, we discussed that users may have individual rating scales. Some users may rarely rate movies with rating 5, while other may always assign 5 stars to almost any movie they watched. Apparently, these users have different perception of the rating scale and its relation to a movie quality. So even if these users will watch the same movies, there will
be a discrepancy in signals from the rating behavior, which will affect the ability of our recommender system
to properly extract patterns in data.

One hypothetical way to mitigate this problem is to introduce “rating scale smoothing” by imposing
some notion of similarity or proximity between different values of ratings. Clearly, rating 4 and 5 should be
closer to each other than ratings 3 and 5. The smoothing effect can be achieved with the hybrid formulation
similarly to the way an attention mechanism was used in Sequential Tensor Factorization model in Lecture
10.

Your task is to implement this idea within the CoFFee model. Adapt the code from Lecture 10 accordingly.
The triangular attention matrix must now be replaced with the Cholesky factor of a rating similarity matrix.
Try several different similarity measures. Perform comparative study of the smoothed version of CoFFee with the original one. Report your finding and explain the obtained results. You can use standard evaluation metrics in this task. Make sure to fairly tune both models and compare their optimal variants. You can use a Movielens dataset for your experiments.

<!-- GETTING STARTED -->
## Getting Started

This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/ctrlzet/imgrestore
   ```
