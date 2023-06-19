---
layout: distill
title: "ML4CO: end-to-end supervised learning for MILP"
date: 2023-06-14
description: a brief overview of machine learning for combinatorial optimization and a less-brief review of supervised end-to-end models for mixed-integer linear programming
tags: ml4co milp msc-thesis deep-learning
categories: literature-review
giscus_comments: false

authors:
  - name: Bruno M. Pacheco
    # url: "https://en.wikipedia.org/wiki/Albert_Einstein"
    affiliations:
      name: DAS, UFSC

bibliography: ML2MILP.bib

# Optionally, you can add a table of contents to your post.
# NOTES:
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - we may want to automate TOC generation in the future using
#     jekyll-toc plugin (https://github.com/toshimaru/jekyll-toc).
toc:
  - name: Machine Learning for Combinatorial Optimization
    subsections:
      - name: Learning-based heuristics
      - name: Training ML models
  - name: Supervised Learning End-to-end Heuristics for MILP
    subsections:
      - name: MILP
        subsections:
          - name: Instances embedding

# Below is an example of injecting additional post-specific styles.
# If you use this post as a template, delete this _styles block.
# _styles: >
#   .fake-img {
#     background: #bbb;
#     border: 1px solid rgba(0, 0, 0, 0.1);
#     box-shadow: 0 0px 4px rgba(0, 0, 0, 0.1);
#     margin-bottom: 12px;
#   }
#   .fake-img p {
#     font-family: monospace;
#     color: white;
#     text-align: left;
#     margin: 12px 0;
#     text-align: center;
#     font-size: 16px;
#   }

---

Let us suppose that a delivery company must plan the route for a carrier given a set of packages.
The route must take the carrier through the recipients of every package and back to the company headquarters.
As there are finitely many packages, there are finitely many possible routes.
To find the optimal route, you can write a computer program that evaluates all possible routes and returns the one with the smallest cost.
It is easy to see that this program will always (eventually) finish and will return the optimal solution.
However, as the number of solutions (routes) grows exponentially with the size of the problem (number of recipients/packages), for a large-enough number of deliveries, committing to a random route will probably be faster than waiting for the program to finish.
This example is a classic occurrence of the Traveling Salesperson Problem (TSP), a widely studied combinatorial optimization (CO) problem.

CO problems are hard.
In fact, CO is often used to refer to NP-hard integer optimization problems.
But despite being NP-hard, many CO problems are solved within reasonable time even for millions of variables (and constraints).
This is usually due to experts being able to exploit structures of the problem to create efficient heuristics.

In the delivery company example, suppose that you are quite familiar with the location of the deliveries.
You modify the computer program, such that instead of iterating over all possible solutions, you remove routes that pass through residential, low-speed areas and prioritize routes that use high-speed roads.
Your new computer program will not iterate over all possible routes, so it may not find the optimal solution, but it will probably give you a very good route much quicker than the previous one.
Furthermore, instead of writing a computer program, you can just use your knowledge to guess a route based on your previous deliveries.
You probably will not guess the best route, but your guess may be good enough for you.

Sadly, the expert knowledge to develop heuristics for CO problems takes a lot of effort to develop and may result in heuristics that are not cheap to compute or easy to implement.
For example, automatically finding which routes pass by low-speed zones may be costly.
Machine learning (ML) techniques, on the other hand, seem like the perfect fit for such heuristics.
In particular, deep learning has shown great results applied to high-dimensional structured data such as image, proteins, and text, which makes us think that it could provide great results as well when applied to CO problem instances.

In the following, you will see a brief introduction to machine-learning-based heuristics and then a more in-depth review of end-to-end approaches to mixed-integer linear programming (MILP).
_MILP_ because it is a formulation that covers many types of CO problems, and there are well-developed algorithms for it.
_End-to-end_ because it will focus on deep learning models that are developed to predict candidate solutions to MILP instances, and heuristics that are built on top of such models.

## Machine Learning for Combinatorial Optimization<d-footnote>The following is largely based on <d-cite key="bengio_machine_2021"></d-cite>, so I recommend you to it for a deeper look and more references.</d-footnote>

A CO problem is a problem of minimizing an _objective function_ given a finite set of _feasible solutions_.
Given an algorithm for solving CO problems, its quality is usually determined from its computational complexity and experimental results on benchmark instances.
However, an algorithm that works well on some CO problems may not work well for others.
Thus, practitioners end up having to experiment between the alternatives on the market and, if none fulfills the requirements, having to adjust configurations, exploring the problem's structure, manually building heuristics, etc.

To better grasp the challenge of selecting an algorithm for a realistic application, let us take the delivery company example of the introduction.
We are going to suppose that the desired computer program must work only for problems on a given city and that the origin (company headquarters) is always the same.
The goal of the company is to find a software that can find a good route within a time limit.
Instead of picking a solver based on benchmarks, you test a few different solvers based on historical data (past orders).
As your instances are too large, none of the algorithms can find a good solution on time given their default setting, so you explore different configurations, tweaking the parameters in the hope of improving their performance.
Furthermore, you notice that the recipients are often grouped in certain regions, so you design an algorithm that first groups nearby recipients in a single vertex, solves this simplified problem, and then finds a route within each group of recipients given the outcome of the simplified problem.

We can provide a mathematical formulation for the problem of finding a good algorithm for the problem of interest.
Let $$\mathcal{I}$$ be the set of all instances of interest and $$P$$ be a probability distribution over $$\mathcal{I}$$.
Let $$\mathcal{A}$$ be the set of all algorithms that can solve instances of the problem of interest and $$m : \mathcal{I}\times \mathcal{A}\to \mathbb{R}$$ be a performance metric
For convenience, we will assume that, for any $$a_1,a_2 \in \mathcal{A}$$ and $$I \in  \mathcal{I}$$, then $$m\left( I,a_1 \right) > m\left( I,a_2 \right) $$ implies that $$a_1$$ outperforms $$a_2$$ in instance $$I$$.
The problem of finding the best algorithm can be described as

$$
    \max_{a\in \mathcal{A}} \, \mathbb{E}_{I\sim P } m(I,a)
.$$

As this is usually impossible to compute, one can use the approximation based on a dataset $$\mathcal{D}$$ of instances independently drawn from $$P$$.
The problem, then, becomes

$$
    \max_{a\in \mathcal{A}} \, \frac{1}{|\mathcal{D}|} \sum_{I\in \mathcal{D}} m(I,a)
.$$

### Learning-based heuristics

- ML models can integrate in 3 different ways with CO solvers
    - Bengio's categories

### Training ML models

- ML models can be trained in 2 ways (imitation and exploration)
    - Imitation => faster computation of a known expert
    - Exploration => possibly a better performance than known expert

## Supervised Learning End-to-end Heuristics for MILP

### MILP

#### Instances embedding

