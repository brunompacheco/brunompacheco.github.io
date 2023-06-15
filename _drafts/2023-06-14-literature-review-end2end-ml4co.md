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

Say you are in a big city and you need to go from one place (point A) to another (point B) in the fastest way.
As there are finite many streets, there are finite many routes<d-footnote>As we are considering only routes without loops (acyclic graphs)</d-footnote>, thus, to find the fastest route, you can write a computer program that evaluates all possible routes and returns to you the fastest.
However, the number of routes grow exponentially with the number of streets, so for a big city a random walk will probably take you to the destination before the computer program can compute all routes.
Combinatorial optimization (CO) problems, such as finding the fastest route, are hard.
In fact, CO is often used to refer to NP-hard integer optimization problems.

Despite being NP-hard, many CO problems are solved within reasonable time even for millions of variables and constraints.
This is usually due to experts being able to exploit structures of the problem to create efficient heuristics that can generate good-enough approximations.
Suppose you are quite familiar with the city in which you want to find the fastest route.
Instead of iterating over all possibilities, you remove streets that take you away from your target, streets that will probably be blocked by traffic, and prioritize streets that you know are quite fast.
Your new computer program will not iterate over all possible routes, so it may not find the optimal, but it will probably give you a very good route much quicker than the old one.
Furthermore, instead of writing a computer program, you can just use your knowledge to guess a route based on your previous trips around the neighborhood.
You probably will not guess the best route, but your guess may be enough for you.

Sadly, the expert knowledge to develop heuristics for CO problems takes a lot of effort to develop and may result in heuristics that are not cheap to compute or easy to implement.
Machine learning (ML) techniques, on the other hand, seem like the perfect fit for such heuristics.
In particular, deep learning has shown great results applied to high-dimensional structured data such as image, proteins, and text, which makes us think that it could provide great results as well when applied to CO problem instances.

In the following, you will see a brief introduction to machine-learning-based heuristics and then a more in-depth review of end-to-end approaches to mixed-integer linear programming (MILP).
_MILP_ because it is a formulation that covers many types of CO problems, and there are well-developed algorithms for it.
_End-to-end_ because it will focus on deep learning models that are developed to predict candidate solutions to MILP instances, and heuristics that are built on top of such models.

## Machine Learning for Combinatorial Optimization

The following is largely based on <d-cite key="bengio_machine_2021"></d-cite>, so I recommend you to it for a deeper look and more references.

Give examples along the way!

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

