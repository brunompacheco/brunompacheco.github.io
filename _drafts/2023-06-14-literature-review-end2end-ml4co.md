---
layout: post
title: a literature review of end-to-end machine learning for combinatorial optimization
date: 2023-06-14 19:19:00-0300
description: a brief review of machine learning for combinatorial optimization with a focus on end-to-end methods for mixed-integer linear programs
tags: ml4co milp msc-thesis deep-learning
categories: literature-review
related_posts: false
---

Combinatorial optimization (CO) is hard.
In fact, CO is often used to refer to NP-hard integer optimization problems.
However, many CO problems can be solved within reasonable time even for millions of variables and constraints.
This is usually due to experts being able to exploit structures of the problem to create efficient heuristics that can generate good-enough approximations.
Intuitively, this is equivalent to the intuition you develop on moving around town.
Let us take as an example the problem of finding a route from point A to point B within your hometown.
Even though you may have never made this trip, you will hardly consider every possible street combination, as you already know that some streets are slow or will take you far from where you want to go, and you are used to a couple of paths that are fast enough.
In other words, evaluating every possible street combination will probably result in a faster route, but the gain will not be worth the effort if compared to just resorting to your intuition.

Sadly, the expert knowledge to develop heuristics for CO problems takes a lot of effort to develop and may result in heuristics that are not cheap to compute or easy to implement.
Machine learning techniques, on the other hand, seem like the perfect fit for such heuristics.
In particular, deep learning has shown great results applied to high-dimensional structured data such as image, proteins, and text, which makes us think that it could provide great results as well when applied to CO problem instances.

In the following, you will see a brief introduction to machine-learning-based heuristics and then a more in-depth review of end-to-end approaches to mixed-integer linear programming (MILP).
_MILP_ because it is a formulation that covers many types of CO problems, and there are well-developed algorithms for it.
_End-to-end_ because it will focus on deep learning models that are developed to predict candidate solutions to MILP instances, and heuristics that are built on top of such models.

