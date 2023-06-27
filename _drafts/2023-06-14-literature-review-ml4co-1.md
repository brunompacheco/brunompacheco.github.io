---
layout: distill
title: "ML4CO - Part 1: A brief overview of machine learning for combinatorial optimization"
date: 2023-06-14
description: The big picture on machine learning applications as heuristics for combinatorial optimization.
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

<!-- TODO: find a better way to highlight this first section -->
*This text is a direct result from the literature review I performed for my Master's.
The main goal of this first part is to provide context and background for the following review on end-to-end learning-based heuristics for MILP.
This is largely based on the work by Bengio et al.<d-cite key="bengio_machine_2021"></d-cite>, so I recommend it to the reader that wants to find more detailed information and more references.*

<!-- ###  ‎ -->
<br/>

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
Machine learning (ML) techniques, on the other hand, seem like the perfect fit for such heuristics.
For example, computing which routes pass by low-speed zones and which do not may be as expensive as finding a good route.
At the same time, historical data on past routes along with the speed at each section can be used to train a classifier that determines whether a new route is slow or not.
Furthermore, deep learning has shown great results applied to high-dimensional structured data such as image, proteins, and text, which makes us think that it could provide great results as well when applied to CO problem instances.

## Algorithm selection

A CO problem is a problem of minimizing an _objective function_ given a finite set of _feasible solutions_.
Given an algorithm for solving CO problems, its quality is usually determined from its computational complexity and experimental results on benchmark instances.
However, an algorithm that works well on some classes of CO problems may not work well on others.
Thus, practitioners end up having to experiment between the alternatives and, if none fulfills the requirements, having to adjust configurations, exploring the problem's structure, manually building heuristics, etc.

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

## Learning-based heuristics

When we consider CO algorithms enhanced with machine learning models, the comparison is often over an uncountable set of algorithms.
For example, let $$\Theta$$ be the parameter space of the ML models, and $$\mathcal{A}=\left\{ a(\theta) : \theta\in \Theta \right\} $$, i.e., the algorithms are defined by the ML model's parameters.
The problem of selecting the best algorithm can be written

$$
    \max_{\theta\in \Theta} \, \frac{1}{|\mathcal{D}|} \sum_{I\in \mathcal{D}} m(I,a(\theta))
.$$

In other words, rather than searching for the best algorithm, one searches for the best vector of parameters.

With respect to how ML models can be used in algorithms for CO, Bengio et al.<d-cite key="bengio_machine_2021"></d-cite> proposed three categories of learning-based heuristics.
Even though the categorization is not exhaustive (as will be seen later on), it is useful to show the possibilities that exist for applying ML models.

At the "deepest" level, an ML model can be trained to take decisions within CO solvers, replacing costly computations or in the place of already existing heuristics within the solver.
An example of this approach can be seen in Nair et al.<d-cite key="nair_solving_2021"></d-cite>, where the authors trained a deep learning model to select between variables for branching, within a branch-and-bound algorithm.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/bengio_aoa.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Diagram of a ML model being used to take decisions within a CO solver (_OR_ block). Image from Bengio et al.<d-cite key="bengio_machine_2021"></d-cite>.
</div>

The second category comprises heuristics with ML models being called to take decisions prior to the execution of the CO solvers.
In this approach, the ML model's output helps to define the information provided to the CO solver.
In Kruber et al.<d-cite key="kruber_learning_2017"></d-cite>, the authors trained a model to decide whether to apply a Dantzig-Wolfe decomposition to reformulate an instance or not, based on the predicted running time reduction of the solver.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/bengio_l2c.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Diagram of a ML model being used to enhance the information provided to a CO solver's (_OR_ block). Image from Bengio et al.<d-cite key="bengio_machine_2021"></d-cite>.
</div>

Finally, ML models can be trained to predict a solution based on the information of an instance, which will be referred to as an _end-to-end_ approach.
An example is the work by Vinyals et al.<d-cite key="vinyals_pointer_2015"></d-cite>, in which the authors propose a novel deep learning model capable of providing feasible solutions to the TSP.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/bengio_e2e.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Diagram of a ML model being used as a heuristic by itself, i.e., without calling an optimization algorithm. This is the end-to-end approach. Image from Bengio et al.<d-cite key="bengio_machine_2021"></d-cite>.
</div>

Note that end-to-end models, beside being trained to predict a solution, can be used in different settings.
For example, the model's output (a candidate solution) can be used to define a region for proximity search, as done by Han et al.<d-cite key="han_gnn-guided_2023"></d-cite>.

## Training ML models

### Supervised learning

For any given structure of CO learning-based heuristic, training through supervision requires us to provide data on the inputs and expected outputs for the model.
We assume that the training data was generated by an "expert", which the model will learn to imitate.
Supervised learning is possible whenever we have either observations on the expert (e.g., historical performance of a human operator), or access to a data generation process.
However, supervised learning is mostly adequate to problems in which the expert is not suitable for the application, e.g., because it is too expensive to compute, once the ML model's performance will, at best, be *on par* with the expert's performance.

In Gasse et al.<d-cite key="gasse_exact_2019"></d-cite>, the authors take as an expert the strong branching rule, which is known to provide good results in branch-and-bound, but is expensive to compute.
The data is generated beforehand and fed to the training algorithm as input-output pairs, which guides the ML model towards imitating the rule.

### Reinforcement learning

By letting the model explore the possible outputs and rewarding it accordingly to the consequences, c
Instead of "teaching" the desired behavior for the model, 

- ML models can be trained in 2 ways (imitation and exploration)
    - Imitation => faster computation of a known expert
    - Exploration => possibly a better performance than known expert
