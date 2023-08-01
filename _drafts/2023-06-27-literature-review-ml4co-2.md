---
layout: distill
title: "ML4CO - Part 2: A less-brief review of supervised end-to-end models for mixed-integer linear programming"
date: 2023-06-14
description: A review of state-of-the-art end-to-end heuristics for MILP built upon deep learning models trained (with supervision) to predict candidate solutions. 
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
  - name: Data generation
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

Mixed-integer programming plays a vital role in many applications, as it can model planning<d-cite key="pochet_production_2006"></d-cite>, scheduling<d-cite key="sawik_scheduling_2011"></d-cite>, routing<d-cite key="malandraki_time_1992"></d-cite> and many other problems.
In fact, most combinatorial optimization problems can be modeled as mixed-integer programs, as the integer variables can model the discrete nature of realistic decisions<d-cite key="ibaraki_integer_1976,nemhauser_scope_2014"></d-cite>.

Furthermore, there are reliable algorithms<d-cite key="morrison_branch-and-bound_2016"></d-cite> and software solutions<d-cite key="bestuzheva_scip_2021"></d-cite> to solve MILPs, which makes them widely used for combinatorial optimization problems even if the problem involves nonlinear objective and/or constraints, e.g., by using piecewise linearization, which admits MILP formulations<d-cite key="bernreuther_solving_2017"></d-cite>.

Heuristics can be used to solve MILP problems whenever there is a time limitation and a tolerance for sub-optimal solutions, as is the case for any combinatorial optimization problem.
In fact, by the NP-hard nature of MILP, heuristics are often vital for large problems.
However, heuristics traditionally require expert knowledge to be developed, which is expensive and may not result in an efficient algorithm.

Instead of an expert, we can use data and machine learning to develop a learning-based heuristic.
One way to apply machine learning algorithms is to train a deep learning model through supervision to predict a candidate solution given a problem instance.
Such *end-to-end* model can be used straight away, as a heuristic itseld, or it can be augmented, for example, to warmstart a MILP solver<d-cite key="khalil_mip-gnn_2022"></d-cite>, in a matheuristic.
The building blocks of an MILP (mat)heuristics based on supervised, end-to-end deep learning models are summarized in the diagram below.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/e2e_heuristic.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Building blocks of a (mat)heuristics based on supervised, end-to-end deep learning models.
    The <em>Prep.</em> block represent the pre-processing operations necessary to embed an instance of an MILP problem into a format that can be fed to the deep learning model (<em>DL</em> block).
    The model returns a predicted candidate solution, which can be rounded and directly provide the heuristic output, or can be optimized over (<em>OPT</em> block, with dashed border since it is optional), forming a matheuristic.
    The style follows the diagrams from Bengio et al.<d-cite key="bengio_machine_2021"></d-cite>.
</div>


- MILP
  - Definition
  - use qualification text
  - Solving through branch-and-bound (maybe introduce through the bnb tree + tree search paradigm?)
- End-to-end models
  - Recap algorithm selection problem
  - end-to-end models imply in algorithms that have as input the instance and as output a predicted optimal solution (note that there might be multiple optimal solution)
  - model architecture (roughly, Embedding MILP instances, from qualification)
    - naive approach = FCN on (A,b,c) tuple
    - feature engineering approach = FCN on f(I), where f may be non-differentiable
    - graph approach = GNN on G(I)
    - text from qualification
    - find references for each approach
  - Data generation
    - of course, one can resort to historical data that is accurate to the target distribution (instances that will be seen in practice), but assuming that is an overly optimistic scenario
    - as DL models require plenty of data for training, it is more encompassing to assume that instances will need to be generated
    - random generation and feasibility verification
    - generating instances of optimization problems is not a challenge just for learning-based applications, effort has already been put
    - works on instance generation
  - training
    - (quasi-)optimal as target
    - multi-target
    - weakly-supervised
- Learning-based heuristics
  - having a model, how to find a good solution fast?
  - naive application = model as heuristic
  - warm starting
  - trust region
  - early fixing
- Challenges
  - Efficient architectures
    - Symmetries of the problem are not well explored for non-GNN approaches
    - GNNs are new?
    - what about trianing algorithms? (Bengio talks a bit about this)
  - Data generation
    - Instance generation
      - high dimensionality => historical data is usually insufficient to grasp the real distribution => generated data is out-of-touch with reality
      - feasible instances are hard to generate - polynomial time => easier subproblem
      - instance generation approaches, maybe combining generated instances, genetic algorithm?
    - Solution finding
      - Same problem as feasibility, but harder?
      - Using non-optimal solutions, weighted by the objective value (multi-target)
  - Guarantees
    - Feasibility and optimality guarantees
  - ?
