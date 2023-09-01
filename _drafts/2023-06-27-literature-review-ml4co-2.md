---
layout: distill
title: "ML4CO - Part 2: A less-brief review of supervised end-to-end models for mixed-integer linear programming"
date: 2023-09-01
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

Mixed-integer programming plays a vital role in many applications, as it can be used to model planning<d-cite key="pochet_production_2006"></d-cite>, scheduling<d-cite key="sawik_scheduling_2011"></d-cite>, routing<d-cite key="malandraki_time_1992"></d-cite> and many other problems.
In fact, most combinatorial optimization problems can be modeled as mixed-integer programs, as the integer variables can model the discrete nature of realistic decisions<d-cite key="ibaraki_integer_1976,nemhauser_scope_2014"></d-cite>.

Furthermore, there are reliable algorithms<d-cite key="morrison_branch-and-bound_2016"></d-cite> and software solutions<d-cite key="bestuzheva_scip_2021"></d-cite> to solve MILPs, which make them widely used for combinatorial optimization problems even if the problem involves nonlinear objective and/or constraints, e.g., by using piecewise linearization, which admits MILP formulations<d-cite key="bernreuther_solving_2017"></d-cite>.

Instead of algorithms, heuristics can be used to solve MILP problems whenever there is a time limitation and a tolerance for sub-optimal solutions.
In fact, by the NP-hard nature of MILP, heuristics are often vital for large problems in realistic applications.
Heuristics traditionally require expert knowledge to be developed, which is expensive and may not result in an efficient algorithm.

Instead of an expert, we can use data and machine learning to develop a learning-based heuristic.
Machine learning allows us to distill the knowledge from data instead of the years of experience from an expert.
Furthermore, machine learning models are fast to compute (during inference) and, more specifically for the case of deep learning models, have shown great results to high-dimensional structured data.

One set of applications with promising results is to train a deep learning model through supervision to predict a candidate solution to an instance of an MILP problem.
Such an *end-to-end* model can be used straight away, as a heuristic itseld, or it can be augmented, for example, to warmstart an MILP solver<d-cite key="khalil_mip-gnn_2022"></d-cite>, in a matheuristic.
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

## MILP

Mathematical programming is the process of finding the best possible solution given a set of constraints by formulating programs to maximize or minimize an objective function.
A mixed-integer linear program describes the objective and the constraints through linear functions over the variables of interest.
Formally, we can formulate an MILP as

$$
\newcommand{\R}{\mathbb{R}}  
\newcommand{\Z}{\mathbb{Z}}
\newcommand{\N}{\mathbb{N}}
\newcommand{\Q}{\mathbb{Q}}
\newcommand{\vtheta}{\boldsymbol\theta}
\newcommand\bm[1]{\boldsymbol{#1}}
\begin{equation}\label{eq:milp-definition}\begin{split}
    \min_{\bm{x}} &\quad \bm{c}^T \bm{x} \\
    \text{s.t.}&\quad A\bm{x} \ge \bm{b} \\
	       & \bm{x} \in \mathbb{Z}^{k}\times \mathbb{R}^{n-k},
\end{split}\end{equation}$$

where vector $$\bm{x}$$ contains the $$n$$ variables of interest, of which $$x_1,\ldots,x_k$$ are integer variables.
$$\bm{c} \in \mathbb{R}^{n}$$ is the cost vector defining the objective function, and $$A \in \mathbb{R}^{m\times n},\bm{b}\in \mathbb{R}^{m}$$ describe the constraints.
Note that this formulation embraces problems with equality constraints (by using two inequalities) as well as maximization problems (by negating the values of the objective vector).
It is also possible to use MILPs to approximately solve non-linear problems by approximating the nonlinearities as, for example, piecewise linear functions or neural networks with ReLU activations, which admit MILP formulations <d-cite key="grimstad_relu_2019"></d-cite>.

Branch-and-bound (B&B) is a widely used technique to solve MILP problems.
It consists of dividing the problem into solving LP relaxations, that is, assigning values for some of the integer variables and solving LP relaxations.
B&B is an exploratory procedure, i.e., a tree search over the possible value assignments for the integer variables.
Thus, even though solving the LP relaxations can be done in polynomial time, the expensive tree search renders the algorithm NP-hard.

Using machine learning algorithms to quickly find solutions to MILP problems is not a new idea.
In fact, in 1985, John Hopfield showed that the Travelling Salesperson Problem could be solved using a neural network<d-cite key="hopfield_neural_1985"></d-cite>.
Unfortunately, the lack of specialized hardware <d-footnote>The lack of GPUs was a major drawback, as neural networks require performing operations with large matrices and vectors, which had to be split up to fit in the CPUs. This made them very slow in comparison to the heuristic competitors.</d-footnote> made this area of research fade out in the late 90s<d-cite key="smith_neural_1999"></d-cite>.

In the past few years, significant contributions have shed a new light into this area of study, now including deep learning techniques, which has driven a lot of attention from the research community<d-cite key="bengio_machine_2021,cappart_combinatorial_2022,zhang_survey_2023"></d-cite>.
Authors have suggested many approaches to incorporate deep learning models into algorithms for MILP.
Examples include using the model to help guide the tree search in a B&B algorithm<d-cite key="nair_solving_2021"></d-cite>, evaluating whether a decomposition can speed up the optimization<d-cite key="kruber_learning_2017"></d-cite>, and switching heuristics during the B&B iterations<d-cite key="liberto_dash_2016"></d-cite>.

## End-to-end models

End-to-end models are algorithms that take as input an instance of an MILP problem of interest, and provide as output a predicted optimal solution<d-footnote>Recall that an MILP may have multiple optimal solutions.</d-footnote>.
Several challenges arise in developing an end-to-end model, such as the representation of the input MILP instances, the choice of deep learning model, and the training algorithm.
In this section, we will present the multiple approaches present in the literature for achieving a model that is capable of predicting an optimal solution for MILP instances.

### Model architecture

<!-- There are clear interdependences between the embedding of the instance (_Prep._ block) and the deep learning model employed (_DL_ block).
However, as the pros and cons can be analysed individually, they will be presented individually. -->

Differently from tabular data, time series, or images, MILP instance are not easily translated into formats suitable for computer programs.
Particularly, deep learning models generally work over features of the input instance.
In other words, the MILP instance must be embedded into features that contain the necessary information for the model to provide the desired output.
<!-- One must describe an MILP instance through numbers and design the programs to handle these numbers in such a way that the outcome (more numbers) represents the desired result. -->

In the case of traditional neural networks, for example, the input must be a feature vector.
Let us call $$\mathcal{I}$$ the set of all instances of an MILP problem of interest.
The most straightforward way to embed an MILP instance $$I\in\mathcal{I}$$ formulated as in \eqref{eq:milp-definition} into a feature vector is by vectorizing the tuple $$(A,\bm{b}, \bm{c}, k)$$.
In fact, this was the way Hopfield and Tank approached a linear programming problem at the dawn of this area of research<d-cite key="tank_simple_1986"></d-cite>.
Note that this representation of the instance contains all information necessary to completely re-assemble the original problem instance.
However, this embedding does not consider some symmetries of optimization problems, which exist even when we restrict ourselves to the formulation of \eqref{eq:milp-definition}.
For example, if the order of the constraints changes, the instance is considered unaltered, but the feature vector representation will change (permutation of rows of $$\left[ A | \bm{b} \right] $$).
Furthermore, the two resulting feature vectors (with and without the permutation) may be distant to each other (using usual distance metrics), even though they represent the same instance of the problem.

An ideal set of features for an instance should provide enough information for the model to provide the desired output, and also be invariant to the symmetries of the problem.
One way to achieve this is to guarantee that there exists a unique mapping between features and instances.
In other words, we can achieve the ideal features by assuming that the set of instances of interest $$\mathcal{I}$$ is a manifold, and define the features as the _coordinates_ of each instance $$I\in\mathcal{I}$$.
More precisely, one can assume that there exists a one-to-one mapping

$$\begin{align*}
    \pi : \Lambda \subset \R^{\kappa} &\longrightarrow \mathcal{I} \\
    \lambda &\longmapsto I=\pi(\lambda)
,\end{align*}$$

and feed the model with $$\pi^{-1}(I)$$.
<!-- This is easily done when we are interested in _parameterized_ problems, in which case $$\lambda$$ is simply the vector of parameters of the problem. -->
Anderson, Turner and Koch<d-cite key="anderson_generative_2022"></d-cite> exploit the parameterized nature of their problem of interest and define $$\lambda$$ as the vector of parameters.
This way, the mapping $$\pi$$ is easily obtained (and computable).

Feature vectors can also be obtained through feature engineering.
The resulting space of features can be seen as an "approximate" coordinate space.
Multiple works have proposed handcrafted features to represent instances of the problem of interest <d-cite key="alvarez_supervised_2014,alvarez_machine_2017,Khalil2016724,liberto_dash_2016,kruber_learning_2017"></d-cite>.
The engineered features can be designed to be invariant to permutations of the constraints/variables of the problem and also be invariant to problem size, which increases significantly the amount of problems a machine learning model can be applied to.

A novel and widely used architecture is to represent the MILP as a graph and use a Graph Convolutional Network (GCN) as the model<d-cite key="peng_graph_2021,cappart_combinatorial_2022,zhang_survey_2023"></d-cite>.
Given a problem as in \eqref{eq:milp-definition}, it is possible to build a bipartite graph $$G=(V_\textrm{var}\cup V_\textrm{con},E)$$ by considering $$A$$ as an incidence matrix, following the approach of <d-cite key="gasse_exact_2019"></d-cite>.
In other words, $$V_\textrm{var}$$ are nodes associated with variables, thus $$|V_{\textrm{var}}| = n$$; $$V_\textrm{con}$$ are nodes associated with constraints, thus $$|V_{\textrm{con}}|=m$$; and $$E=\{(v_\textrm{var,i},v_\textrm{con,j}) : A_{i,j} \neq 0\}$$ indicate whether a variable is present in a given constraint.
Note that the graph has no ordering of nodes, thus, it is invariant to permutations of constraints and variables.
In fact, if the graph is enhanced with node and edge weights derived from $$A$$, $$\bm{b}$$ and $$\bm{c}$$, and the nodes in $$V_{\rm var}$$ are annotated on whether they represent continuous or binary variables, the graph representation uniquely identifies MILP instances.

### Training

Whathever the inner architecture of the model, the only requirement for the training algorithms is that the model's output is differentiable with respect to the model's parameter.
In our context, we will define end-to-end models as _parameterized_ functions of the form

$$\begin{align*}
f: \mathcal{I}\times \Theta &\longrightarrow Y \\
I,\theta &\longmapsto \bm{y} = f_\theta(I),
\end{align*}$$

in which $$Y\supseteq X$$, where $$X = \left\{ \mathbf{x} \in \mathbb{Z}^{k}\times \mathbb{R}^{n-k} : A\mathbf{x} \ge \mathbf{b} \right\} $$.
The parameter vector $$\theta\in\Theta$$ contains the entirety of the trainable parameters, such as the weights and biases of a neural network.
Note that this definition is such that $$f$$ contains both the _Prep._ and the _DL_ blocks of the diagram in the introduction.
Furthermore, the output of the model is not properly a candidate solution, but rather an approximation, as an integer output would no be differentiable with respect to the parameters, a requirement for gradient-based optimization (e.g., stochastic gradient descent, Adam<d-cite key="kingma_adam_2014"></d-cite>).



- End-to-end models
  - Recap algorithm selection problem
  - end-to-end models imply in algorithms that have as input the instance and as output a predicted optimal solution (note that there might be multiple optimal solution)
  - model architecture (roughly, Embedding MILP instances, from qualification)
    - naive approach = FCN on (A,b,c) tuple
    - feature engineering approach = FCN on f(I), where f may be non-differentiable
    - graph approach = GNN on G(I)
    - text from qualification
    - find references for each approach
  - training
    - (quasi-)optimal as target
    - multi-target
    - weakly-supervised
- Data generation
  - of course, one can resort to historical data that is accurate to the target distribution (instances that will be seen in practice), but assuming that is an overly optimistic scenario
  - as DL models require plenty of data for training, it is more encompassing to assume that instances will need to be generated
  - random generation and feasibility verification
  - generating instances of optimization problems is not a challenge just for learning-based applications, effort has already been put
  - works on instance generation
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
