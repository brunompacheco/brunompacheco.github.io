---
layout: distill
title: "ML4CO - Part 2: A (less-brief) review of supervised learning models for solution prediction of MILPs"
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

One set of applications with promising results is to train a deep learning model through supervision to predict variable assignments to an instance of an MILP problem.
This *solution prediction* model can be used straight away, as a heuristic itself, or it can be augmented, for example, to warmstart an MILP solver<d-cite key="khalil_mip-gnn_2022"></d-cite>, in a matheuristic.
The building blocks of an MILP (mat)heuristics based on supervised, end-to-end deep learning models are summarized in the diagram below.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/e2e_heuristic.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Building blocks of a (mat)heuristics based on deep learning models trained through supervision to predict solutions.
    The <em>Prep.</em> block represent the pre-processing operations necessary to embed an instance of an MILP problem into a format that can be fed to the deep learning model (<em>DL</em> block).
    The model returns a variable assignment, which can be rounded and directly provide the heuristic output, or can be optimized over (<em>OPT</em> block, with dashed border since it is optional), forming a matheuristic.
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

## Solution prediction models

Solution prediction models are algorithms that take as input an instance of an MILP problem of interest, and provide as output a variable assignment.
Intuitively, the closer the predicted variable assignment is to the optimal solution (in terms of objective value), the better.
However, due to the presence of constraints and the nonlinearity of the integrality requirements, providing high quality assignments becomes a significant challenge.
In this section, I will present the multiple approaches in the literature for developing a model that is capable of predicting optimal variable assignments for MILP instances.
<!-- More specifically, I will focus on the impacts of difference choices for the MILP instance representation, model architecture, and training algorithm. -->

### Model input

<!-- There are clear interdependences between the embedding of the instance (_Prep._ block) and the deep learning model employed (_DL_ block).
However, as the pros and cons can be analysed individually, they will be presented individually. -->

Differently from tabular data, time series, or images, MILP instances are not easily represented by formats suitable for machine learning models.
These models are usually defined over real-valued vectors, commonly called feature vectors.
Therefore, an MILP instance to be fed to a machine learnig model must be embedded into features, which should contain the necessary information for the model to provide the desired output.
<!-- One must describe an MILP instance through numbers and design the programs to handle these numbers in such a way that the outcome (more numbers) represents the desired result. -->

One of the first appraches of feeding optimization problems to machine learning models as proposed by Hopfield and Tank<d-cite key="tank_simple_1986"></d-cite>, who built the feature vectors by vectorizing all values of the LP instances.
Given an MILP instance formulated as in \eqref{eq:milp-definition}, this would be equivalent to vectorizing the tuple $$(A,\bm{b}, \bm{c}, k)$$.
Note that the resulting feature vector contains all information necessary to completely re-assemble the original instance of the problem.
However, this embedding does not consider some symmetries of optimization problems, which exist even when we restrict ourselves to an LP relaxation of \eqref{eq:milp-definition}.
For example, if the order of the constraints changes (permutation of rows of $$\left[ A | \bm{b} \right] $$), the instance is considered unaltered, but the feature vector representation will change.
Furthermore, the two resulting feature vectors (with and without the permutation) may be distant<d-footnote>Considering usual distance metrics.</d-footnote> to each other, even though they represent the same instance of the problem.

One way to ensure that the symmetries of the MILP problem are respected is to guarantee that there exists a unique mapping between feature vectors and problem instances.
Intuitively, the features must be _coordinates_ of each instance $$I$$ in a manifold $$\mathcal{I}$$, which is the set of instances of interest.
More precisely, we desire a one-to-one mapping

$$\begin{align*}
    \pi : \Lambda \subset \R^{\kappa} &\longrightarrow \mathcal{I} \\
    \lambda &\longmapsto I=\pi(\lambda)
,\end{align*}$$

and feed the model with $$\pi^{-1}(I)$$.
<!-- This is easily done when we are interested in _parameterized_ problems, in which case $$\lambda$$ is simply the vector of parameters of the problem. -->
Anderson, Turner and Koch<d-cite key="anderson_generative_2022"></d-cite> exploit the parameterized nature of their problem, i.e., they formulate the optimization problem as a function of "problem-defining parameters"<d-cite key="anderson_generative_2022"></d-cite>, which are then used as the feature fector.
Since the set of instances of interest (within all possible MILP instances) is defined by the problem-defining parameters, the unique mapping is secured.

If the problem-defining parameters are not available (or are not expressive enough), e.g., $$\mathcal{I}$$ comes from historical data, feature vectors can be obtained through feature engineering.
In this case, the resulting space of features can be seen as an "approximate" coordinate space of $$\mathcal{I}$$.
Multiple works have proposed handcrafted features to represent instances of the problem of interest <d-cite key="alvarez_supervised_2014,alvarez_machine_2017,Khalil2016724,liberto_dash_2016,kruber_learning_2017"></d-cite>.
The engineered features can be designed to be invariant to permutations of the constraints/variables of the problem and also be invariant to problem size, which increases significantly the amount of problems a machine learning model can be applied to.

A novel and widely used approach is to represent the MILP as a graph and use a Graph Convolutional Network (GCN) as the model<d-cite key="peng_graph_2021,cappart_combinatorial_2022,zhang_survey_2023"></d-cite>.
Given a problem as in \eqref{eq:milp-definition}, it is possible to build a bipartite graph $$G=(V_\textrm{var}\cup V_\textrm{con},E)$$ by considering $$A$$ as an incidence matrix, following the approach of <d-cite key="gasse_exact_2019"></d-cite>.
In other words, $$V_\textrm{var}$$ are nodes associated with variables, thus $$|V_{\textrm{var}}| = n$$; $$V_\textrm{con}$$ are nodes associated with constraints, thus $$|V_{\textrm{con}}|=m$$; and $$E=\{(v_\textrm{var,i},v_\textrm{con,j}) : A_{i,j} \neq 0\}$$ indicate whether a variable is present in a given constraint.
Note that the graph has no ordering of nodes, thus, it is invariant to permutations of constraints and variables.
In fact, if the graph is enhanced with node and edge weights derived from $$A$$, $$\bm{b}$$ and $$\bm{c}$$, and the nodes in $$V_{\rm var}$$ are annotated on whether they represent continuous or binary variables, the graph representation uniquely identifies MILP instances.

### Model output

The choice of output architecture for deep learning models for solution prediction depends on the specific problem and objectives.
Commonly, the focus is on the integer variables.
Given an assignment for the integer variables, finding optimal values for the continuos variables resumes to solving an LP, which can be done in polynomial time.
However, to use gradient-based training (a cornerstone of deep learning), the output of the model must be differentiable with respect to its parameters.
Therefore, the output of the model cannot be integer values.

Naturally, the model can be designed for a regression task, with a continuous output that contains the possible assignments for the integer variables.
Then, to provide assignments, one can simply round the model's output to the nearest integer value.
More formally, let us define the model as a parameterized function

$$\begin{align*}
f: \mathcal{I}\times \Theta &\longrightarrow \R^k \\
I,\theta &\longmapsto y = f_\theta(I),
\end{align*}$$

where an assignment for the integer variables can be generated from $$\lfloor y \rceil$$.

A "classification" approach, as presented by Ding et al.<d-cite key="ding_accelerating_2020"></d-cite>, is to define the model's ouput as a probability estimate of the variable assignments in the optimal solution for the MILP instances.
For example, let us reformulate \eqref{eq:milp-definition} such that all integer variables are binary<d-footnote>Every MILP can be reformulated using solely binary variables.</d-footnote>.
Let $$\bm{z}\in\{0,1\}^{k}$$ be the vector of binary variables of the problem.
Then, the model is a parameterized function

$$\begin{align*}
f: \mathcal{I}\times \Theta &\longrightarrow [0,1]^k \\
I,\theta &\longmapsto \hat{p}(\bm{z}=\bm{1}|I) = f_\theta(I),
\end{align*}$$

where $$\hat{p}(\bm{z}=\bm{1}|I)\in [0,1]^k$$ is the vector of probability estimates that the binary variables take value 1 in an optimal solution to $$I$$.
The model's output can easily be constrained to the unit interval, e.g., through the sigmoid function.
Furthermore, this approach can be generalized to provide probability estimates to integer variables with any (finite) number of possible values, e.g., through softmax layers.
One advantage of the classification approach is that the probability estimate can be used as a measure of _confidence_ in the predicted assignment for each variable.
This confidence measure can be used to generate partial assignments (e.g., only assigning values if the respective confidence is above a certain threshold) or as a priority order (e.g., for branching).

Further structures of the problem can be exploited in the architecture of the model's output.
For example, to predict solutions for the planar Traveling Salesperson Problem, Vinyals et al. <d-cite key="vinyals_pointer_2015"></d-cite> use attention to provide as the output a permutation of the graph nodes that form the input sequence.
Therefore, it is guaranteed that the model always predicts a valid Hamiltonian path (visits every city exactly once).

### Training

The parameter vector $$\theta$$ can be adjusted through maximum likelihood estimation, using as observed data a set of instance-solution pairs.
Let $$\mathcal{D}=\{(I,\bm{x}^\star)\}$$ be such set of instances and optimal solutions, then the parameters of the model can be estimated through

$$
  \theta^\star = \arg\min_{\theta\in\Theta} \sum_{(I,\bm{x}^\star)}\ell(f_\theta(I), \bm{x}^\star)
$$

where $$\ell:[ 0,1]^k\times \{0,1\}^k \to \R$$ is a loss function (inverse likelihood) such as the binary cross-entropy<d-cite key="ding_accelerating_2020"></d-cite>.

A predicted candidate solution can be obtained from a maximum likelihood estimation of the variables' assignment by simply rounding the output of the model.
Even more, the confidence of the model (max. between $$\hat{p}(\bm{x}=\bm{1}|I)$$ and $$\hat{p}(\bm{x}=\bm{0}|I)$$) can be used to guide the decision to trust or not in the model's output.

Instead of using (quasi-)optimal solutions as targets, one can use multiple feasible solutions as targets for each instance, as Nair et al. <d-cite key="nair_solving_2021"></d-cite> propose.
Then, the training set can be described as $$\mathcal{D}=\{(I,X^\star)\}$$, where $$X^\star\subset \Z^k\times \R^{n-k}$$ represents a set of feasible solutions for $$I$$.
The parameters of the mode are estimated as 

$$
  \theta^\star = \arg\min_{\theta\in\Theta} \sum_{(I,X^\star)\in \mathcal{D}}\sum_{\bm{x}\in X^\star}w_{I,\bm{x}}\ell(f_\theta(I), \bm{x})
,$$

where the weight is 

$$
  w_{I,\bm{x}} = \frac{\exp(c(\bm{x}|I))}{\sum_{\bm{x}'\in X^\star}\exp(c(\bm{x}'|I))}
,$$

where $$c(\bm{x}|I)$$ is the objective value of assignment $$\bm{x}$$ for problem intance $$I$$.
Note that the influence of each feasible solution of an instance in the gradient computation is weighed by its associated objective value, such that solutions with higher objective value will have (exponentially) more impacty.
Furthermore, all instances have equal weight in the gradient estimation ($$\sum_{\bm{x}\in X^\star} w_{I,\bm{x}} = 1, \forall I$$).
Intuitively, this approach would guide the model towards predicting candidate solutions within the feasible region.

A clear obstacle in both training approaches presented above is that they require solving the combinatorial optimization problems to build the training data.
This is not a major problem if historical data is available, as the historical solutions may be used.
However, if new instances are required for training or if the historical solutions are not good enough, then building the training set may require a lot of computational effort.
To circumvent this problem, a weakly-supervised training was proposed and evaluated by Anderson et al.<d-cite key="anderson_generative_2022"></d-cite> and Pacheco et al.<d-cite key="pacheco2023deeplearningbased"></d-cite>.
The authors propose to train a surrogate model 

$$\begin{align*}
g: [0,1]^k \times \mathcal{I}\times \Theta &\longrightarrow \R \\
\hat{p}, I, \theta &\longmapsto \hat{c}(\hat{x} |I) = g_\theta(\hat{p}, I),
\end{align*}$$

to predict the objective value of an assignment $$\hat{x}=\lceil \hat{p} \rfloor






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
