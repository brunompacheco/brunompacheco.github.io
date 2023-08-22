---
layout: distill
title: "Second derivative (and further) of deep equilibrium/implicit models"
date: 2023-01-14
description: A brief study of the techniques for differentiating deep equilibrium/implicit models and an analytic approach for computing higher order derivatives. 
tags: bsc-thesis deep-learning deq implicit-models auto-grad
categories: unpublished-research
giscus_comments: false

authors:
  - name: Bruno M. Pacheco
    # url: "https://en.wikipedia.org/wiki/Albert_Einstein"
    affiliations:
      name: DAS, UFSC

bibliography: DEQDIFF.bib

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
  - name: The Jacobian of a DEQ
    subsections:
    - name: Computing gradients through DEQs
  - name: Second-order derivatives of DEQs
    subsections:
    - name: Computing gradients through derivatives of DEQs
  - name: Hessian of DEQs
    subsections:
    - name: Computing gradients through Jacobians of DEQs

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

My Bachelor's dissertation was on putting together two novelties from the machine learning community: physics-informed training and deep equilibrium models <d-cite key="pacheco_pideq_2022"></d-cite>.
Physics-informed training, or physics-informed neural networks (PINNs), is a regularization technique that adds to the loss function a gradient regularization term based on differential equations <d-cite key="karniadakis_physics"></d-cite>.
Deep equilibrium models (DEQs) are infinite-depth models with all layers sharing the same weights<d-footnote>Ghaoui et al.<d-cite key="Ghaoui2019"></d-cite> proposed almost identical models naming them "implicit models", but we will stick to DEQs throughout the paper.</d-footnote>.

Physics-informed DEQs are not trivially implemented, as the computation of second derivative of DEQs can be highly inefficient.
DEQs require us to solve fixed-point equations to compute the output (forward pass) and the gradients (backward pass).
Solving fixed-point equations can be done efficiently through, e.g., quasi-Newton methods, but this rules out the use of automatic differentiation (auto-diff) to compute the second derivatives necessary for physics-informed training.
Therefore, the straightforwad approach is to solve the backward pass' fixed-point equation iteratively and then use auto-diff to unroll the iterations, which can have high computational costs.

<!-- Physics-informed training, or physics-informed neural networks (PINNs), is a regularization technique to teach deep learning models to respect differential equations <d-cite key="karniadakis_physics"></d-cite>.
PINNs are very helpful for applications on dynamical systems <d-cite key="Antonelo2021"></d-cite>, as the physics regularization reduces the amount of training data needed.
Deep equilibrium models (DEQs) are infinite-depth models with all layers sharing the same weights<d-footnote>Ghaoui et al.<d-cite key="Ghaoui2019"></d-cite> proposed almost identical models naming them "implicit models", but we will stick to DEQs throughout the paper.</d-footnote>.
DEQs have shown impressive results at a low parameter 
The works by Bai et al.<d-cite key="Bai2019"></d-cite> and Ghaoui et al.<d-cite key="Ghaoui2019"></d-cite> have enabled the use of big DEQs by applying the implicit function theorem to compute the gradients instead of using automatic differentiation through the forward pass.

PINNs regularize the model through gradient regularization.
More precisely, the loss function considers the difference of the model's derivative and the expected derivative (that comes from differential equations) at random (collocation) points in the domain.
As the training uses the gradient descent to optimize the model's weights, PINNs require the computation of (at least) second-order derivatives of the deep learning model.
For traditional neural networks, this is computed using automatic differentiation through the backward pass.

Differentiating through the backward pass of DEQs is not directly possible.
The forward pass of DEQs can be seen as solving a fixed-point equation, which can be done efficiently by using, e.g., quasi-Newton methods.
Using such methods requires us to give up on automatic differentiation.
Thus, to compute the backward pass of DEQs, the implicit function theorem allows us to solve another fixed-point equation.
Similar to the forward pass, the backward pass can be computed efficiently by using, e.g., quasi-Newton methods, but this impossibilitates the computation of higher order derivatives through automatic differentiation.
For the BSc. Dissertation, I had to iteratively solve the backward pass so that the second derivative can be computed by unrolling the operations (automatic differentiation) <d-cite key="pacheco_pideq_2022"></d-cite>.
Despite effective, this is highly inefficient, as ignores one of the major characteristics of DEQs which is to efficiently compute the output of an infinite-depth model. -->

In theory, the implicit function theorem can be applied to the backward pass and any other computation of derivatives of DEQs. 
This could enable the computation of higher order derivatives by solving fixed-point equations, instead of relying in automatic differentiation.
In the following, I present my theoretical development on the subject, starting on the computation of the derivatives of DEQs as proposed by Bai et al.<d-cite key="Bai2019"></d-cite> and Ghaoui et al.<d-cite key="Ghaoui2019"></d-cite>, and going up to an analytical solution to computing the Hessian of DEQs.
Be aware: the following is untested work, so use at you own risk<d-footnote>and remember to cite me ;)</d-footnote>.

## The Jacobian of a DEQ

Fundamental for learning through gradient descent (the cornerstone of deep learning) is the differentiation of a model's output with respect to its parameters.
The entirety of this section is based on the works by Bai et al.<d-cite key="Bai2019"></d-cite> and Ghaoui et al.<d-cite key="Ghaoui2019"></d-cite>.

Let

$$
\newcommand{\R}{\mathbb{R}}  
\newcommand{\Z}{\mathbb{Z}}
\newcommand{\N}{\mathbb{N}}
\newcommand{\Q}{\mathbb{Q}}
\newcommand{\vtheta}{\boldsymbol\theta}
\newcommand\bm[1]{\boldsymbol{#1}}
\begin{equation}\label{eq:layer-definition}\begin{split}
  \bm{f}:\R^{n}\times\mathbb{R}^{m}&\longrightarrow \mathbb{R}^{m} \\
  \bm{x},\bm{z} &\longmapsto \bm{z} = \bm{f}(\bm{x},\bm{z})
\end{split}\end{equation}$$

be the layer function of a DEQ, where $$\bm{x}$$ is both the inputs and parameters of the model and $$\bm{z}$$ is the hidden state.
The output of a DEQ is the equilibrium point of $$\bm{f}$$ in $$\bm{z}$$.
Formally, an equilibrium of $$\bm{f}$$ given $$\bm{x}\in \R^n$$ is a vector $$\bm{z}\in\R^m$$ such that $$\bm{z}=\bm{f}(\bm{x},\bm{z})$$.
Then, let $$\bm{z}^*:\R^n\to\R^m$$ be the function of the equilibrium of $$f$$, that is,  $$\bm{z}^*(\bm{x})=\bm{f}(\bm{x},\bm{z}^*(\bm{x}))$$, whenever the equilibrium exists.

Note that $$\bm{z}^*(\bm{x})$$ is precisely the output of the DEQ.
Therefore, to compute $$\bm{z}^*(\bm{x})$$ we can use any root-finding algorithm.
However, to train the DEQ using gradient descent, we need

$$
J_{\bm{z}^*} = \left[ \frac{\partial z^*_i}{\partial x_j} \right]_{\substack{i=1,\ldots,m \\ j=1,\ldots,n}}
,$$

the gradients of the DEQs output with respect to the inputs.

Let $$\bm{f}^\star(\bm{x},\bm{z}) = \bm{f}(\bm{x},\bm{z}) - \bm{z}$$.
Naturally, $$\bm{z}^*(\cdot)$$ is a parametrization of $$\bm{z}$$ on $$\bm{f}^\star(\bm{x},\bm{z})=0$$, i.e., $$\bm{f}^\star(\bm{x},\bm{z}^*(\bm{x}))=0$$.

By the _implicit function theorem_, we have

$$
J_{\bm{z}^*}(\bm{x}) = -\left[ J_{\bm{f}^\star,\bm{z}}(\bm{x},\bm{z}^*(\bm{x})) \right]^{-1} J_{\bm{f}^\star,\bm{x}}(\bm{x},\bm{z}^*(\bm{x}))
,$$

where

$$
J_{\bm{f}^\star,\bm{x}}(\bm{x},\bm{z}) = \left[ \frac{\partial f^\star_i}{\partial x_j}(\bm{x},\bm{z}) \right],\,J_{\bm{f}^\star,\bm{z}}(\bm{x},\bm{z}) = \left[ \frac{\partial f^\star_i}{\partial z_j}(\bm{x},\bm{z}) \right]
.$$

But,

$$
\frac{\partial f^\star_i}{\partial x_j}(\bm{x},\bm{z}) = \frac{\partial f_i}{\partial x_j}(\bm{x},\bm{z})
$$

and

$$
\frac{\partial f^\star_i}{\partial z_j}(\bm{x},\bm{z}) = \frac{\partial f_i}{\partial z_j}(\bm{x},\bm{z}) - 1
.$$

Therefore, we can rewrite

$$
\begin{equation}\label{eq:deq-jacobian}
J_{\bm{z}^*}(\bm{x}) = -\left[ J_{\bm{f},\bm{z}}(\bm{x},\bm{z}^*(\bm{x})) - I \right]^{-1} J_{\bm{f},\bm{x}}(\bm{x},\bm{z}^*(\bm{x}))
,\end{equation}$$

which is an analytical way to compute the derivatives we desire, without the need for accumulating the operations throughout the forward pass as done by automatic differentiation.
In other words, we have a way to compute the derivatives which is independent of the computations in the forward pass.

### Computing gradients through DEQs

The problem of \eqref{eq:deq-jacobian} is that it requires a matrix inversion to determine the Jacobian of our DEQ.
Luckily, in gradient descent optimization, we just need to compute the gradient of the loss function with respect to the parameters.
Through the chain rule, this gradient computation can be perform through a series of vector-Jacobian products (vJps).
In other words, we don't actually need to compute the Jacobian itself, just the results of its product with a given vector.

To better visualize this, consider that $$\bm{z}^*(\bm{x})$$ is part of a network

$$\begin{equation}\label{eq:net-definition}\begin{split}
  NN:\R^{n'}&\longrightarrow \mathbb{R}^{m'} \\
  \bm{x}' &\longmapsto \bm{y} = NN(\bm{x}') = \bm{n}_{post}(\bm{z}^*(\bm{n}_{prior}(\bm{x}')))
\end{split},\end{equation}$$

where $$\bm{n}_{post}:\R^m\to\R^{m'}$$ and $$\bm{n}_{prior}:\R^{n'}\to\R^n$$ are two "easily" differentiable functions.
Let $$\ell:\R^{m'}\to\R$$ be a loss function.
For gradient-descent methods, we need to compute $$\nabla (\ell \circ NN)(\bm{x}')$$<d-footnote>The composition $$(\ell\circ NN)$$ is a scalar-valued function, and, thus, this is properly a gradient computation, rather than a Jacobian.</d-footnote>.
Through the chain rule, we have

$$\begin{align*}
\nabla(l \circ NN)(\bm{x}') &= \left[\nabla l(\bm{y})\right]_{1\times m'} \left[J_{\bm{n}_{post}}(\bm{z})\right]_{m'\times m} \left[J_{\bm{z}^*}(\bm{x})\right]_{m\times n} \left[J_{\bm{n}_{prior}}(\bm{x}')\right]_{n \times n'} \\
&= -\left[\nabla l(\bm{y})\right]_{1\times m'} \left[J_{\bm{n}_{post}}(\bm{z})\right]_{m'\times m} \left[ J_{\bm{f},\bm{z}}(\bm{x},\bm{z}^*(\bm{x})) - I \right]^{-1}_{m\times m} \left[J_{\bm{f},\bm{x}}(\bm{x},\bm{z}^*(\bm{x}))\right]_{m\times n} \left[J_{\bm{n}_{prior}}(\bm{x}')\right]_{n \times n'}
.\end{align*}$$

Running backwards (from left to right) in the matrix products necessary to compute the gradient, it is easy to see that we can compute the whole equation through a series of vJps of the form $$\bm{u}^T J$$.
Let $$\bm{u}^T = \nabla l(\bm{y}) J_{\bm{n}_{post}}(\bm{z})$$.
Then, the "next step" in the gradient computation would be

$$\begin{equation}\label{eq:deq-backward-fixed-point}\begin{split}
    \bm{g}^T &= -\bm{u}^T \left[ J_{\bm{f},\bm{z}}(\bm{x},\bm{z}^*(\bm{x})) - I \right]^{-1} \\
    &= \bm{g}^T J_{\bm{f},\bm{z}}(\bm{x},\bm{z}^*(\bm{x})) + \bm{u}^T
,\end{split}\end{equation}$$

which is a linear fixed-point equation and, therefore, can be solved using any root-finding algorithm.
In comparison, this avoids the computation of the complete $$J_{\bm{f},\bm{z}}(\bm{x},\bm{z}^*(\bm{x}))$$, as this is a vJp as well, and avoids the matrix inversion $$\left[ J_{\bm{f},\bm{z}}(\bm{x},\bm{z}^*(\bm{x})) - I \right]^{-1}$$, which can be costly for large $$m$$.

## Second-order derivatives of DEQs

In the previous section, we have seen that the backward pass of a DEQ can be performed throguh \eqref{eq:deq-jacobian}, which allows us to compute the forward pass efficiently.
More precisely, we have avoided the problem of unrolling the operations in the forward pass of a DEQ (which is necessary for automatic differentiation) by computing the derivatives of a DEQ implicitly.
This, however, implies that unrolling the operations of the backward pass becomes costly, as it also requires a root-finding operation.
Therefore, computing its second derivative (and any higher-order derivatives) becomes, in the best case, computationaly expensive.

Initially, we will explore the case of gradient regularization with respect to a single variable, which is a common case when training neural networks with physics-informed regularization, e.g., PINNs to solve differential equations with respect to time.
Let $$\bm{f}(t,\bm{x},\bm{z}):\R^{1+n+m}\to \R^{m}$$ and $$\bm{z}^*:\R^{n+1}\to\R^m$$ similar to \eqref{eq:layer-definition}, but with $$t$$ explicit.
We want to compute the second-derivatives

$$
H_{\bm{z}^*,t\bm{x}} \triangleq \frac{\partial J_{\bm{z}^*,t}}{\partial \bm{x}} = \left[ \frac{\partial^2 z^*_i}{\partial t \partial x_j} \right]_{\substack{i=1,\ldots,m \\ j=1,\ldots,n}}
.$$

First, we tackle

$$
  H_{\bm{z}^*,t x_k} = \frac{\partial J_{\bm{z}^*,t}}{\partial x_k}
.$$

By \eqref{eq:deq-jacobian}, we can write 

$$
  J_{\bm{z}^*,t}(t,\bm{x}) = -\left[ J_{\bm{f},\bm{z}}(t, \bm{x},\bm{z}^*(t,\bm{x})) - I \right]^{-1} J_{\bm{f},t}(t,\bm{x},\bm{z}^*(t,\bm{x}))
.$$

Then, through some matrix calculus identities, we see that

$$\begin{align*}
    H_{\bm{z}^*,t x_k} &= \frac{\partial J_{\bm{z}^*,t}}{\partial x_k} \\
    &= -(J_{\bm{f},\bm{z}}-I)^{-1}\frac{\partial J_{\bm{f},t}}{\partial x_k} - \frac{\partial (J_{\bm{f},\bm{z}}-I)^{-1}}{\partial x_k} J_{\bm{f},t} \\
    &= -(J_{\bm{f},\bm{z}}-I)^{-1}\frac{\partial J_{\bm{f},t}}{\partial x_k} + (J_{\bm{f},\bm{z}}-I)^{-1}\frac{\partial (J_{\bm{f},\bm{z}}-I )}{\partial x_k}(J_{\bm{f},\bm{z}}-I)^{-1}J_{\bm{f},t} \\
    &= -(J_{\bm{f},\bm{z}}-I)^{-1}\frac{\partial J_{\bm{f},t}}{\partial x_k} + (J_{\bm{f},\bm{z}}-I)^{-1}\frac{\partial J_{\bm{f},\bm{z}}}{\partial x_k}(J_{\bm{f},\bm{z}}-I)^{-1}J_{\bm{f},t}
.\end{align*}$$

To ease the notation, we will write

$$
H_{\bm{f},tx_k} \triangleq \frac{\partial J_{\bm{f},t}}{\partial x_k} = \left[ \frac{\partial^2 f_i}{\partial t \partial x_k} \right]_{i=1,\ldots,m}
$$

and

$$
H_{\bm{f},\bm{z}x_k} \triangleq \frac{\partial J_{\bm{f},\bm{z}}}{\partial x_k} = \left[ \frac{\partial^2 f_i}{\partial z_j \partial x_k} \right]_{\substack{i=1,\ldots,m\\j=1,\ldots,m}}
,$$

and recall that $$ (J_{\bm{f},\bm{z}}-I)^{-1}J_{\bm{f},t} = J_{\bm{z}^*,t} $$.
Therefore, 

$$\begin{align*}
    H_{\bm{z}^*,tx_k} &= -\left[ (J_{\bm{f},\bm{z}}-I)^{-1} \right]_{m\times m} \left[ H_{\bm{f},tx_k} \right]_{m\times 1} + \left[ (J_{\bm{f},\bm{z}}-I)^{-1} \right]_{m\times m} \left[ H_{\bm{f},\bm{z}x_k} \right]_{m\times m} \left[ J_{\bm{z}^*,t} \right]_{m\times 1} \\
    &= -(J_{\bm{f},\bm{z}}-I)^{-1}\left[ H_{\bm{f},tx_k} - H_{\bm{f},\bm{z}x_k}J_{\bm{z}^*,t} \right]
.\end{align*}$$

We can "stack" over $$ x_k $$ and see that

$$\begin{equation}\label{eq:hessian-t}
H_{\bm{z}^*,t\bm{x}} = -(J_{\bm{f},\bm{z}}-I)^{-1}\left[ H_{\bm{f},t\bm{x}} - \mathbf{H}_{\bm{f},\bm{z}\bm{x}}\times_2 J_{\bm{z}^*,t}^T \right]
,\end{equation}$$

where

$$
  \mathbf{H}_{\bm{f},\bm{z}\bm{x}} \triangleq \frac{\partial J_{\bm{f},\bm{z}}}{\partial x_k} = \left[ \frac{\partial^2 f_i}{\partial z_j \partial x_k} \right]_{\substack{i=1,\ldots,m\\j=1,\ldots,m\\k=1,\ldots,n}}
$$

is a tensor with the partial derivatives and $$ \times_2 $$ is the 2-mode tensor-matrix product, of which the result (an $$m$$-by-$$1$$-by-$$n$$ tensor) is already "squeezed" into an $$m$$-by-$$n$$ matrix.

### Computing gradients through derivatives of DEQs

Consider now that, similar to \eqref{eq:net-definition}, $\bm{z}^*(t,\bm{x})$ is part of a network $$NN:\R^{n+1}\to\R^{m'}$$ such that the output is

$$
\bm{y} = \bm{n}_{post}(\bm{z}^*(t,\bm{x}))
,$$

where $$\bm{n}_{post}:\R^m\to\R^{m'}$$ is a differentiable functions.
Let

$$\begin{align*}
  \ell:\R^{1\times m'}&\longrightarrow\R \\
  J_{NN,t}(t,\bm{x}) &\longmapsto (\ell \circ J_{NN,t})(t,\bm{x})
\end{align*}$$

be a loss function on the _Jacobian_ of the network with respect to a single variable $$t$$.

We need $$\nabla_{\bm{x}} (\ell \circ J_{NN,t})(t,\bm{x})$$ to use gradient-descent methods to train the model.
By the chain rule,

$$
\nabla_{\bm{x}} (\ell \circ J_{NN,t}) = \left[ \nabla \ell \right]_{1\times m'}  \left[\frac{\partial J_{NN,t}}{\partial \bm{x}}\right]_{m' \times n} = \left[ \nabla \ell \right]_{1\times m'}  \left[ H_{NN,t\bm{x}} \right]_{m' \times n}
$$

We assume that $$\nabla \ell$$ is easily computable, so the challenge lies in computing $$H_{NN,t\bm{x}}$$.
Following the network definition and using the chain rule, we have

$$\begin{align*}
    J_{NN,t} &= J_{\bm{n}_{post}} J_{\bm{z}^*,t} \\
    &= - J_{\bm{n}_{post}} \left[ J_{\bm{f},\bm{z}} - I \right]^{-1} J_{\bm{f},t}
,\end{align*}$$

i.e., for each output of the network,

$$
\frac{\partial NN_{i'}}{\partial t} = J_{n_{post}^{(i')},\bm{z}} J_{\bm{z}^*,t},\,i'=1,\ldots,m'
.$$

Thus, following the same approach of the previous section,

$$\begin{align*}
H_{NN_{i'},tx_j} &\triangleq\frac{\partial^2 NN_{i'}}{\partial t\partial x_j} = \frac{\partial J_{n_{post}^{(i')},\bm{z}}}{\partial x_j} J_{\bm{z}^*,t} + J_{n_{post}^{(i')},\bm{z}} \frac{\partial J_{\bm{z}^*,t}}{\partial x_j} \\
&= H_{n_{post}^{(i')},\bm{z}x_j} J_{\bm{z}^*,t} + J_{n_{post}^{(i')},\bm{z}} H_{\bm{z}^*,tx_j} \\
\implies H_{NN,tx_j} &= H_{\bm{n}_{post},\bm{z}x_j} J_{\bm{z}^*,t} + J_{\bm{n}_{post},\bm{z}} H_{\bm{z}^*,tx_j} \\
\implies H_{NN,t\bm{x}} &= H_{\bm{n}_{post},\bm{z}\bm{x}} \times_2 J_{\bm{z}^*,t}^T + J_{\bm{n}_{post},\bm{z}} H_{\bm{z}^*,t\bm{x}}
.\end{align*}$$


Therefore, by \eqref{eq:hessian-t}, we have

$$\begin{align*}
\nabla_{\bm{x}} (\ell \circ J_{NN,t}) &= \nabla \ell \left[ H_{\bm{n}_{post},\bm{z}\bm{x}} \times_2 J_{\bm{z}^*,t}^T + J_{\bm{n}_{post},\bm{z}} H_{\bm{z}^*,t\bm{x}} \right] \\
&= \nabla \ell \left[ H_{\bm{n}_{post},\bm{z}\bm{x}} \times_2 J_{\bm{z}^*,t}^T - J_{\bm{n}_{post},\bm{z}}(J_{\bm{f},\bm{z}}-I)^{-1}\left[ H_{\bm{f},t\bm{x}} - \mathbf{H}_{\bm{f},\bm{z}\bm{x}}\times_2 J_{\bm{z}^*,t}^T \right] \right]
,\end{align*}$$

which can be computed through a series of vJps.

To compute the vJp $$\bm{u}^T (H_{\bm{n}_{post},\bm{z}\bm{x}} \times_2 J_{\bm{z}^*,t}^T)$$, where $$\bm{u}^T = \nabla\ell$$, we first note that

$$
  \bm{u}^T \left( H_{\bm{n}_{post},\bm{z}\bm{x}} \times_2 J_{\bm{z}^*,t}^T\right) = J_{\bm{z}^*,t}^T \left( H_{\bm{n}_{post},\bm{z}\bm{x}} \times_1 \bm{u}^T \right)
.$$

The product $$G = H_{\bm{n}_{post},\bm{z}\bm{x}} \times_1 \bm{u}^T$$ is assumed to be easily computed through automatic differentiation as $$\bm{n}_{post}$$ has a differentiable forward pass.
Then, to compute $$J_{\bm{z}^*,t}^T G$$ we see that

$$\begin{equation}\label{eq:G-computation}
    G^T J_{\bm{z}^*,t} = G^T (J_{\bm{f},\bm{z}}-I)^{-1} J_{\bm{f},t}
,\end{equation}$$

which can be computed as a series of fixed-point solutions, as each row of $$G^T$$ can be computed as in \eqref{eq:deq-backward-fixed-point}, followed by vJps with $$J_{\bm{f},t}$$.

Now, for the vJp

$$\bm{u}^T J_{\bm{n}_{post},\bm{z}} (J_{\bm{f},\bm{z}}-I)^{-1}\left[ H_{\bm{f},t\bm{x}} - H_{\bm{f},\bm{z}\bm{x}}\times_2 J_{\bm{z}^*,t}^T \right]
,$$

we first recall that the vJps $$\bm{v}^T = \bm{u}^TJ_{\bm{n}_{post},\bm{z}}(J_{\bm{f},\bm{z}}-I)^{-1}$$ can be computed following what was stablished in the previous section.
Furthermore, $$\bm{v}^TH_{\bm{f},t\bm{x}}$$ can be computed through automatic differentiation as the forward pass of $$\bm{f}$$ is differentiable.
Finally,

$$\begin{align*}
    -\bm{v}^T \left( H_{\bm{f},\bm{z}\bm{x}}\times_2 J_{\bm{z}^*,t}^T \right) &= -J_{\bm{z}^*,t}^T \left( H_{\bm{f},\bm{z}\bm{x}}\times_1 \bm{v}^T \right) \\
    &= -J_{\bm{z}^*,t}^T V
,\end{align*}$$

as $$V=H_{\bm{f},\bm{z}\bm{x}}\times_1 g^T$$ can be computed by the definition of $$f$$.
Then, $$-J_{\bm{z}^*,t}^T V$$ can be computed just as done for $$G$$ in \eqref{eq:G-computation}, solving a fixed-point equation.

In summary, to compute the gradients of a loss over the Jacobian (on a single variable) of a DEQ, we need to implement the vJps to compute $$\bm{u}^TH_{\bm{z}^*,t\bm{x}}$$ following equation \eqref{eq:hessian-t}, which also results in solving fixed-point problems.

## Hessian of DEQs

- Generalization of second-order

### Computing gradients through Jacobians of DEQs
