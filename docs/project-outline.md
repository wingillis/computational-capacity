# Developing a theory underlying the relationship between network topology and computational capacity

## Abstract

Neural networks were conceived to be general function approximators, where a network with a sufficiently large amount of parameters can represent an arbitrarily complex continuous function. However, it is not clear if this is the most robust way of representing continuous functions, nor is it clear how connectivity between "neurons" plays a role in facilitating computation. Here, we explore the relationship between connectivity and computational capacity in small networks, eventually building a predictive theory to generate network architectures to solve arbitrary computations.

## Background

Modern neural networks, such as Transformers, CNNs, and MLPs, are designed as universal function approximators capable of representing any continuous function. Although each network has a different architecture, they have similar inductive biases, meaning they don't impose strong constraints on the types of functions they can approximate well [1]. However, this theoretical power comes with practical challenges. While these models can achieve high accuracy on a wide range of tasks, they often struggle to generalize beyond their training data, and require immense computational resources and complex curricula for training. Recent research highlights limitations in scaling large language models (LLMs) due to increases required for training data size [2], suggesting a need for more efficient approaches.

In contrast, biological neural networks are capable of generalizing well and performing their function efficiently. Animals have evolved specialized circuits - like sound localization in owls and bats or the crayfish's escape response - to generalize well and excel in specific niches without extensive learning. These networks are typically degenerate and operate within lower-dimensional manifolds than their theoretical capacity allows, potentially improving robustness and generalizability. Additional evidence comes from precocial species - those that give birth to offspring that are expected to survive on their own. They develop with neural connectivity patterns that enable them to immediately interact with their environment at birth, likely with significant variability in synaptic weight between individuals. This suggests that connectivity plays an integral role in determining function, rather than relying solely on learnable weightings.

This project draws inspiration from these natural constraints to address the drawbacks of current AI models. We propose exploring classes of neural networks with limited computational capacity but designed for specific tasks through their intrinsic structure and nonlinearities. By focusing on weight-agnostic solutions - networks that solve problems without training - we aim to disentangle the influence of topology from learned weights, leading to more efficient and generalizable architectures. Our approach will involve identifying network properties that facilitate computation under various constraints (e.g., minimal size, quick trainability), ultimately building a predictive theory linking connectivity to computational capacity and relating these findings to principles observed in embodied neural systems.

## Aim 1: develop a framework for assessing network topology

In this aim we develop a framework capable of representing, modifying, and storing information related to network topology. This framework will be the foundation the optimization methods will work upon. The main goal of this framework is to build a representation of a network's connectivity and functionality. We propose the following topological representation:

A network's topology can be represented via three matrices _C_, _M_, and _L_ where _C_ represents the connectivity matrix for the entire network, _M_ represents the modules associated with each node of the network, and _L_ represents the nonlinearities following each node of the network. The size of _C_ is $n \times n$, where $n$ is the number of nodes in the network; the size of _M_ is $n \times m$ where $m$ is the number of unique modules that can represent a node; the size of _L_ is $n \times l$ where $l$ is the number of unique non-linearities that can follow a node.

$C_{i,j}$ has a value of 1 if node $i$ sends the output of its computation to node $j$, and $i \neq j$, and otherwise has a value of 0. Matrices _M_ and _L_ are one-hot encodings of the modules and nonlinearities associated with each node $n$, and are thus sparse.

A purely-feedforward network would be represented by the upper triangle of _C_. In the case of a recurrent network, the lower triangle of _C_ integrates the outputs of each node $n$ at time $t-1$, and the upper triangle of _C_ integrates the outputs of each node $n$ at time $t$.

Possible modules $m$:

- Individual "neuron"
- Single hidden layer MLP
- Single layer CNN
- Transformer block
- LSTM/GRU block
- Other/manually defined?

Possible nonlinearities $l$:

- ReLU
- Leaky ReLU
- GELU
- Sigmoid
- Tanh
- (Log)Softmax
- Softplus
- Softsign
- Sin/Cos

Why this representation is beneficial: by representing connectivity with a set of 3 matrices, it can be possible to build a representational space based on the patterns within these networks. To build this space, we can create a matrix _R_ that is a concatenations of _C_, _M_, and _L_ with size $n \times (n + m + l)$. This makes it easy to build manifold representations of this space to look for underlying structure.

Sampling from the connectivity matrix should place heavy priors on selecting individual neurons over the more complex modules. For starters, we will focus on individual neuron connectivity; therefore, the most important matrices in this Aim are _C_ and _L_.

Network architectures will be sampled and modified to optimize certain objective functions. Separate samplers will be used to sample connectivity, modules, and nonlinearities. Initially, sampling will be uniform with the exception of sampling modules, which will be constrained to individual neurons.

Possible optimization methods:

- NEAT (neuro-evolution via augmenting topology; probably preferred)
- Alternative evolutionary algorithms
- Differentiable methods such as reinforcement learning

Possible objective functions:

- Find weight-agnostic networks capable of solving tasks without being trained. The core goal here is to disentangle the influence of network weights from the influence of topology.
- Find networks that capable of being trained _very_ quickly to solve the task. The core goal here is to find topologies that can facilitate learning.
- Find the smallest (or most energy-efficient) networks capable of solving a task while also generalizing to new similar tasks. The core goal here is to understand how energy/space constraints could influence topological solutions. This might be a necessary constraint to add regardless, reducing the solution space we are studying to be more manageable and interpretable.

Approach: start with the smallest possible network size that allows for variation, where nodes $n = 1$. In an evolutionary algorithm, allow mutations in the following features: number of nodes $n$, nonlinearities in _L_, and connectivity in _C_. When mutating the number of nodes, $n$ can either increase by 1, stay the same, or decrease by 1. Always check if a network configuration has been tried before at least $b$ times. If it has, mutate until finding a novel configuration.

Regarding packages, both `pytorch` and `jax` seem like reasonable choices, with `pytorch` being slightly easier to represent and modify network architecture.

## Aim 2: analyze functional topologies that solve a core set of simple yet unique tasks

Here we use the framework developed in Aim 1 to begin experimenting with architectural solutions to a constrained set of simple problems. We aim to develop a coarse understanding about the relationship between connectivity and capacity. We have selected the following tasks to explore solutions specific problems:

- memory: remember one of two objects presented at time $t=0$, evaluated at time $t=5$. Different distractor objects are presented between time $t\in{1-4}$.
- classification of a positive/negative number
- solving addition/subtraction problems
- pattern recognition, pattern completion

And some agentic problems:

- cart-pole
- breakout-style atari games
- object classification (while only allowing model to attend to a fraction of image at once; actions change location that is being attended to)

Compare generalizability to more expressive networks (i.e., MLPs, transformers) trained on the same data.

Other questions:

- what happens when training a weight-agnostic network to be as bad as possible at the task? Is there a limit to how bad they can be? Are they less bad than a more general network? If so, this suggests a constraint baked into the topology.

## Aim 3: extend research to more complicated tasks

Tasks could include:

- a form of causal inference
- symbolic logic

## Aim 4: extend research to embodied networks

## Aim 5: use data gathered from Aims 2-4 to extract patterns, build predictive models

## Aim 6: synthesize model with modules specialized on subtasks

Does this model perform better than a model with greater computational capacity, but a similar number of parameters?

Does this model perform better or similarly to a larger model trained to perform selected task?

[1]: https://arxiv.org/abs/2502.20237
[2]: https://arxiv.org/abs/2211.04325v2
