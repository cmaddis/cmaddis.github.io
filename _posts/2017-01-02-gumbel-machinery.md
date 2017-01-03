---
layout: post
title: Gumbel Machinery
author: Chris J. Maddison and Danny Tarlow
permalink: gumbel-machinery
comments: true
---

Recently, Laurent Dinh wrote a great [blogpost](https://laurent-dinh.github.io/2016/11/22/gumbel-max.html) asking (and answering) whether it was possible to invert the Gumbel-Max trick: given a sample from a discrete random variable, can we sample from the Gumbels that produced it?

We thought it would be valuable to show an alternative approach to Laurent's question. Taking this tack we can prove the Gumbel-Max trick, answer Laurent's question, and more, all in a few short lines. All together this results in four central properties that form a sort of Gumbel machinery of intuitions, which were indispensable during the development of our NIPS 2014 paper [A* Sampling](https://arxiv.org/abs/1411.0030).

Implemented in Python, the Gumbel-Max trick looks like this:
{% highlight python %}
alpha = np.array([1.0, 4.0, 6.0, 4.0, 1.0])
uniform = np.random.rand(5)
gumbels = -np.log(-np.log(uniform)) + np.log(alpha)
K = np.argmax(gumbels)
{% endhighlight %}
The trick is to show that `K` sampled in such a way produces a sample from the discrete distribution proportional to `alpha[i]`. In mathematical notation,

$$\mathbb{P}(K = i) = \frac{\alpha_i}{Z}$$

where $$\alpha_i > 0$$ and $$Z = \sum_i \alpha_i$$. We will get to the proof, but for now Laurent's question was simply, given `K`, what is the distribution over `gumbels`?

## The Gumbels

The apparently arbitrary choice of noise in the Gumbel-Max trick is its namesake; If $$U \sim \mathrm{uniform}[0,1]$$, then

$$-\log(-\log U) + \log \alpha$$ 

has a [Gumbel distribution](https://en.wikipedia.org/wiki/Gumbel_distribution) with location $$\log \alpha$$. The cumulative distribution function (CDF) of a Gumbel is

$$F_{\log \alpha}(g) = \exp(-\exp(-g + \log \alpha)).$$

The derivative of this is the density of the Gumbel,

$$f_{\log \alpha}(g) = \exp(-g + \log \alpha) F_{\log \alpha}(g).$$

Thus, the joint density of `gumbels` is

$$p(g_1, \ldots, g_n) = \prod_{i=1}^n f_{\log \alpha_i}(g_i)$$

#### Product of Gumbel CDFs.

The keystone for any understanding of Gumbels is that multiplying their CDFs accumulates the $$\alpha_i$$ parameters.

$$F_{\log \alpha_0}(g)F_{\log \alpha_1}(g) = F_{\log(\alpha_0 + \alpha_1)}(g)$$

The derivation of this property is just some simple algebra,

$$ \begin{align*}
F_{\log \alpha_0}(g)F_{\log \alpha_1}(g) &= \exp(-\exp(-g)\alpha_0 -\exp(-g)\alpha_1) \\
&= \exp(-\exp(-g)(\alpha_0 + \alpha_1)) \\
&= F_{\log (\alpha_0 + \alpha_1)}(g).
\end{align*}$$

## The Joint

Our strategy will be to write down the joint distribution of the `gumbels` and `K` in an intuitive form, then to manipulate it to reveal a structure like

$$p(K, g_1, \ldots, g_n) = p(K)p(g_K) \prod_{i \neq K} p(g_i | K, g_K),$$

which answers Laurent's question. This section is a bit tedious, but it only needs to be done once before getting at the valuable properties of Gumbels.


Knowing `K` restricts the possible Gumbel events to ones in which `gumbels[K] > gumbels[i]` for `i != K`. So, intuitively the joint of `gumbels` and `K` is

$$p(K, g_1, \ldots, g_n) = \prod_{i=1}^n f_{\log \alpha_i}(g_i) [g_K \geq g_i]$$

where $$K$$ is one of $$\{1, \ldots, n\}$$ and $$[A]$$ is the [Iverson bracket notation](https://en.wikipedia.org/wiki/Iverson_bracket) for the indicator function of set $$A$$. If you're still not convinced, then sum over $$K$$ or integrate over $$g_1, \ldots, g_n$$. In both cases you will get the correct marginal events.

First, we multiply by a judicious choice of 1,

$$\require{color}
\begin{align*}
= { \color{red} \frac{F_{\log \sum_{i \neq K} \alpha_i}(g_K)}{F_{\log \sum_{i \neq K} \alpha_i}(g_K)}}\prod_{i=1}^n f_{\log \alpha_i}(g_i) [g_K \geq g_i],
\end{align*}$$

pull the density of `gumbels[K]` out of the product,

$$\require{color}
\begin{align*}
= \frac{ {\color{red} f_{\log \alpha_K}(g_K)} F_{\log \sum_{i \neq K} \alpha_i}(g_K)}{F_{\log \sum_{i \neq K} \alpha_i}(g_K)} \prod_{\color{red} i \neq K} f_{\log \alpha_i}(g_i) [g_K \geq g_i],
\end{align*}$$

apply the product of Gumbel CDFs property in reverse,

$$\require{color}
\begin{align*}
= \frac{f_{\log \alpha_K}(g_K) F_{\log \sum_{i \neq K} \alpha_i}(g_K)}{ { \color{red} \prod_{i \neq K} F_{\log \alpha_i}(g_K)}}\prod_{i \neq K} f_{\log \alpha_i}(g_i) [g_K \geq g_i],
\end{align*}$$

distribute into the product,

$$\require{color}
\begin{align*}
= f_{\log \alpha_K}(g_K) F_{\log \sum_{i \neq K} \alpha_i}(g_K)\prod_{i \neq K} \frac{f_{\log \alpha_i}(g_i) [g_K \geq g_i]}{\color{red} F_{\log \alpha_i}(g_K)},
\end{align*}$$

expand the density of `gumbels[K]` and apply the product of Gumbel CDFs,

$$\require{color}
\begin{align*}
&= {\color{red} \exp(-g_K + \log \alpha_K)F_{\log \alpha_K}(g_K)} F_{\log \sum_{i \neq K} \alpha_i}(g_K)\prod_{i \neq K} \frac{f_{\log \alpha_i}(g_i) [g_K \geq g_i]}{F_{\log \alpha_i}(g_K)} \\
&= \exp(-g_K + \log \alpha_K){\color{red} F_{\log Z}(g_K)}\prod_{i \neq K} \frac{f_{\log \alpha_i}(g_i) [g_K \geq g_i]}{F_{\log \alpha_i}(g_K)},
\end{align*}$$

finally, multiply by $$\exp(\log Z - \log Z)$$ to get:

$$\require{color}
\begin{align*}
p(K, g_1, \ldots, g_n) = \frac{\alpha_K}{Z}f_{\log Z}(g_K)\prod_{i \neq K} \frac{f_{\log \alpha_i}(g_i) [g_K \geq g_i]}{F_{\log \alpha_i}(g_K)}.
\end{align*}$$

## Four Important Properties of Gumbels

The value of that tedious algebra is what it reveals.
We can now simply read off the following properties from our form of the joint density of `K` and `gumbels`. Refering back to the code
{% highlight python %}
alpha = np.array([1.0, 4.0, 6.0, 4.0, 1.0])
uniform = np.random.rand(5)
gumbels = -np.log(-np.log(uniform)) + np.log(alpha)
K = np.argmax(gumbels)
{% endhighlight %}

#### 1. Gumbel-Max Trick.

`K` is distributed as $$\mathbb{P}(K = k) = \alpha_k / Z$$, since

$${\color{red} \underbrace{\frac{\alpha_K}{Z}}_{p(K)}} f_{\log Z}(g_K) \prod_{i \neq K} \frac{f_{\log \alpha_i}(g_i) [g_K \geq g_i]}{F_{\log \alpha_i}(g_K)}$$

#### 2. The max Gumbel integrates over alphas.

`gumbels[K]` is distributed as a Gumbel with location $$\log Z$$ where $$Z = \sum_i \alpha_i$$, since

$$\frac{\alpha_K}{Z} {\color{red} \underbrace{ f_{\log Z}(g_K)}_{p(g_K)}} \prod_{i \neq K} \frac{f_{\log \alpha_i}(g_i) [g_K \geq g_i]}{F_{\log \alpha_i}(g_K)}$$

#### 3. Argmax and max are independent. 

`K` and `gumbels[K]` are independent,

$${\color{red} \underbrace{\frac{\alpha_K}{Z} f_{\log Z}(g_K)}_{p(K)p(g_K)}} \prod_{i \neq K} \frac{f_{\log \alpha_i}(g_i) [g_K \geq g_i]}{F_{\log \alpha_i}(g_K)}$$

#### 4. The remaining Gumbels are still independent but truncated.

Given `K` and `gumbels[K]` consider 
{% highlight python %}
  remaining_gumbels = gumbels[:K] + gumbels[K+1:]
  remaining_alpha = alpha[:K] + alpha[K+1:]{% endhighlight %}
The remaining Gumbels `remaining_gumbels[i]` are independent Gumbels with location `remaining_alpha[i]` 
truncated at `gumbels[K]`, since

$$\frac{\alpha_K}{Z} f_{\log Z}(g_K) \prod_{i \neq K} {\color{red}  \underbrace{ \frac{f_{\log \alpha_i}(g_i) [g_K \geq g_i]}{F_{\log \alpha_i}(g_K)}}_{p(g_i | K, \ g_K)} }$$

## Laurent's Question and Beyond

Reading off the density, we also get a simple answer to Laurent's question. To sample from 
$$p(g_1 \ldots, g_n | K)$$,
sample the top Gumbel $$g_K$$ with location $$\log Z$$ and for $$i \neq K$$ sample truncated Gumbels with locations $$\log \alpha_i$$. This code samples a truncated Gumbel,

{% highlight python %}
def truncated_gumbel(alpha, truncation):
    gumbel = np.random.gumbel() + np.log(alpha)
    return -np.log(np.exp(-gumbel) + np.exp(-truncation))
{% endhighlight %}

This code samples from the desired posterior,

{% highlight python %}
def topdown(alphas, k):
    topgumbel = np.random.gumbel() + np.log(sum(alphas))
    gumbels = []
    for i in range(len(alphas)):
        if i == k:
            gumbel = topgumbel
        else:
            gumbel = truncated_gumbel(alphas[i], topgumbel)
        gumbels.append(gumbel)
    return gumbels
{% endhighlight %}

For reference, here is the rejection version

{% highlight python %}
def rejection(alphas, k):
    log_alphas = np.log(alphas)
    gumbels = np.random.gumbel(size=len(alphas))
    while k != np.argmax(gumbels + log_alphas):
        gumbels = np.random.gumbel(size=len(alphas))
    return (gumbels + log_alphas).tolist()
{% endhighlight %}

Note that this code differs slightly from the routine in Laurent's blogpost, up to a shift of $$\log \alpha_i$$ for each Gumbel.

The above alternative factorization of the joint and the properties of Gumbels that it implies have been valuable to us, allowing manipulation of Gumbels at a higher level of abstraction. In fact, the factorization is a special case of the Top-Down construction that is at the core of [A* Sampling](https://arxiv.org/abs/1411.0030). By applying these intuitions recursively, we can easily derive algorithms for sampling a set of Gumbels in decreasing order, or even sampling a heap of Gumbels from the top down. We hope to cover the more general version in a future blog post.