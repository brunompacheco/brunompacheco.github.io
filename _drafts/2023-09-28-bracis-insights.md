---
layout: distill
title: "BRACIS 2023 - Insights and Takeaways"
date: 2023-09-28
description: Key insights and major takeaways from the presentations and discussions that I attended at BRACIS 2023. 
tags: ai deep-learning gnns
categories: blog
giscus_comments: false

authors:
  - name: Bruno M. Pacheco
    # url: "https://en.wikipedia.org/wiki/Albert_Einstein"
    affiliations:
      name: DAS, UFSC

bibliography: BRACIS.bib

# Optionally, you can add a table of contents to your post.
# NOTES:
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - we may want to automate TOC generation in the future using
#     jekyll-toc plugin (https://github.com/toshimaru/jekyll-toc).
toc:
  - name: Safety and regulatory challenges
  - name: Guarantees and structured data
  - name: Leveraging AI R&D in Brazil
  - name: Concluding remarks

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

This week I have attended the [12th Brazilian Conference on Intelligent Systems - BRACIS](https://www.bracis.dcc.ufmg.br/).
Besides presenting our paper on deep-learning-based brain age estimation and the effects of pre-training such models <d-cite key="pacheco2023does"></d-cite>, I was able to attend to great presentations and have wonderful discussions with the researchers present at the event.
This post is a loose tentative of summarizing the main ideas that I'm bringing back home.

## Safety and regulatory challenges

A theme that was present in almost all keynotes and panels was the challenges of minimizing the risks of machine learning applications.
The discussions ranged from techniques to improve reliability, to the challenges of creating (and approving) laws and norms for AI systems, and to the ethical implications of regulating such a technology.

Particularly, prof. Edgar Lyra (PUC-RIO) presented three perspectives on ensuring the ethical use of AI that I found very useful as an analysis tool.
One is to limit the use of machine learning technologies, for example, through laws prohibiting high-risk applications.
Another one is the creation of techniques and (beaurocratic) processes to ensure ethical applications.
I believe one can think of creating councils to aid the certification of experts, such as in the civil construction industry.
Then, there is the education perspective, to ensure that all parts impacted by ML applications are aware of the risks and limitations.

A great example of the relevance of wide-spread education came in the presentation of "Regulation and Ethics of Facial Recognition Systems: an analysis of cases in the Court of Appeal in the State of São Paulo", during the best papers session.
One of the observations was the lack of well-grounded arguments with respect to the machine learning techniques in the judges' decisions.
Nevertheless, from all approaches discussed, almost none tackled education.

With respect to regulatory efforts, I'm still not sure that creating relevant laws is possible in face of the speed of disruption from this research area.
A major example of this is the EU AI Act, mentioned throughout the conference, which after years of discussion had to be postponed by the launch of ChatGPT.
However, I believe the law makers' discussions on possible regulations for AI are the major drivers of wide-spread education on the subject, along with (often misleading) news.
Thus, at least some benefit is being taken out of it.

## Guarantees and structured data

The discussion on ML safety is surrounded by terms such as explainability, trust, interpretability, and model transparency.
I find it difficult to buying the idea that any technique that provides an "intuition" on model behavior serves more purpose than just calming down the uneducated user.
In fact, for deep learning models, I have a huge difficulty to trust anything beyond generalization performance estimates (which are hard to communicate) and theoretical guarantees (which are hard to find).

However, one thing that was mentioned by prof. Paulo Quaresma and was clear throughout the GNN paper sessions was that encoding symbolic knowledge in the data, such that the model's output can be enhanced by this information, is a viable way to provide guarantees.
This is clear in node classification tasks where the output follows the structure of the input graph.
Of course, there is a bit of bias here as my M.Sc. is on GNNs.

## Leveraging AI R&D in Brazil

Beyond strategies from academic institutions to promote AI research and development, BRACIS hosted some very insightful discussions including representatives from the government, private sector, academia and even from the third sector.
This articulation is driven by several factors, of which I may list:

- [Academia] Brain drain from research institutions driven by the high industry demand for computer scientists and the remote work possibility;
- [Government] Overwhelming impact of recent AI-based technologies (facial recognition, LLMs, AI-based decision making);
- [All] Opportunity to become an active player in the current wave of AI innovation, instead of a mere consumer.

One approach to leverage our AI R&D that had a lot of attention was the financing of hardware resources for researchers.
Although interesting at first, once it increases data privacy and enables research on large DL models, I imagine we would start with a significant lag begind the major players and, since our country is not a player in the semiconductor industry, I'm not very optimistic that we would be able to achieve state-of-the-art performance (both in terms of computing capacity and cost per unit) anytime soon.
I wonder whether it wouldn't be more efficient to exploit our current multi-lateral and bi-lateral partinerships (which are plenty) and try to setup a secure link to privately use computational resources abroad.

Another thing that came to mind during the C&T&I on AI in Brazil panel was that maybe we should not be trying to compete in the hot topics, but rather try to find a way to fit in the global value chain.
I doubt the next LLMs will come from Brazil, however much the government invests in hardware.
I just do not see any advantage that would make us leapfrog the current players.
However, I believe we can use our "Amazons" to fit in the AI revolution.
As pointed out by Luiz Reali, from OBIA, we are at the forefront of digitalization of several state-wide services (e.g., gov.br, pix, e-proc), which means we have the data and the interface to work with.
This could lead to Brazil pioneering AI governmental applications.

## Concluding remarks

I had a wonderful time at BRACIS and not everything fits in the previous sections.
Beyond the ones already mentioned above, some great keynotes that I would like to highlight: prof. Banzhaf on evolutionary machine learning, with some very interesting applications to reinforcement learning; prof. Benevenuto on data science techniques to audit social networks and messaging apps; prof. Bazzan on graph analysis and applications; and prof. Hutter and the meta-learning paradigm for tabular data.

Furthermore, several great papers were presented throughout the week.
My favorite were: _Embracing data irregularities in multivariate time series with Recurrent and Graph Neural Networks_, by Barros et al.; _A combinatorial optimization model and polynomial time heuristic for a problem of finding specific structural patterns in networks_, by Lima and Sampaio; _The Multi-Attribute Fairer Cover Problem_, by Dantas et al.; and _Allocating Dynamic and Finite Resources to a Set of Known Tasks_, by Silva et al.

I'm returning home more motivated and eager to collaborate, after getting to know other researchers, their contributions, visions, struggles, and "a-has".
Also, by feeding from the experts and more experienced researchers, I'm more confident and with expanded horizons.
It was a gret event.
