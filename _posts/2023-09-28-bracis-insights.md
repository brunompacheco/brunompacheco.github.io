---
layout: distill
title: "Reflections on BRACIS 2023: AI Regulation, guarantees and the case for Brazil"
date: 2023-10-01
description: An opinion post on key insights and takeaways from the event. 
tags: ai deep-learning gnns
categories: opinion
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
  - name: Navigating ethic and regulatory challenges
  - name: AI safety, guarantees, and structured data
  - name: Leveraging research and development in Brazil
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

I have just returned home from a week at Belo Horizonte (MG), attending the [12th Brazilian Conference on Intelligent Systems - BRACIS](https://www.bracis.dcc.ufmg.br/).
Besides presenting<d-footnote>Slides available at <a href="https://github.com/gama-ufsc/brain-age/">github.com/gama-ufsc/brain-age</a></d-footnote> our paper on deep-learning-based brain age estimation and the effects of pre-training such models <d-cite key="pacheco2023does"></d-cite>, I was able to attend great presentations and have wonderful discussions with the researchers present at the event.
This post is a tentative summary of my major takeaways from the event, some insights I had while there, and a few highlights of the presentations.
As this blog is my personal space, I'll take the liberty of being provocative<d-footnote>Feel free to reach me out if you want to weigh in :)</d-footnote>.

## Navigating ethic and regulatory challenges

One recurring theme that resonated throughout BRACIS 2023 was the baffling task of mitigating the risks associated with machine learning applications.
The discussions encompassed a wide spectrum, from enhancing the reliability of AI systems to the complexities of crafting and ratifying regulations for these technologies, all while pondering the ethical implications of such advancements.

One illuminating presentation came from prof. Edgar Lyra of PUC-RIO, who presented three complimentary perspectives on ensuring the ethical use of AI, which I found very useful to break down the analysis.
One approach, as I understood, involves restricting the application of machine learning technologies (e.g., through legislation), as in the [EU AI Act](https://www.europarl.europa.eu/news/en/headlines/society/20230601STO93804/eu-ai-act-first-regulation-on-artificial-intelligence).
Another avenue is to develop techniques and bureaucratic processes that guarantee ethical applications, perhaps by establishing councils to certificate professionals, akin to those in the civil construction industry.
An example, [PL 2338/2023](https://www25.senado.leg.br/web/atividade/materias/-/materia/157233) stipulates the creation of several governance measures from the parts involved, making it somewhat possible to audit the applications.

Then, there is the education perspective, ensuring that all stakeholders impacted by ML applications are well-versed in the associated risks and limitations.
A great example of the relevance of widespread education came in the presentation of "Regulation and Ethics of Facial Recognition Systems: an analysis of cases in the Court of Appeal in the State of São Paulo", during the best papers session.
The authors observed a lack of well-grounded arguments with respect to the machine learning techniques in the judges' decisions, leading to unfair settlements, which highlights the urgency of the educational aspect.

Regarding regulatory efforts, I'm still grappling with the idea of crafting pertinent laws in the face of the rapidly evolving landscape of AI research.
An illustrative example is the delay of the EU AI Act after the launch of ChatGPT.
However, I believe the lawmakers' discussions on possible regulations for AI are, currently, the major drivers of wide-spread education on the subject, along with (often misleading) news.
Thus, at least some benefit is being drawn out of it.

## AI safety, guarantees, and structured data

Discussions on machine learning safety typically revolved around terms like explainability, trust, interpretability, and model transparency.
Personally, I find it difficult to see how any technique that provides an "intuition" on model behavior serves more purpose than just calming down the uneducated user<d-footnote>This is true in the context of the deployed application. I value (and, actually, frequently use) explanation techniques to debug models and guide development and research.</d-footnote>.
In fact, for deep learning models, I have a huge difficulty to trust anything beyond generalization performance estimates (which are often hard to communicate effectively) and theoretical guarantees (which can be very hard to provide for high-performance models).
As discussed by prof. Baeza-Yates, in the absence of theoretical guarantees, i.e., if we know that a model will eventually fail, one might question the ethics of deploying such a deep-learning-based application.

However, one intriguing insight was mentioned by prof. Paulo Quaresma and was further materialized during the GNN paper sessions.
These conversations underscored the value of encoding symbolic knowledge into the data.
More specifically, using such knowledge to augment the model's output is a viable way to provide solid guarantees.
A clear example can be seen in tasks like node classification, where the model's output aligns with the structure of the input graph—a domain I find compelling due to my current research interest in GNNs.
For me, this is a reminder that, amidst the quest for safety in AI, leveraging structured data can be a promising avenue for crafting truly reliable applications.

## Leveraging research and development in Brazil

Beyond the challenges of AI, BRACIS 2023 also hosted very insightful discussions on strategies for Brazil to harness the potential of AI research and development.
The discussions involved representatives from various sectors, including government, academia, and the private sector.
This collaboration was driven by several factors, of which I highlight a few in the following.

### Brain drain

Our research institutions face a significant "brain drain".
The talent retention difficulty comes mainly from an increasing demand for computer scientists in the private sector.
Not only our companies are opening more positions for the researchers, but now, due to the remote work opportunities, the academia has to compete with foreign companies.
Therefore, the competition for human resources ends up happening in foreign currency, which is particularly hard on academic institutions with government funding.

### Government impact

Recent AI-based technologies (e.g., facial recognition, employment decision-making, chatbots) have had a profound impact on various aspects on society and governance.
I have mentioned above how lawmakers and judges are struggling to handle these new technologies, but the challenges spread through all three powers of the government.
The executive has to position the country with respect to the international partnerships to avoid misuse of our resources (human and data), and nationally to stimulate the use of these technologies to promote sustainable development.

### Innovation opportunities

At the same time, as the AI industry is still in its early stages, we have the opportunity to become active players in the global AI landscape, rather than passive consumers.
Brazil's public and private sectors are united in this perspective.
However, neither have clear paths for promoting the national role in the global community.

<br/>
<br/>

One approach to leverage our AI research and development that garnered significant attention was the provision of hardware resources for researchers.
While this idea initially holds promise, especially in terms of enhancing data privacy and facilitating research on large deep learning models, I have a concern that Brazil may have an unbridgeable gap behind global leaders.
Given that our country does not have a significant presence in the semiconductor industry, I believe the costs of achieving and maintaining state-of-the-art computing performance may be prohibitively high.
An alternative could be to leverage our existing multilateral and bi-lateral partnerships, which the representatives of the [MCTI](https://www.gov.br/mcti/pt-br) ensured are abundant, and establish secure connections to privately access dedicated computational resources abroad.
This way, we could leapfrog some of the infrastructure limitations and stay competitive on the global stage.

Another insight emerged in my mind during the "C&T&I on AI in Brazil" panel – perhaps our focus should not be on competing in the hottest AI topics but rather on finding our place within the global AI value chain.
I find it unlikely that the next breakthroughs in LLMs will come from Brazil, no matter how much the government may invest in hardware, because I don't see unique strengths that could make us leapfrog the current players.
I believe we should find our "Amazons" of AI, that is, our assets that could become an advantage in the competition.
For example, as highlighted by Luiz Reali from OBIA, we are at the forefront of digitizing several state-wide services (see [gov.br](https://www.gov.br/pt-br), [pix](https://www.bcb.gov.br/estabilidadefinanceira/pix), and [e-proc](https://jus.com.br/artigos/4795/a-virtualizacao-dos-processos-judiciais-e-proc-e-a-dispensabilidade-de-autenticacao-documental-por-tabeliao)).
This grants us access to vast amounts of data and a natural interface for developing AI applications in the governmental sector.
By capitalizing on these strengths, Brazil could pioneer AI-driven governmental applications, carving out a distinct and impactful role in the AI revolution.

## Concluding remarks

I had a wonderful time at BRACIS and not everything fits in the previous sections.
Beyond the ones already mentioned above, some great keynotes that I would like to highlight: prof. Banzhaf on evolutionary machine learning, with some very interesting applications to reinforcement learning; prof. Benevenuto on data science techniques to audit social networks and messaging apps; prof. Bazzan on graph analysis and applications; and prof. Hutter and the meta-learning paradigm for tabular data.

Furthermore, many great papers were presented throughout the week.
My favorite were: _Embracing data irregularities in multivariate time series with Recurrent and Graph Neural Networks_, by Barros et al.; _A combinatorial optimization model and polynomial time heuristic for a problem of finding specific structural patterns in networks_, by Lima and Sampaio; _The Multi-Attribute Fairer Cover Problem_, by Dantas et al.; and _Allocating Dynamic and Finite Resources to a Set of Known Tasks_, by Silva et al.

I'm returning home more motivated and eager to collaborate, after getting to know other researchers, their contributions, visions, struggles, and "a-has".
Also, by feeding from the experts and more experienced researchers, I'm more confident and with expanded horizons.
It was a great event.
