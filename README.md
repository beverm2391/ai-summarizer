The full writeup and documentation are right here on my blog, [Ben's Blocks](https://blocks.beneverman.com/projects/ai-summarizer).

# Accurate Summarization of Scientific Articles with AI

Out of the box LLMs like GPT-3 are not well suited for summarizing scientific articles. They hallucinate, have limited to no knowledge of current events, and they have limited context window. I built a NLP pipeline to solve these problems and generate accurate summaries (with citations) in about 30 seconds.

## Introduction
I started playing with GPT-3 in November of 2021. I quickly realized I could leverage Natural Language Processing (NLP) to significantly speed up the pace of my schoolwork. Out of the box, the newest Large Language Models (LLMs) can do quite a bit out of the box. They can:

1. Generate text
2. Converse
3. Answer open book questions (you provide the context in the prompt)
4. Answer closed book questions (semi-accurately)

They are essentially *complex data structures that can be queried with natural language*. When scaled up to billions of params, increased performance on un-trained benchmarks *emerges*, and we get these psuedo-intelligent feeling models like GPT-3. However, they are not perfect. For summarization, they have a few major limitations that I will discuss.

In my case, I want to generate accurate and thoughtful summaries (with sources) from a single or multiple source documents. Not just to read, but because summarizing scientific journals is one of my assignments! Later, I'll integrate this into my process of writing research papers.

## Demo

Here's a demo of my summarizer in action. My plan is to integrate this into [semantics.cloud](https://www.semantics.cloud) (my semantic search app) within the next few weeks.

**Input:** [The Core Mechanisms of Cognitive Behavioral Therapy for Anxiety and Depression: A Review](https://pubmed.ncbi.nlm.nih.gov/29080589/)

**Output** (with citations, in 38.74 seconds):

>Cognitive-behavioral therapies (CBTs) serve as a beneficial treatment for symptoms of anxiety and depression. Research on CBT aims to identify the core mechanisms of change responsible for symptom improvement. There are two primary core mechanisms for CBTs for anxiety: fear extinction and threat reappraisal. Cognitive change is the most researched mechanism for CBTs for depression (Powers, 2017). Although these two theories may seem incompatible, they may represent differences in levels of analysis. Cognitive change is a critical factor in CBT for both anxiety and depression, regardless of how it is achieved.
>
>Powers' (2017) article, Core Mechanisms of Cognitive Behavioral Therapy for Anxiety and Depression, delves into the core mechanisms of cognitive behavioral therapy for treating mental health disorders like anxiety and depression. The article discusses various techniques such as fear extinction, threat reappraisal, exposure therapy, and cognitive restructuring, as well as newer techniques such as acceptance and mindfulness-based therapies, cognitive bias modification interventions, positive mood induction, exercise augmentation, and the importance of sudden gains and critical sessions. All these psychological processes are used to target faulty threat appraisals, which ultimately lead to symptom improvement and reduced anxiety. 
>
>Cognitive change plays a crucial role in the success of CBT for both anxiety and depression. In CBT for anxiety, fear extinction and threat reappraisal are central mechanisms of change. Fear extinction involves reducing the fear response to a conditioned stimulus by exposing individuals to the feared stimulus repeatedly without any adverse consequences. Threat reappraisal is a cognitive process that aims to change an individual's perception of a stressor, ultimately leading to decreased anxiety. In contrast, cognitive changes are primarily responsible for symptom improvement in CBTs for depression. Research has revealed cognitive restructuring as the most researched psychological mechanism in CBT for depression, observed in both antidepressant and CBT interventions.
>
>Additionally, various methods have been suggested to enhance threat reappraisal. Mindfulness-based approaches and acceptance and commitment therapy are two such methods. The study by Alfei, Ferrer Monti, Molina, et al., (2015) emphasizes that prediction error and trace dominance are essential factors in determining the fate of fear memories after post-training manipulations. On the other hand, Julian, Beard, Schmidt, et al., (2012) study presents how attention training can reduce attention bias and social stressor reactivity, while Vittengl, Clark, & Jarrett (2005) discuss the validity of sudden gains in acute-phase treatment of depression. 
>
>In conclusion, cognitive-behavioral therapies (CBTs) have proved effective in healing symptoms of anxiety and depression. Anxiety and depression have different core mechanisms for CBTs, but cognitive change is an integral factor in both. Fear extinction and threat reappraisal are two central mechanisms for CBT for anxiety, while cognitive restructuring is crucial for CBT for depression. Therapists can use various techniques, such as acceptance and mindfulness-based therapies, to enhance threat reappraisal. Therefore, understanding the core mechanisms of CBT encourages the development of future CBTs to improve mental health treatments. 
>
>References:
>
>Powers, M. B. (2017). Core Mechanisms of Cognitive Behavioral Therapy for Anxiety and Depression. Psychiatric Clinics of North America, 40, 611-623. doi:10.1016/j.psc.2017.08.010

Pretty sweet, right? I'm really happy with the results - it's a huge improvement over an out-of-the-box LLM.

Here's the debug output. You can see that in total, I made 9 calls in link one, 3 calls in link two (I'm batching them so you dont see them all in the output), and 1 call in link three. The total time was 38.74 seconds.

```
Starting Chain
Starting Link 1:
Response 5 of 9 complete.
Response time: 3.43 seconds.
Response 2 of 9 complete.
Response time: 3.96 seconds.
Response 1 of 9 complete.
Response time: 4.06 seconds.
Response 4 of 9 complete.
Response time: 5.53 seconds.
Response 8 of 9 complete.
Response time: 6.35 seconds.
Response 9 of 9 complete.
Response time: 6.35 seconds.
Response 7 of 9 complete.
Response time: 6.49 seconds.
Response 6 of 9 complete.
Response time: 8.76 seconds.
Response 3 of 9 complete.
Response time: 9.24 seconds.
Link 1 Complete
Link 1 Complete in 9.24 seconds.
Starting Link 2:
Link 2 Complete
Link 2 Complete in 8.27 seconds.
Starting Link 3:
Link 3 Complete
Link 3 Complete in 8.27 seconds.
Chain Complete in 38.74 seconds.
```
