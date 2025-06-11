# query_level_uncertainty

This is the repo for our ongoing work "Query-Level Uncertainty in Large Language Models". Feel free to contact us if me missed any relevant papers or you would like to provide feedback. 

## Internal Confidence
The benefits of using our proposed internal confidence
* Training-free. No requirements for training samples.
* Fast. Estimating uncertainty using only a single forward pass of a given query

## Adaptive Inference
In terms of applications, we showcase that our proposed method can help efficient RAG and model cascading. 
On the one hand, internal confidence can guide users to assess the trade-offs between cost and quality when invoking additional services. On the other hand, it brings a ``benefit region'', where inference overhead can be reduced without compromising performance.

<p float="left">
  <img src="figure/rag_acc_cost.png" width="45%" />
  <img src="figure/cascade_acc_cost.png" width="45%" />
</p>
