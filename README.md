Employing Recurrent Neural Networks in Session-Based Recommender Systems


Recommender systems have grown in importance and application diversity; from recommendation of Facebook friends, Spotify Music, and products on Amazon to applications in healthcare and finance. Recommenders assume that a user-tracking mechanism or a user profile is available that can be leveraged to identify user interests. However, in practice very often there is no such mechanism available, as in e-commerce, news, and media sites. The recommendation method therefore relies only on the actions of the user in the current session. This is known as the session-based recommender system and poses new challenges. 

The most commonly used method for session-based recommenders is neighborhood-based Collaborative Filtering (CF). This simple method often takes into account only the last action of the user, ignoring all past events that are available in a sequential data.   

Recurrent Neural Networks (RNN) were developed to deal with sequential data; and Gated Recurrent Units (GRU) were devised to deal with the vanishing/exploding behavior of the gradients in RNNs. It was recently shown [1] that a recommender system based on a GRU RNN that is tailored for this task outperforms the common CF methods by up to 53% in terms of accuracy (Recall@N) on the RecSys Challenge 2015 click-stream data. 

RNNs are commonly fitted with backpropagation through time (BPTT), which has computation and memory cost that scales linearly with the number of time steps. For click-stream data, the session lengths may be very long, which renders BPTT impractical. Another fitting is Recurrent Backpropagation (RBP) [Pineda 1987]. This algorithm generalizes BP to networks where connections can have arbitrary topology. however, RBP is unstable due to non-symmetric connections of the RNN. Conjugate gradient methods for optimization are known to be effective in large-scale problems, and have faster convergence than ordinary BP. Liao et. al [2] have shown that two variants of RBP, based on a conjugate gradient method and the Neumann Series, are stable and match the performance of BPTT, while reducing memory cost and compute time.

In this project we compare recommender systems based on CF, on GRU fitted with BPTT, and on GRU fitted with RBP; and implement the two RBP variants in an attempt to improve backpropagation quality and reduce memory cost and computation time.


[1] Balazs Hidasi, and Alexandros Karatzoglou. 2018. Recurrent Neural Networks with Top-k Gains for Session-based Recommendations.
[2]Renjie Liao, Yuwen Xiong, Ethan Fetaya, Lisa Zhang, KiJung Yoon, Xaq Pitkow, Raquel Urtasum, and Richard Zemel. 2018. Reviving and Improving Recurrent Back-Propagation. 
