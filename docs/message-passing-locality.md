# Phase 4 mathematical theory

## Model and locality

Let G be a finite, simple, undirected graph. Write B_k(v) for the rooted induced subgraph on vertices at hop distance at most k from v. Each vertex carries an input attribute x and each oriented edge carries an attribute e. The attributes are part of the comparison. The theorem allows any supplied attributes, but a claim about the spread of source information requires that attributes other than the source indicator be independent of source choice.

Use a shared nodewise encoder. In each synchronous round, form messages from the previous states of the edge endpoints and the edge attribute, combine the incoming multiset by a permutation-invariant operation, and apply a shared nodewise update. The update may use the previous state, the node attribute and a residual connection. Use a shared nodewise readout from the final state and, optionally, the local input. Fix all parameters across the queries. Nodewise normalization with fixed statistics or normalization using only the incoming multiset is allowed. Exclude graph-wide statistics computed at inference, graph pooling broadcast to nodes, virtual edges and other side channels. A coefficient such as a sender's full degree must be included in the compared attributes if the model uses it, since a boundary sender's full degree need not be determined by the induced ball.

**Proposition 1.** If two attributed rooted k-hop balls are isomorphic, the roots have the same output after k rounds.

**Proof.** Let f be an attribute-preserving rooted isomorphism. Distances to the root within each ball equal the original hop distances because shortest paths to the root stay inside the ball. At time zero, corresponding states match everywhere in the balls by the encoder assumption. Suppose states match at time t at all vertices at distance at most k − t from the root. Every vertex at distance at most k − t − 1 has all its neighbors within distance k − t. The isomorphism therefore matches the entire incoming multiset with matching previous states and edge attributes. Aggregation, update and any residual addition preserve equality. Induction leaves matching states at the roots at time k, and the readout preserves equality. The same argument with a full graph relabeling proves permutation equivariance, meaning that relabeling inputs only relabels outputs. This proof uses exact arithmetic. Floating-point sums can change slightly when edge order changes. ∎

This is a receptive-field consequence of the established message-passing framework. Isomorphism is sufficient for equality. A particular aggregator can also map nonisomorphic balls to equal states. [Gilmer et al., Neural Message Passing for Quantum Chemistry](https://arxiv.org/abs/1704.01212).

### Implementation audit

- The encoder sees only the Boolean source mask, one value per vertex. Each undirected edge becomes two directed entries carrying the same weight divided by ten. Sorted vertex identifiers determine storage indices only. Hop counts and scaled Dijkstra labels are held separately for evaluation and supervision.
- The message MLP uses the previous sender state and edge attribute. The receiver sums incoming messages, then an MLP uses that sum and the receiver's previous state. The residual version also adds the receiver's previous state. All receivers update synchronously, with one round per layer.
- The readout is affine and applied independently at each vertex. The source-zero override uses the same local mask. There is no degree, batch or graph normalization. The fixed factor ten is independent of each graph and source.
- The fixed-graph source intervention keeps the edge array and accumulation order identical. Phase 3 verified exact equality on CPU. The mathematical equivariance claim for arbitrary relabelings permits floating-point roundoff in software.

## Shortest-path lower bound

Write d_H(s,v) for unweighted hop distance and d_w(s,v) for the minimum sum of positive edge weights on a path. If two indistinguishable inputs have scalar targets a and b and common prediction p, then

> |a − b| ≤ |a − p| + |p − b| ≤ 2 max(|p − a|, |p − b|).

Thus both the maximum and the equally weighted pair mean of absolute errors are at least |a − b|/2. The midpoint attains both bounds. This gives no universal dataset-wide MAE floor of the same size. For example, if these are two equally weighted nodes among N observations, their combined contribution to pooled MAE is at least |a − b|/N.

For any k, on the unit-weight path 0 through k+2 compare sources k+1 and k+2 at vertex 0. The source flag vanishes throughout both balls and all edge attributes coincide. The targets differ by one. At k=4 the targets are 5 and 6, giving a 0.5 weight-unit bound, or 0.05 in the network's target units. More generally on a path extending to b, unit-weight sources at integers k < a < b realize gap b − a. On the same positively weighted path the gap is the sum of weights between the two source positions. With unrestricted path size this yields arbitrarily large worst-case error at fixed depth even under bounded positive weights. Phase 3's ratio of summed absolute errors to summed target distances is a different metric from fixed scaling by ten. Applied to just the pair it has lower bound |a − b|/(a+b).

### Limits and counterexamples

- Hop distance sets the earliest possible influence of the indicator. A weighted triangle with direct weight 10 and two indirect edges of weight 1 has hop distance 1 but weighted distance 2. At depth one the source is visible, yet a two-edge optimum can depend on weights outside the messages received in round one. For example, keeping the two edges incident to the prediction vertex fixed and increasing the opposite edge from 1 to 9 changes the target from 2 to 10 while leaving its one-round inputs identical. Even the induced one-hop ball in this example has information the architecture cannot use in one round.
- Broadcast the sum over nodes of source indicator times a fixed position attribute on a numbered path. A node can now recover the source position even beyond k hops. This broadcast is equivariant if position attributes travel with vertices, demonstrating that equivariance alone does not imply locality. Precomputed source distances or global source-dependent normalization likewise bypass the source-mask propagation assumption.
- An invisible indicator still leaves local structure and weights. A model can exploit correlations between local observations and distance, and more training can change those conditional predictions. The lower bound persists for any parameters shared by both queries.

## Phase 3 interpretation and open questions

Connected Erdős–Rényi raw MAE rose with size in all six models. Diameter and target scale rose alongside it, even with relatively stable mean degree. Standard depth-4 normalized MAE rose from 0.323 at 8 vertices to 0.407 at 64, so larger raw targets alone do not describe the pattern. The experiment did not intervene on any one of these factors.

At 64 vertices paths combined large targets (mean 120.13) with 4,422 of 5,040 evaluated nodes beyond four hops. Their unique routes accumulate many edge weights, and a bounded neighborhood leaves part of a distant route unobserved. Cycles and grids have alternative routes, and random graphs had substantially smaller distances. The 8-vertex cycle error exceeded Erdős–Rényi error even with complete four-hop source coverage, so source visibility does not suffice to explain the family ordering.

Standard depth 8 improved on depth 4 on average in every 64-vertex family, while residual depth 8 worsened. Depth-8 residual validation performance was similar to standard despite worse mean large-graph performance. These comparisons do not isolate capacity, optimization or shift. Fixed width increases parameters and work with depth. Thirteen of 30 fits selected the last epoch. The training graph and target distributions also differ from the structured probes. Fixed division by ten preserves all target ratios and leaves those distribution differences intact.

For structured families the standard model's average improvements at hops 9+ were comparable to improvements at hops 5–8, with large paired variability. Proposition 1 excludes newly arriving source information as an explanation in the 9+ band. More structural context, different fitted parameters and optimization remain possible explanations. Final-output supervision and an exact invariance check provide no evidence about internal Dijkstra traces. [Phase 3 results](size-and-topology-results.md)

The controlled path study asks whether a depth advantage persists near matched budgets and whether altered training distance coverage helps on identical evaluation queries. Open questions include separating target-scale coverage from source-placement priors, checking optimizer convergence independently of architecture, and finding matched comparisons across graph families. The locality proof and elementary error bound specialize established ideas to the implemented model.
