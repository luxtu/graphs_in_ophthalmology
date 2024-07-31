# Evaluating graphs as interpretable data representations of OCTA images for disease staging and biomarker prediction in ophthalmology

We present and evaluate two graph representations of OCTA images.

- Vessel Graphs
- Intercapillary Area Graphs

The basis of both representations is a high-quality segmentation map. In vessel graphs, the nodes represent vessel segments that end either at a bifurcation point or at the end of a vessel. Edges are introduced if two vessel segments are connected through a bifurcation point. The intercapillary area contains the intercapillary areas as nodes. It connects intercapillary area nodes through edges if they are separated by just a single pixel of a vessel centerline (skeletonized segmentation map).

