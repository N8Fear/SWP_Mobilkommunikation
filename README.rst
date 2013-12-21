People Recognition using a Hidden Markov Model on infraread images
==================================================================

TODO:
-----

1. Create Model
	- what is used (pixel, "objects", whole image)
	- k-means for clustering

2. Implementation

3. Evaluation

Model:
------
- Use DCT to preprocess image data:
  Right now we are just using the DC value of the DCT
  Possible refinement: using chosen AC values; a few values from the border of
  the block seem reasonable: a high derivation from the DC value inferes
  a change and therefore hints at possible movement (especially if we factor in
  the last known value)
- Use k-means to cluster the data from the DCT to gain an ideally small
  alphabet of observations for the HMM
- HMM for infering the presence of people in the picture
