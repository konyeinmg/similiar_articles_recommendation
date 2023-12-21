# similar_articles_recommendation
Similar Articles Recommendation by Locality Sensitive Hashing

### Project
<p>This project is implemented with <b>locality sensitive hashing and cosine similarity</b> to recommend similar articles.</p>

### Data
<p>The dataset is downloaded from <a href="https://data.world/opensnippets/al-jazeera-news-dataset">Al Jazeera English News Dataset</a> and it has 3400+ articles and I use 90% of data to make hash table.</p>

### TF-IDF Vectorizer
<p>I extracted <b>1024 features</b> from these articles using TF-IDF vectorizer and remove words under appearance of ten times.</p>

### Word Embeddings
<p>It is possible to use only TF-IDF feature extraction in this project but to get better similar articles I also use word embeddings to know deep meaning of words. The word embedding I used in this project is downloaded from google(<i>GoogleNews-vectors-negative300.bin</i>) with (n,300) dimensions. It is a big file so it is listed in gitignore file. You can also use other word embedding files. </p>
Link => <a href="https://www.kaggle.com/datasets/leadbest/googlenewsvectorsnegative300">https://www.kaggle.com/datasets/leadbest/googlenewsvectorsnegative300</a>
<a></a>

### Locality Sensitive Hashing
<p>I use locality sensitive hashing algorithm in finding similar articles to improve searching time complexity in large datasets. I also implement cosine similarity and KNN to compare the results. In LSH, I used 8 planes to get about 213 buckets so that each bucket will have nearly 16 vectors.</p>

### Principal Component Analysis
<p>As a bonus, I also implemented PCA.py to visualize word embeddings of opposite words in 2D plot after transformation of word embeddings from 300 dimensions to two dimensions with principal component analysis.</p>