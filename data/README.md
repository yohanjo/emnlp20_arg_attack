# CMV

* `posts-sents.csv.zip`: All posts with individual sentences.
  * `post_id`: Post ID.
  * `sentence_no`: Sentence number.
  * `sentence`: Sentence text.
  * `sentence_token`: Tokenized sentence.

* `feat-combined.csv.zip`: All features and labels for individual sentences.
  * `split`: {train, val, test}.
  * `post_id`: Post ID.
  * `sentence_no`: Sentence number.
  * `direct`: 1 if the sentence is directly quoted by a comment (using the `>` symbol); 0 otherwise.
  * `success_direct`: 1 if the sentence is directly quoted and successfully attacked; 0 otherwise. If `direct`=1 and `success_direct`=0, the sentence is directly attacked unsuccessfully.
  * `all_4`: 1 if the sentence is indirectly quoted (4+ words overlap); 0 otherise.
  * `success_all_4`: 1 if the sentence is indirectly quoted and successfully attacked; 0 otherwise. If `all_4`=1 and `success_all_4`=0, the sentence is indirectly attacked unsuccessfully. 
  * `senti_score`: Sentiment score (float).
  * `senti_class:[pos|neu|neg]`: Sentiment category (binary).
  * `arousal`: Arousal score (float).
  * `dominance`: Dominance score (float).
  * `concreteness`: Concreteness score (float).
  * `subjectivity`: Subjectivity score (float).
  * `hedging`: Hedges score (float).
  * `quantification`: Quantification score (float).
  * `topic50:X`: Sentence topic (binary).
  * `kialo_wo5_freq`: Kialo frequency (log2).
  * `kialo_wo5_attr`: Kialo attractiveness (log2).
  * `kialo_wo5_extreme`: Kialo extremeness (float).
  * `kialo_ukp_avgdist10`: UKP avg distance 10 (float). (Table 5)
  * `kialo_ukp0.X_freq`: UKP 0.X frequency (log2). (Table 5)
  * `kialo_ukp0.X_attr`: UKP 0.X attractiveness (log2). (Table 5)
  * `kialo_ukp0.X_extreme`: UKP 0.X extremeness (float). (Table 5)
  * `kialo_frame_consistent`: Frame knowledge consistent (int). (Table 5)
  * `kialo_frame_conflict`: Frame knowledge conflict (int). (Table 5)
  * `kialo_wklg2_consistent`: Word sequence knowledge consistent (int). (Table 5)
  * `kialo_wklg2_conflict`: Word sequence knowledge conflict (int). (Table 5)
  * `domain40:X`: Domain of the post (binary).


## Data Filtering
* Successfully attacked sentences: filter sentences by `success_direct=1 or success_all_4=1`.
* Unsuccessfully attacked sentences: filter sentences by `(direct=1 or all_4=1) and success_direct=0 and success_all_4=0`.
* Unattacked sentences: filter sentences by `direct=0 and all_4=0`.

## Mapping Posts with Attacking Comments
In case you want to map each post with the attacking comments, use `posts-qsents.csv`.
This file may contain some posts that are not included in our dataset.
The columns are as follows:
* `post_id`: Post ID.
* `n_sentences`: Number of sentences in the post.
* `comment_id`: Comment ID that quotes the post. One post may have multiple attacking comments.
* `delta`: 1 if the comment received a delta; 0 otherwise.
* `direct_sents`: Sentences that are quoted directly by the comment with the `>` symbol (comma-separated).
* `direct_n_quotes`: Number of direct quotes by the comment.
* `all_4_sents`: Sentences that are quoted indirectly by the comment with word overlap (comma-separated).
* `all_4_n_quotes`: Number of indirect quotes by the comment.

The text of comments could be obtained from the raw json below.


# CMV Raw 
Raw posts and comments written between January 1, 2014 and September 30, 2019, scraped using the Pushshift API. Go to the `cmv` folder and extract the compressed files. You need to reassemble `comments.jsonlist.zip` before unzipping it:
```
$ cat comments.jsonlist.zip.?? > comments.jsonlist.zip
```
Note that the `id` field of each post and comment is prefixed with `t3_` and `t1_`, respectively, in our dataset. This is a Reddit convention.


# Kialo
* `kialo.csv`: Kialo statements from kialo.com written until October 2019.
  * `did`: Discussion ID.
  * `cid`: Statement ID.
  * `author`: User ID.
  * `parent`: Parent statement ID.
  * `relation`: Relation type (1=pro, -1=con, 0=neutral).
  * `deleted`: 1 if the statement was deleted. This statement is not used in the paper.
  * `rel-pro`: Number of pro responses.
  * `rel-neu`: Number of neutral responses.
  * `rel-con`: Number of con responses.
  * `text`: Text.
