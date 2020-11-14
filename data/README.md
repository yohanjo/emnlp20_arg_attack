# CMV

* `posts-sents.csv.zip`: All posts with individual sentences.
  * `post_id`: Post ID.
  * `sentence_no`: Sentence number.
  * `sentence`: Sentence text.

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
