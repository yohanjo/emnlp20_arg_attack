FILENAME_DATA = "~../data/feat-combined.csv"
VAR_CONTENT = c(
  "X.topic50.0.", "X.topic50.1.", "X.topic50.2.",             
  "X.topic50.3.", "X.topic50.4.", "X.topic50.5.",             
  "X.topic50.6.", "X.topic50.7.", "X.topic50.8.",             
  "X.topic50.9.", "X.topic50.10.", "X.topic50.11.",            
  "X.topic50.12.", "X.topic50.13.", "X.topic50.14.",            
  "X.topic50.15.", "X.topic50.16.", "X.topic50.17.",            
  "X.topic50.18.", "X.topic50.19.", "X.topic50.20.",            
  "X.topic50.21.", "X.topic50.22.", "X.topic50.23.",            
  "X.topic50.24.", "X.topic50.25.", "X.topic50.26.",            
  "X.topic50.27.", "X.topic50.28.", "X.topic50.29.",            
  "X.topic50.30.", "X.topic50.31.", "X.topic50.32.",            
  "X.topic50.33.", "X.topic50.34.", "X.topic50.35.",            
  "X.topic50.36.", "X.topic50.37.", "X.topic50.38.",            
  "X.topic50.39.", "X.topic50.40.", "X.topic50.41.",            
  "X.topic50.42.", "X.topic50.43.", "X.topic50.44.",            
  "X.topic50.45.", "X.topic50.46.", "X.topic50.47.",            
  "X.topic50.48.", "X.topic50.49.")

VAR_KNOWLEDGE = c("X.kialo_wo5_freq.", "X.kialo_wo5_attr.", "X.kialo_wo5_extreme.",
                  "X.kialo_ukp_avgdist10.",
                  "X.kialo_ukp0.1_freq.", "X.kialo_ukp0.1_attr.", "X.kialo_ukp0.1_extreme.",
                  "X.kialo_ukp0.2_freq.", "X.kialo_ukp0.2_attr.", "X.kialo_ukp0.2_extreme.", 
                  "X.kialo_ukp0.3_freq.", "X.kialo_ukp0.3_attr.", "X.kialo_ukp0.3_extreme.",
                  "X.kialo_ukp0.4_freq.", "X.kialo_ukp0.4_attr.", "X.kialo_ukp0.4_extreme.", 
                  "X.kialo_frame_consistent.", "X.kialo_frame_conflict.", 
                  "X.kialo_wklg_consistent.", "X.kialo_wklg_conflict.")

VAR_PROP = c("X.question_confusion.", "X.question_whyhow.", "X.question_other.", "X.normative.", 
             "X.prediction.", "X.if.", "X.source.", "X.comparative.", "X.example.", "X.definition.",
             "X.my.", "X.you.", "X.we.")

VAR_TONE = c("X.subjectivity.", "X.concreteness.", "X.quantification.", "X.hedging.", 
             "X.senti_score.", "X.senti_class.pos.", "X.senti_class.neu.", "X.senti_class.neg.", 
             "X.arousal.", "X.dominance.")

FEATURES = c(VAR_CONTENT, VAR_KNOWLEDGE, VAR_PROP, VAR_TONE)

############################
## Load & Preprocess data ##
############################
dat = read.csv(FILENAME_DATA, stringsAsFactors = FALSE)
dat = dat[which(dat$split %in% c("train", "val")),] 

# standardize features
dat$X.subjectivity. = dat$X.subjectivity./sd(dat$X.subjectivity.)
dat$X.concreteness. = dat$X.concreteness./sd(dat$X.concreteness.)
dat$X.hedging. = dat$X.hedging./sd(dat$X.hedging.)
dat$X.quantification. = dat$X.quantification./sd(dat$X.quantification.)
dat$X.senti_score. = dat$X.senti_score./sd(dat$X.senti_score.)
dat$X.arousal. = dat$X.arousal./sd(dat$X.arousal.)
dat$X.dominance. = dat$X.dominance./sd(dat$X.dominance.)
dat$X.kialo_ukp_avgdist10. = dat$X.kialo_ukp_avgdist10./sd(dat$X.kialo_ukp_avgdist10.)

# merge domain features
dat$domain = 0
columnlist = grep("X.domain40[.]", colnames(dat), value = T)
for (mycol in columnlist) {
  j = as.numeric(strsplit(mycol, "[.]")[[1]][3])
  dat$domain[which(dat[mycol] == 1)] = j  
}
dat$domain = as.factor(dat$domain)

############################
## Labeling Attackability ##
############################
# Each sentence is labeled as attacked if it is directly quoted with > symbol
# or at least four non-stopwords appear in a comment's sentence.
dat$attacked = as.numeric(dat$direct | dat$all_4) 

# Each attacked sentence is labeled as successfully attacked 
# if any of the comments that attack it or their lower-level comments
# win a Delta.
dat$suc_attacked = as.numeric(dat$success_direct | dat$success_all_4) 


######################################################################
## Conditional Feature Effect and P-value (Attacked vs. Unattacked) ##
######################################################################
# Only use sentences that are attacked or in the same posts with the attacked sentences
post_attacked = dat$post_id[which(dat$attacked == 1)]
idx_attacked = which(dat$post_id %in% post_attacked)
dat_attacked = dat[idx_attacked, ]

# Fit logistic regression model for each feature
res_attacked = data.frame(Label = character(), 
                          Feature = character(), 
                          Effectsize = double(), 
                          EffectsizeOR = double(), 
                          Pvalue = double(), 
                          stringsAsFactors = FALSE)
for (feature in FEATURES) {
  eval(parse(text = paste0("fit <- glm(as.factor(attacked) ~ ", feature, " + as.factor(domain), data = dat_attacked, family = 'binomial')")))
  fitsummary <- coef(summary(fit))
  effect <- fitsummary[2,1]
  pval <- fitsummary[2,4]
  res_attacked = rbind(res_attacked, 
                       data.frame(Label = "Attacked", 
                                  Feature = feature, 
                                  Effectsize = effect, 
                                  EffectsizeOR = exp(effect),
                                  Pvalue = pval))
}
res_attacked

#######################################################################################
## Conditional Feature Effect and P-value (Successfully vs. Unsuccessfully Attacked) ##
#######################################################################################
# Only use sentences that are attacked
idx_suc_attacked = intersect(idx_attacked, which(dat$attacked == 1))
dat_suc_attacked = dat[idx_suc_attacked, ]

# Fit logistic regression model for each feature
res_suc_attacked = data.frame(Label = character(), 
                          Feature = character(), 
                          Effectsize = double(), 
                          EffectsizeOR = double(), 
                          Pvalue = double(), 
                          stringsAsFactors = FALSE)
for (feature in FEATURES) {
  eval(parse(text = paste0("fit <- glm(as.factor(suc_attacked) ~ ", feature, " + as.factor(domain), data = dat_suc_attacked, family = 'binomial')")))
  fitsummary <- coef(summary(fit))
  effect <- fitsummary[2,1]
  pval <- fitsummary[2,4]
  res_suc_attacked = rbind(res_suc_attacked, 
                           data.frame(Label = "SuccessfullyAttacked", 
                                      Feature = feature, 
                                      Effectsize = effect, 
                                      EffectsizeOR = exp(effect), 
                                      Pvalue = pval))
}
res_suc_attacked
    