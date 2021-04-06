#SA_engageAna_permutations.R

# Sophie Wohltjen, April 2021
# This script computes the mixed model regression of pupillary synchrony and eye contact on mean engagement,
# then scrambles dyads and performs regressions on pseudo-dyads
# then compares our estimate to the null

base_directory = '/Users/sophie/Dropbox/EyeContactinConversation'

#read in the dataset
ec_logreg_engage <- read.csv(paste(base_directory,'/Analyses/SA_data_for_logreg_engage_anas.csv',sep=''))
#make binary eye contact a factor
ec_logreg_engage$ec <- factor(ec_logreg_engage$ec)
#get negative dtw for synchrony variable
ec_logreg_engage$synchrony <- -log(ec_logreg_engage$dtw)
#make dyad a factor
ec_logreg_engage$dyad <- factor(ec_logreg_engage$dyad)

#make sure we're not including moments of eye contact where an individual wasn't making it
indec <- read.csv(paste(base_directory,'/Analyses/individual_ec_timeseries.csv',sep=''))
#partner1 eye contact
indec_p1 <- data.frame(matrix(ncol=4,nrow=0))
colnames(indec_p1) <- c('X','dyad','subject','ind_ec')
for (i in 1:47){
  indec_p1 <- rbind(indec_p1,subset(indec, dyad == dyads[i])[1:600,])
}
#partner2 eye contact
indec_p2 <- data.frame(matrix(ncol=4,nrow=0))
colnames(indec_p2) <- c('X','dyad','subject','ind_ec')
for (i in 1:47){
  indec_p2 <- rbind(indec_p2,subset(indec, dyad == dyads[i])[601:1200,])
}
#make those values zero!
ec_logreg_engage$ind_ec_p1 <- indec_p1$ind_ec
ec_logreg_engage$ec[ec_logreg_engage$ind_ec_p1 == 0] = 0

ec_logreg_engage$ind_ec_p2 <- indec_p2$ind_ec
ec_logreg_engage$ec[ec_logreg_engage$ind_ec_p2 == 0] = 0


# density plot to see dtw data skew after log-correcting
ggplot(ec_logreg_engage,aes(x=synchrony)) + geom_density(alpha = 0.2,fill='purple',size=1) + 
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(), axis.line = element_line(colour = "black",size=1),
        plot.title = element_text(hjust = 0.5),text=element_text(size=20)) + 
  #geom_vline(xintercept = 2,size=1) + 
  scale_y_continuous(name=("Density")) + 
  scale_x_continuous(name = ("synchrony"))

# first perform the true regression
true_engagemodel <- lmer(scale(engage_mean) ~ scale(synchrony) * ec + (1|dyad), 
                         data=ec_logreg_engage)
summary(true_engagemodel)
estimates <- fixef(true_engagemodel)
true_sync_est <- as.numeric(estimates[2])
true_ec_est <- as.numeric(estimates[3])
true_int_est <- as.numeric(estimates[4])

# next, set some vars
nperm <- 5000
dyads <- c('D001','D002','D003','D004','D005','D006','D007','D008',
           'D009','D010','D011','D012','D013','D014','D015','D016',
           'D017','D018','D019','D020','D021','D022','D023','D024',
           'D025','D026','D027','D028','D029','D030','D031','D032',
           'D033','D034','D035','D036','D037','D038','D039','D040',
           'D041','D042','D043','D044','D045','D046','D047')
pseudo_sync_ests <- vector(length=nperm)
pseudo_ec_ests <- vector(length=nperm)
pseudo_int_ests <- vector(length=nperm)

#between subjects permutation
for (i in seq(nperm)){
  if (i%%100 == 0){
    print(paste('running permutation number ',i))
  }
  #so we can reproduce the exact perms
  set.seed(i)
  #shuffle dyads
  newdyads <- sample(dyads)
  newecdyads<- sample(dyads)
  #initialize a temporary synchrony variable
  temp_synchrony <- vector()
  #initialize a temporary eye contact variable
  temp_ec <- vector()
  #re-assign synchrony and eye contact time series to different dyads
  for (j in seq(length(dyads))){
    temp_synchrony <- append(temp_synchrony,ec_dtw_lf_short$synchrony[ec_dtw_lf_short$dyad == newdyads[j]])
    temp_ec <- append(temp_ec,ec_dtw_lf_short$ec[ec_dtw_lf_short$dyad == newecdyads[j]])
  }
  #make a dataframe for these shuffled values
  temp_df <- data.frame(ec_dtw_lf_short$dyad,temp_ec,ec_dtw_lf_short$engage_mean,temp_synchrony)
  
  #run the pseudo model
  pseudo_engagemodel <- lmer(scale(ec_dtw_lf_short.engage_mean) ~ scale(temp_synchrony) * temp_ec + (1|ec_dtw_lf_short.dyad),
                             data=temp_df)
  #save the estimates for comparison
  pseudo_estimates <- fixef(pseudo_engagemodel)
  sync_est <- as.numeric(pseudo_estimates[2])
  ec_est <- as.numeric(pseudo_estimates[3])
  int_est <- as.numeric(pseudo_estimates[4])
  pseudo_sync_ests[i] <- sync_est
  pseudo_ec_ests[i] <- ec_est
  pseudo_int_ests[i] <- int_est
  
}

# determine the probability that our estimate was due to chance
#plug and play for that value if you're in R studio!
groupMeanProb = sum(pseudo_ec_ests > true_ec_est)/nperm

#make a df of the permutation results
pseudo_ests_df <- data.frame(pseudo_sync_ests,pseudo_ec_ests,pseudo_int_ests)
#write the pseudo estimates to a csv
write.csv(pseudo_ests_df,paste(base_directory,'/Analyses/pseudo_ests_betweenSubEngage.csv',sep=''))

#within subjects permutation
pseudo_sync_ests <- vector(length=nperm)
pseudo_ec_ests <- vector(length=nperm)
pseudo_int_ests <- vector(length=nperm)
library(permute)

for (i in seq(nperm)){
  if (i%%100 == 0){
    print(paste('running permutation number ',i))
  }
  #so we can reproduce the exact perms
  set.seed(i)
  #make sure we're shuffling within dyad
  ctrl <- how(blocks=ec_dtw_lf_short$dyad)
  #initialize temporary synchrony and eye contact variables
  temp_synchrony <- vector()
  temp_ec <- vector()
  #shuffle eye contact and synchrony time series within a dyad
  ec_inds <- shuffle(ec_dtw_lf_short$ec,control = ctrl)
  sync_inds <- shuffle(ec_dtw_lf_short$synchrony,control = ctrl)
  #add shuffled data to new variable
  temp_ec <- ec_dtw_lf_short$ec[ec_inds]
  temp_synchrony <- ec_dtw_lf_short$synchrony[sync_inds]
  #make a dataframe for these shuffled values
  temp_df <- data.frame(ec_dtw_lf_short$dyad,temp_ec,ec_dtw_lf_short$engage_mean,temp_synchrony)
  #run the pseudo model
  pseudo_engagemodel <- lmer(scale(ec_dtw_lf_short.engage_mean) ~ scale(temp_synchrony) * temp_ec + (1|ec_dtw_lf_short.dyad),data=temp_df)
  #save the estimates for comparison
  pseudo_estimates <- fixef(pseudo_engagemodel)
  sync_est <- as.numeric(pseudo_estimates[2])
  ec_est <- as.numeric(pseudo_estimates[3])
  int_est <- as.numeric(pseudo_estimates[4])
  pseudo_sync_ests[i] <- sync_est
  pseudo_ec_ests[i] <- ec_est
  pseudo_int_ests[i] <- int_est
  
}

# determine the probability that our estimate was due to chance
#plug and play for that value if you're in R studio!
groupMeanProb = sum(pseudo_ec_ests > true_ec_est)/nperm

#make a df of the permutation results
pseudo_ests_df <- data.frame(pseudo_sync_ests,pseudo_ec_ests,pseudo_int_ests)
#write the pseudo estimates to a csv
write.csv(pseudo_ests_df,paste(base_directory,'/Analyses/pseudo_ests_withinSubEngage.csv',sep=''))
