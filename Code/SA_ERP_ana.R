#SA_ERP_ana.R

base_directory = '/Users/sophie/Dropbox/EyeContactinConversation'

#erp dataframe
erp_data <- read.csv(paste(base_directory,'/Analyses/SA_synchrony_ERPana.csv',sep=''))
erp_data$dyad <- as.factor(erp_data$dyad)
erp_data$time <- as.factor(erp_data$time)
contrasts(erp_data$time) <- contr.poly(9)

erponsetmodel <- lm(onset ~ time,data = erp_data)
summary(erponsetmodel)
erpoffsetmodel <- lm(offset ~ time,data = erp_data)
summary(erpoffsetmodel)

##supplementary ERP Analyses
erp_30trialmin<- read.csv(paste(base_directory,'/Supplement/SA_synchrony_ERPana_30trialmin.csv',sep=''))
erp_30trialmin$dyad <- as.factor(erp_30trialmin$dyad)
erp_30trialmin$time <- as.factor(erp_30trialmin$time)
contrasts(erp_30trialmin$time) <- contr.poly(9)

erponset_supp <- lm(onset ~ time,data = erp_30trialmin)
summary(erponset_supp)
erpoffset_supp <- lm(offset ~ time,data = erp_30trialmin)
summary(erpoffset_supp)

trialmins = c(20,25,30,35,40,45)
onset_ests = list()
offset_ests = list()
onset_ts = list()
offset_ts = list()
onset_ps = list()
offset_ps = list()

for (trial in seq(trialmins)){
  erp_trialmin_data <- read.csv(paste(base_directory,'/Supplement/SA_synchrony_ERPana_',
                            trialmins[trial],'trialmin.csv',sep=''))
  erp_trialmin_data$dyad <- as.factor(erp_trialmin_data$dyad)
  erp_trialmin_data$time <- as.factor(erp_trialmin_data$time)
  contrasts(erp_trialmin_data$time) <- contr.poly(9)

  erptrialminonset <- lm(onset ~ time,data = erp_trialmin_data)
  coef_onset <- summary(erptrialminonset)$coefficients
  onset_ests[trial] <- coef_onset["time.Q","Estimate"]
  onset_ts[trial] <- coef_onset["time.Q","t value"]
  onset_ps[trial] <- coef_onset["time.Q","Pr(>|t|)"]

  erptrialminoffset <- lm(offset ~ time,data = erp_trialmin_data)
  coef_offset <- summary(erptrialminoffset)$coefficients
  offset_ests[trial] <- coef_offset["time.Q","Estimate"]
  offset_ts[trial] <- coef_offset["time.Q","t value"]
  offset_ps[trial] <- coef_offset["time.Q","Pr(>|t|)"]
}
