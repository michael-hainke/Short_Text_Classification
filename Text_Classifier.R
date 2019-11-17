
#################
### Packages  ###
#################

library(quanteda)     # text classification package
library(RColorBrewer) # colors for word clouds
library(tidyverse)    # data manipulation

##################
### Paramters  ###
##################

set.seed(1912)
train_prop <- 0.7 # % of data to use for training
threshold <- 0.7  # set probability threshold for classification

###########################
### Read and Prep Data  ###
###########################

# choose which file to use
df <- read.table("SMSSpamCollection.txt", header=FALSE, sep="\t", quote="", stringsAsFactors=FALSE) # Ham/Spam test data
df <- read.csv("test_reviews_labelled.csv", header=FALSE, stringsAsFactors=FALSE) # Product review test data

# prepare data
names(df) <- c("Label", "Text")                         # add column labels
df <- df[sample(nrow(df)),]                             # randomize data
df <- df %>% filter(Text != '') %>% filter(Label != '') # filter blank data

# create document corpus  
df_corpus <- corpus(df$Text)   # convert Text to corpus 
docvars(df_corpus) <- df$Label # add classification label as docvar

# build document term matrix
df_dfm <- dfm(df_corpus, tolower = TRUE)                                # convert to dtm
df_dfm <- dfm_wordstem(df_dfm)                                          # stem words
df_dfm <- dfm_trim(df_dfm, min_termfreq = 5, min_docfreq = 3)           # remove low frequency occurence words
df_dfm <- dfm_tfidf(df_dfm, scheme_tf = "count", scheme_df = "inverse") # tf-idf weighting

# split data train/test
size <- dim(df)
train_end <- round(train_prop*size[1])
test_start <- train_end + 1
test_end <- size[1]

df_train <- df[1:train_end,]
df_test <- df[test_start:test_end,]

df_dfm_train <- df_dfm[1:train_end,]
df_dfm_test <- df_dfm[test_start:test_end,]

#############################
### Train and Test Model  ###
#############################

# build model with training set
df_classifier <- textmodel_nb(df_dfm_train, df_train$Label)

# test model with testing set - Option 1 defaults parameters
df_predictions <- predict(df_classifier, newdata = df_dfm_test)               # predict categories

# test model with testing set - Option 2 with custom threshold
df_predictions <- predict(df_classifier,newdata = df_dfm_test,type = 'prob')  # predict probabilities
df_predictions <- ifelse(df_predictions > threshold,1,0)                      # classify based on provided threshold

#############################
### Output Model Results  ###
#############################

conf_matrix <- table(df_predictions, df_test$Label)
accuracy <- (conf_matrix[1,1] + conf_matrix[2,2]) / sum(conf_matrix)
precision <- conf_matrix[2,2] / sum(conf_matrix[2,])
recall <- conf_matrix[2,2] / sum(conf_matrix[,2])
f1_score <- 2 * ((precision * recall) / (precision + recall))

cat("Confidence Matrix:")
conf_matrix
cat("Accuracy: ", accuracy)
cat("Precision: ", precision)
cat("Recall: ", recall)
cat("F1 Score: ", f1_score)

##########################
### Output Word Cloud  ###
##########################

category <- "spam"  # specify which group of documents to produce wordcloud for

df_plot <- corpus_subset(df_corpus, docvar1 == category)
df_plot <- tokens(df_plot, what = c("word"), remove_numbers = TRUE, remove_punct = TRUE, remove_twitter = TRUE)
df_plot <- dfm(df_plot, tolower = TRUE, remove=stopwords(source = "smart"))
df_col <- brewer.pal(10, "BrBG")
df_cloud <- textplot_wordcloud(df_plot, min_count = 16, color = df_col)
title("Wordcloud", col.main = "grey14")
