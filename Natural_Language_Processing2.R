library(data.table)
library(tidyverse)
library(text2vec)
library(caTools)
library(glmnet)


#Q1. Import nlpdata and get familiarized with it.
data <- fread('nlpdata.csv')
data %>% view()

#Q2. Define preprocessing function and tokenization function.
data$V1 <- data$V1 %>% as.character()

# Split data
set.seed(123)
split <- data$Liked %>% sample.split(SplitRatio = 0.8)
train <- data %>% subset(split == T)
test <- data %>% subset(split == F)

it_train <- train$Review %>% 
  itoken(preprocessor = tolower, 
         tokenizer = word_tokenizer,
         ids = train$V1,
         progressbar = F) 

vocab <- it_train %>% create_vocabulary()

vocab %>% 
  arrange(desc(term_count)) %>% 
  head(10)

vectorizer <- vocab %>% vocab_vectorizer()
dtm_train <- it_train %>% create_dtm(vectorizer)

dtm_train %>% dim()
identical(rownames(dtm_train), train$V1)

#Q3. Model: Normal Nfold GLm
glmnet_classifier <- dtm_train %>% 
  cv.glmnet(y = train[['Liked']],
            family = 'binomial', 
            type.measure = "auc",
            nfolds = 10,
            thresh = 0.001,
            maxit = 1000)

#The maximum AUC for train data is 0.87
glmnet_classifier$cvm %>% max() %>% paste("-> Max AUC")

#Vectorizing the test data
it_test <- test$Review %>% tolower() %>% word_tokenizer()

it_test <- it_test %>% 
  itoken(ids = test$V1,
         progressbar = F)

dtm_test <- it_test %>% create_dtm(vectorizer)

#Making predictions for test data
preds <- predict(glmnet_classifier, dtm_test, type = 'response')[,1]

#The AUC for test data is 0.907. There is no overfitting
glmnet:::auc(test$Liked, preds) 

#Q4. Predict model and remove Stopwords.
stop_words <- c("i", "you", "he", "she", "it", "we", "they",
                "me", "him", "her", "them",
                "my", "your", "yours", "his", "our", "ours",
                "myself", "yourself", "himself", "herself", "ourselves",
                "the", "a", "an", "and", "or", "on", "by", "so",
                "from", "about", "to", "for", "of", 
                "that", "this", "is", "are")

#Q5. Create DTM for Training and Testing with new pruned vocabulary.
vocab <- it_train %>% create_vocabulary(stopwords = stop_words)

pruned_vocab <- vocab %>% 
  prune_vocabulary(term_count_min = 10, 
                   doc_proportion_max = 0.5,
                   doc_proportion_min = 0.001)

pruned_vocab %>% 
  arrange(desc(term_count)) %>% 
  head(10) 

#Q6. Apply vectorizer.
vectorizer <- pruned_vocab %>% vocab_vectorizer()

#Q7. Give interpretation for model.
dtm_train <- it_train %>% create_dtm(vectorizer)
dtm_train %>% dim()

#Creating the model
glmnet_classifier <- dtm_train %>% 
  cv.glmnet(y = train[['Liked']], 
            family = 'binomial',
            type.measure = "auc",
            nfolds = 10,
            thresh = 0.001,
            maxit = 1000)

#After applying pruning and removing stop words, our train AUC has dropped from 0.87 to 0.827
glmnet_classifier$cvm %>% max() %>% paste("-> Max AUC")

#The AUC for test data has also dropped from 0.91 to 0.847.
dtm_test <- it_test %>% create_dtm(vectorizer)
preds <- predict(glmnet_classifier, dtm_test, type = 'response')[,1]
glmnet:::auc(test$Liked, preds) 
