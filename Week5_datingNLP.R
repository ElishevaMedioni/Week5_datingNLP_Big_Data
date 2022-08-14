# Elisheva Medioni 337628390
# Hila Avraham 209290246
# Ruth Ovadia 212121375

# -- installing packages --
# install.packages("parallel")
# install.packages("stringr")
# install.packages("dplyr")
# install.packages("ggplot2")
# install.packages("mosaic")
# install.packages("xtable")
# install.packages("gridExtra")
# install.packages("stopwords")
# install.packages("quanteda")
# install.packages("lattice")
# install.packages("caret")
# install.packages("rpart")
# install.packages("limma")
# install.packages("edgeR")

# load libraries
library(stringr)
library(dplyr)
library(ggplot2)
library(mosaic)
library(dplyr)
library(stringr)
library(xtable)
library(gridExtra)
library(stopwords)
library(quanteda)
library(caret)
library(rpart)
library(rpart.plot)
library(caTools)
library(recipes)
library(ClusterR)
library(cluster)
library(Rtsne)

# Set rounding to 2 digits
options(digits=2)

# Load texts (read.csv)
profiles <- read.csv( file.path('/Users/Elish_1/Downloads', 'okcupid_profiles.csv'), header=TRUE, stringsAsFactors=FALSE)
n <- nrow(profiles)

str(profiles)

essays <- select(profiles, starts_with("essay"))
essays <- apply(essays, MARGIN = 1, FUN = paste, collapse=" ")

html <- c( "<a[^>]+>", "class=[\"'][^\"']+[\"']", "&[a-z]+;", "\n", "\\n", "<br ?/>", "</[a-z]+ ?>" )
stop.words <-  c( "a", "am", "an", "and", "as", "at", "are", "be", "but", "can", "do", "for", "have", "i'm", "if", "in", "is", "it", "like", "love", "my", "of", "on", "or", "so", "that", "the", "to", "with", "you", "i" )

html.pat <- paste0( "(", paste(html, collapse = "|"), ")" )
html.pat
stop.words.pat <- paste0( "\\b(", paste(stop.words, collapse = "|"), ")\\b" )
stop.words.pat
essays <- str_replace_all(essays, html.pat, " ")
essays <- str_replace_all(essays, stop.words.pat, " ")

# Clean texts, tokenize, remove stop words (token, tokens_select)
# Tokenize essay texts
all.tokens <- tokens(essays, what = "word",
                     remove_numbers = TRUE, remove_punct = TRUE,
                     remove_symbols = TRUE, remove_hyphens = TRUE)

# Take a look at a specific message and see how it transforms.
all.tokens[[357]]

# Lower case the tokens.
all.tokens <- tokens_tolower(all.tokens)

# Use quanteda's built-in stopword list for English.
# NOTE - You should always inspect stopword lists for applicability to
#        your problem/domain.
all.tokens <- tokens_select(all.tokens, stopwords(),
                            selection = "remove")
all.tokens[[357]]


# Perform stemming on the tokens.
all.tokens <- tokens_wordstem(all.tokens, language = "english")
# remove single-word tokens after stemming. Meaningless
all.tokens <- tokens_select(all.tokens, "^[a-z]$",
                            selection = "remove", valuetype = "regex")
all.tokens[[357]]

# DFM (dfm)
# Create a bag-of-words model (document-term frequency matrix)
all.tokens.dfm <- dfm(all.tokens, tolower = FALSE)

# clear some space
rm(all.tokens)

all.tokens.dfm

sparsity(all.tokens.dfm)
# meaning that 99.90% of the cells are zeros. Even if you could
# create the data frame, fitting a model is not going to work
# because of this extreme lack of information in the features.
# Solution? Trim some features.

dfm.trimmed <- dfm_trim(all.tokens.dfm, min_docfreq = 25, min_termfreq = 35, verbose = TRUE)
dfm.trimmed


# Transform to a matrix and inspect.
all.tokens.matrix <- as.matrix(dfm.trimmed)
object.size(all.tokens.matrix)
#View(all.tokens.matrix[1:20, 1:100])
dim(all.tokens.matrix)


# Investigate the effects of stemming.
# [A]
colnames(all.tokens.matrix)[1:50]

# [B]
sort(colnames(all.tokens.matrix))[1:100]


# Setup a the feature data frame with labels.
all.tokens.df <- cbind(Label = profiles$sex, data.frame(dfm.trimmed))


# Often, tokenization requires some additional pre-processing
names(all.tokens.df)[c(146, 148, 235, 238)]


# Cleanup column names.
names(all.tokens.df) <-make.names(names(all.tokens.df))

# Use caret to create a 70%/30% stratified split. Set the random
# seed for reproducibility.
set.seed(32984)
indexes <- createDataPartition(all.tokens.df$Label, times = 1,
                               p = 0.7, list = FALSE)

train <- all.tokens.df[indexes,]
test <- all.tokens.df[-indexes,]

# remove not numeric feature
train <- subset( train, select = -c(doc_id))
test <- subset( test, select = -c(doc_id))

object.size(train)
object.size(test)

# Verify proportions.
prop.table(table(train$Label))
prop.table(table(test$Label))


# Use caret to create stratified folds for 10-fold cross validation repeated
# 3 times (i.e., create 30 random stratified samples)
set.seed(48743)
cv.folds <-createMultiFolds(labels, k = 10, times = 3)

cv.cntrl <-trainControl(
  method = "repeatedcv",
  number = 10,
  repeats = 3,
  index = cv.folds
)

# clear some space before the train
gc()
rm(dfm.trimmed)
rm(cv.folds)
rm(all.tokens.dfm)

# Time the code execution
start.time <- Sys.time()

#train a decision tree model
tree <- rpart(Label~., data=train, cp=.02)
total.time <- Sys.time() - start.time
total.time

# print the tree
tree

#test the model by calculating a confusion matrix
pred <- predict(tree, newdata= test, type="class")

table(pred,test$Label)

#plot the tree
#rpart.plot(tree, box.palette="RdBu", shadow.col="gray", nn=TRUE)

# clear some space
gc()
rm(train)
rm(test)

# The use of Term Frequency-Inverse Document Frequency (TF-IDF) is a 
# powerful technique for enhancing the information/signal contained
# within our document-frequency matrix. Specifically, the mathematics
# behind TF-IDF accomplish the following goals:
#    1 - The TF calculation accounts for the fact that longer 
#        documents will have higher individual term counts. Applying
#        TF normalizes all documents in the corpus to be length 
#        independent.
#    2 - The IDF calculation accounts for the frequency of term
#        appearance in all documents in the corpus. The intuition 
#        being that a term that appears in every document has no
#        predictive power.
#    3 - The multiplication of TF by IDF for each cell in the matrix
#        allows for weighting of #1 and #2 for each cell in the matrix.


# Our function for calculating relative term frequency (TF)
term.frequency <- function(row) {
  row / sum(row)
}

# Our function for calculating inverse document frequency (IDF)
inverse.doc.freq <- function(col) {
  corpus.size <- length(col)
  doc.count <- length(which(col > 0))
  
  log10(corpus.size / doc.count)
}

# Our function for calculating TF-IDF.
tf.idf <- function(x, idf) {
  x * idf
}


# First step, normalize all documents via TF.
all.tokens.df1 <- apply(all.tokens.matrix, 1, term.frequency)
dim(all.tokens.df1)
View(all.tokens.df1[1:20, 1:100])

# Second step, calculate the IDF vector that we will use - both
# for training data and for test data!
all.tokens.idf <- apply(all.tokens.matrix, 2, inverse.doc.freq)
str(all.tokens.idf)


# Lastly, calculate TF-IDF for our training corpus.
all.tokens.tfidf <-  apply(all.tokens.df1, 2, tf.idf, idf = all.tokens.idf)
dim(all.tokens.tfidf)
View(all.tokens.tfidf[1:25, 1:25])

#save(file='Week5_datingNLP.rdata', tree, all.tokens.tfidf)

rm(all.tokens.df1)
rm(all.tokens.idf)
rm(all.tokens.tfidf)


# Remove the batch effect: find and eliminate “male words” and “female words”

male.words <- subset(essays, profiles$sex == "m") %>%
  str_split(" ") %>%
  unlist() %>%
  table() %>%
  sort(decreasing=TRUE) %>%
  names()

female.words <- subset(essays, profiles$sex == "f") %>%
  str_split(" ") %>%
  unlist() %>%
  table() %>%
  sort(decreasing=TRUE) %>%
  names()

# Top 25 male words:
print( male.words[1:25] )
# Top 25 female words
print( female.words[1:25] )

# Words in the males top 500 that weren't in the females' top 500:
male_words<-as.data.frame(setdiff(male.words[1:500], female.words[1:500]))
# Words in the female top 500 that weren't in the males' top 500:
female_words<-as.data.frame(setdiff(female.words[1:500], male.words[1:500]))


#remove the male/female words from the df
all.tokens.df<-all.tokens.df[,!names(all.tokens.df)%in% male_words]

all.tokens.df<-all.tokens.df[,!names(all.tokens.df)%in% female_words]

# clear some memory
rm(male_words)
rm(female_words)
rm(male.words)
rm(female.words)

#remove the labels and non numeric features
all.tokens.df.sub<-all.tokens.df[,-c(1,2)]


# Cluster the applicants to 2,3,4 and 10 clusters (kmeans)

#k=2
kmeans_word_2 = kmeans(all.tokens.df.sub, centers = 2, nstart = 50)
fviz_cluster(kmeans_word_2, data = all.tokens.df.sub)

#k=3
kmeans_word_3 = kmeans(all.tokens.df.sub, centers = 3, nstart = 50)
fviz_cluster(kmeans_word, data = all.tokens.df.sub)

#k=4
kmeans_word_4 = kmeans(all.tokens.df.sub, centers = 4, nstart = 50)
fviz_cluster(kmeans_word, data = all.tokens.df.sub)

#k=10
kmeans_word_10 = kmeans(all.tokens.df.sub, centers = 10, nstart = 50)
fviz_cluster(kmeans_word, data = all.tokens.df.sub)


# plot T-SNE or PCA of the cluster results
set.seed(42) # Set a seed if you want reproducible results
tsne_out <- Rtsne(all.tokens.matrix) # Run TSNE

# Show the objects in the 2D tsne representation
plot(tsne_out$Y,col=all.tokens.df$Label)


pca_word = prcomp(all.tokens.df.sub, center = TRUE, scale = TRUE)
summary(pca_word)




                    
