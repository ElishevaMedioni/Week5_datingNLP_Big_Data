# Elisheva Medioni 337628390

# installing packages
install.packages("parallel")
install.packages("stringr")
install.packages("dplyr")
install.packages("ggplot2")
install.packages("mosaic")
install.packages("xtable")
install.packages("gridExtra")
install.packages("stopwords")
install.packages("quanteda")
install.packages("lattice")
install.packages("caret")
install.packages("rpart")

library(parallel)
library(stringr) 
library(dplyr) 
library(ggplot2)
library(mosaic)
library(xtable)
library(gridExtra)
library(stopwords)
library(quanteda)
library(lattice)
library(caret)
library(rpart)



profiles <- read.csv( file.path( '/Users/Elish_1/Downloads', 'okcupid_profiles.csv' ), header=TRUE, stringsAsFactors=FALSE)
str(profiles)

profiles.subset <- filter(profiles, height>=55 & height <=80)

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



# Create our first bag-of-words model.
all.tokens.dfm <- dfm(all.tokens, tolower = FALSE)


# clear some space
rm(all.tokens)
gc()

all.tokens.dfm

sparsity(all.tokens.dfm)
# meaning that 99.90% of the cells are zeros. Even if you could
# create the data frame, fitting a model is not going to work
# because of this extreme lack of information in the features.
# Solution? Trim some features.

dfm.trimmed <- dfm_trim(all.tokens.dfm, min_docfreq = 10, min_termfreq = 20, verbose = TRUE)
dfm.trimmed

# top-50 frequent features
topfeatures(dfm.trimmed, 50)

# Top features of individual documents. WARNING - takes long to run
# top.features <- topfeatures(dfm.trimmed, n = 7, groups = docnames(dfm.trimmed))
#save(top.features, file=file.path( 'dating', "Week5_datingNLP.rdata") )

# features by document frequencies
tail(topfeatures(dfm.trimmed, scheme = "docfreq", n = 200))

#tstat_freq <- textstat_frequency(dfm.trimmed, n = 5, groups = lang)
#head(tstat_freq, 20)

# Transform to a matrix and inspect.
all.tokens.matrix <- as.matrix(dfm.trimmed)
#View(all.tokens.matrix[1:20, 1:100])
dim(all.tokens.matrix)


# Investigate the effects of stemming.
# [A]
colnames(all.tokens.matrix)[1:50]

# [B]
sort(colnames(all.tokens.matrix))[1:100]

# clear some space
#rm(all.tokens.matrix)
gc()

# Convert our class label into a factor.
Label <- as.factor(profiles$sex)

cv.cntrl <- trainControl(method = "repeatedcv", number = 10,
                         repeats = 3)

data_frame_test <- convert(dfm.trimmed, to = "data.frame")
data_frame_test_subset <- subset(data_frame_test,age==22 )

object.size(data_frame_test)
object.size(data_frame_test_subset)

rm(data_frame_test)

rpart.cv.1 <- train(Label ~ . , 
                    data = data_frame_test_subset, 
                    method = "rpart", 
                    trControl = cv.cntrl, 
                    tuneLength = 7, 
                    maxdepth=3)

dimnames(data_frame_test_subset)

# Per best practices, we will leverage cross validation (CV) as
# the basis of our modeling process. Using CV we can create
# estimates of how well our model will do in Production on new,
# unseen data. CV is powerful, but the downside is that it
# requires more processing and therefore more time.
#
# If you are not familiar with CV, consult the following
# Wikipedia article:
#
#   https://en.wikipedia.org/wiki/Cross-validation_(statistics)
#

#--make a data frame of the DFM (as.data.frame)
#--add labels of male / female to the DFM data frame
# Setup a the feature data frame with labels.
all.tokens.df <- cbind(Label = profiles$sex, convert(dfm.trimmed, to = "data.frame"))
all.tokens.df$

#all.tokens.df.subset <- subset(all.tokens.df,age==22 )
#object.size(all.tokens.df.subset)
#object.size(all.tokens.df)


# Often, tokenization requires some additional pre-processing
names(all.tokens.df)[c(146, 148, 235, 238)]


# Cleanup column names. -- rectify the names of the variables (make.names)
names(all.tokens.df) <- make.names(names(all.tokens.df))
#all.tokens[[357]]


# Use caret to create stratified folds for 10-fold cross validation repeated 
# 3 times (i.e., create 30 random stratified samples)
#--10-fold cross validation 3 times (createMultiFolds)
set.seed(48743)
cv.folds <- createMultiFolds(profiles$sex, k = 10, times = 3)

cv.cntrl <- trainControl(method = "repeatedcv", number = 10,
                         repeats = 3, index = cv.folds)

#train a decision tree model
rpart.cv.1 <- train(Label ~ ., 
                    data = all.tokens.df, 
                    method = "rpart", 
                    trControl = cv.cntrl, 
                    tuneLength = 7, 
                    maxdepth=3)
rpart.cv.1

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

object.size(all.tokens.tfidf)
#----------------------Test

# mysample <- all.tokens.df[sample(1:nrow(all.tokens.df), 50,
#                           replace=FALSE),]
# 
# decision_tree_model <- train(
#   Label~., data=mysample, 
#   trControl=cv.cntrl, 
#   method = "rpart", 
#   maxdepth=3,
#   na.action = na.omit, 
#   tuneLength = 7
# )
# 
# decision_tree_model

library(rpart)


fit <- rpart(Label~., 
             method = "anova", data = all.tokens.df.subset)

# Output to be present as PNG file
png(file = "decTreeGFG.png", width = 600, 
    height = 600)

# Plot
plot(fit, uniform = TRUE,
     main = "Decision 
                 Tree using Regression")
text(fit, use.n = TRUE, cex = .7)

# Saving the file
dev.off()

# Print model
print(fit)

#----------------------

library(datasets)
library(caTools)
library(party)
library(dplyr)
library(magrittr)

sample_data = sample.split(all.tokens.df, SplitRatio = 0.8)
train_data <- subset(all.tokens.df, sample_data == TRUE)
test_data <- subset(all.tokens.df, sample_data == FALSE)

model<- ctree(profiles$sex ~ ., train_data)
plot(model)

options(expressions = 5e5)

decision_tree_model <- train(
  Label~., data=all.tokens.df.subset, 
  trControl=cv.cntrl, 
  method = "rpart", 
  maxdepth=3,
  na.action = na.omit
)


target <- as.factor(profiles$sex)

rpart.cv.1 <- train(x=t(all.tokens.tfidf), y=target, method = "rpart", 
                    trControl = cv.cntrl, tuneLength = 7)

# rpart.cv.1 <- train(Label ~ ., data = all.tokens.tfidf, method = "rpart", 
#                     trControl = cv.cntrl, tuneLength = 7)

#--train a decision tree model (train( ..., method = “rpart” ))
# As our data is non-trivial in size at this point, use a single decision
# tree alogrithm as our first model. We will graduate to using more 
# powerful algorithms later when we perform feature extraction to shrink
# the size of our data.

rpart.cv.1 <- train(all.tokens.tfidf, profiles$sex, method = "rpart", 
                    trControl = cv.cntrl, tuneLength = 7)


#-----------------------Error------ Error: protect(): protection stack overflow
rpart.cv.1



system("--max-ppsize=500000")
