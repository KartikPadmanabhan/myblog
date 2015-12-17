sayHello <- function(){
library(devtools)
library(knitrBootstrap)
library(rmarkdown)
render('non_personalized_recommenders.Rmd', 'knitrBootstrap::bootstrap_document')
}

sayHello()
