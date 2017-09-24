install.packages("ddply")
install.packages("magrittr")
install.packages("ggplot2")

library(ddplyr)
library(magrittr)
library(ggplot2)

ggplot(data=res1.s1,aes(yhat,cnt,color=factor(yhat))) + geom_point() + facet_grid(~y)


