
res = read.csv("t1.csv")
res1 = subset(res,as.character(y) != as.character(yhat))
res2 = res1 %>% select(y,yhat) %>% group_by(y,yhat) %>% summarise(cnt=length(yhat)) %>% arrange(y,yhat,desc(cnt))

res2[as.character(res2$y) == '_tag_isu',]

