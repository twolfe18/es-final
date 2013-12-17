
d = read.table("learning.txt")
pdf("learningCurve.pdf")
plot(d$V1, xlab="Training Epoch", ylab="Dev Set Error",
  main="Learning Curves", ylim=c(0.1,0.46))
points(d$V2, col="blue")
dev.off()


