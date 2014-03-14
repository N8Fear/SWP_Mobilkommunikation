#!/usr/bin/Rscript
#
# arguments:
#	* args[1] = real data
# 	* args[2] = histogram data
# 	* args[3] = HMM data
#
# example:
#	./safest_eval.r real.csv histo.csv hmm.csv	

# get arguments
args <- commandArgs(trailingOnly = TRUE)

# load data
data_r <- read.csv(args[1], header=FALSE, sep=";")
data_t1 <- read.csv(args[2], header=FALSE, sep=";")
data_t2 <- read.csv(args[3], header=FALSE, sep=";")

# name data
names(data_r) = c("Time", "People", "Alg")
names(data_t1) = c("Time", "PeopleT1")
names(data_t2) = c("Time", "PeopleT2")

# calculate delta offsets
data_t1$PeopleT1 = data_t1$PeopleT1-data_r$People
data_t2$PeopleT2 = data_t2$PeopleT2-data_r$People

# merge data
graph_data = merge(x = data_t1, y = data_t2, by = "Time", all.y=TRUE)

# get some info about the data
y_max = max(graph_data$PeopleT1, graph_data$PeopleT2)
y_min = min(graph_data$PeopleT1, graph_data$PeopleT2)
duration = nrow(graph_data)

# define output file
pdf("safest_plot.pdf", width=12)

# plot it
plot(x=graph_data$Time, y=graph_data$PeopleT1, xlab="Sekunden", ylab="Anzahl von Personen",
     main="Abstand zum Sollwert in Personen im Verlauf der Zeit.", 
     type="b", col="red", ylim=c(y_min-2,y_max+2))
lines(x=graph_data$Time, y=graph_data$PeopleT2, col="blue", type="b", pch=22)

# add legend and grid for easy readability
legend("topleft", c("Histogramm","HMM"), cex=1.2, 
       col=c("red","blue"), pch=c(21,22), lty=c(1,1))
abline(v=(seq(2, duration, by=10)), col="lightgray", lty="dotted")
abline(h=(seq(y_min-2, y_max+2, 1)), col="lightgray", lty="dotted")
abline(h=0, col="gray")

# close output file
graphics.off()
