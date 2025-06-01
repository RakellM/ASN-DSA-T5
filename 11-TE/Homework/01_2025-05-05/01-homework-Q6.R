# LIBRARY
library(haven)
library(dplyr)
library(car)

getwd()

data <- read_sas("ASN-DSA-T5/11-TE/data/normtemp.sas7bdat")

summary(data)

hist(data$BodyTemp)
boxplot(data$BodyTemp)

# H0: mu = 98.6
# H1: mu <> 98.6
mu = 98.6
alpha = 0.01
H1 = "different"

# Test Normality
sw_test = shapiro.test(data$BodyTemp)
# H0: sample comes from a normal population
# H1: sample does not come from a normal population
sw_test


# Test t for one sample
# H0: mu = m0 (mu is our famous mean mi and m0 is the number we want to measure)
# H1: mu <> m0

t.test(data$BodyTemp, mu=mu)

# Calculate 99% confidence interval for the mean
conf_level <- 0.99
t_test <- t.test(data$BodyTemp, mu = mu, conf.level = conf_level)
ci <- t_test$conf.int

# Plot confidence interval and mu
plot(1, type = "n", xlim = c(0.5, 1.5), ylim = range(c(ci, mu)), xaxt = "n",
    xlab = "", ylab = "Body Temperature", main = "99% Confidence Interval for Mean Body Temp")
arrows(1, ci[1], 1, ci[2], code = 3, angle = 90, length = 0.1, lwd = 2, col = "blue")
points(1, mean(data$BodyTemp), pch = 19, col = "red")
abline(h = mu, col = "darkgreen", lty = 2)
legend("bottomright", legend = c("99% CI", "Sample Mean", "mu = 98.6"),
      col = c("blue", "red", "darkgreen"), lty = c(1, NA, 2), pch = c(NA, 19, NA))

### Part 2
# Test if the body temperature is different for Male and Female
# H0: muA = muB
# H1: muA <> muB

male <- data %>%
  filter(Gender=="Male")

female <- data %>%
  filter(Gender=="Female")

# Test Normality
## Male
shapiro.test(male$BodyTemp)

## Female
shapiro.test(female$BodyTemp)

par(mfrow = c(1, 2))
hist(male$BodyTemp, main = "Male Body Temp", xlab = "Body Temperature", col = "lightblue", breaks = 10)
hist(female$BodyTemp, main = "Female Body Temp", xlab = "Body Temperature", col = "pink", breaks = 10)
par(mfrow = c(1, 1))

boxplot(BodyTemp ~ Gender, data = data,
    main = "Body Temperature by Gender",
    xlab = "Gender",
    ylab = "Body Temperature",
    col = c("pink","lightblue"))

# Test for equal variances

leveneTest(BodyTemp ~ Gender, data = data)

# Test t for two independent samples
t.test(female$BodyTemp, male$BodyTemp, paired=FALSE, conf.level=0.99)
