
FLAGS <- flags(
  flag_integer("units", 10)
)


sae_input = layer_input(shape = (ncol(xgene)-1))
sae_output = sae_input %>%
  encoder1() %>%
  encoder2() %>%
  encoder3() %>%
  layer_dense(FLAGS$units,activation = "relu") %>%
  layer_dense(1,activation = "sigmoid")
sae = keras_model(sae_input, sae_output)
summary(sae)

sae %>% compile(
  optimizer = "rmsprop",
  loss = 'binary_crossentropy',
  metric = "acc"
)

sae %>% fit(
  x=xtrain,
  y=ylabels,
  epochs = 15,
  batch_size=64,
  validation_split = 0.2
)

yhat <- predict(sae,as.matrix(xtest))

yhatclass<-as.factor(ifelse(yhat<0.5,0,1))
table(yhatclass, ylabelstest)
confusionMatrix(yhatclass,as.factor(ylabelstest))
roc_sae_test <- roc(response = ylabelstest, predictor =yhat)
plot(roc_sae_test, col = "blue", print.auc=TRUE)
legend("bottomright", legend = c("sae"), lty = c(1), col = c("blue"))