library(keras)
library(magrittr)

# load ----
mnist <- keras::dataset_mnist()

# reshape ----
train_x <- mnist$train$x
train_y <- mnist$train$y
test_x <- mnist$test$x
test_y <- mnist$test$y

# reshape array ----
train_x %<>% array_reshape(c(nrow(train_x), 784))
test_x %<>% array_reshape(c(nrow(test_x), 784))

# normalize ----
train_x <- train_x/255
test_x <- test_x/255

# one hot encoding ----
train_y <- to_categorical(train_y)
test_y <- to_categorical(test_y)

# define model ----
model <- keras_model_sequential()
model %>% layer_dense(units = 128, activation = "relu", input_shape = c(784)) %>% 
                layer_dropout(rate = 0.3) %>% 
                layer_dense(units = 64, activation = "relu") %>% 
                layer_dropout(0.2) %>% 
                layer_dense(units = 16, activation = "relu") %>% 
                layer_dense(units = 10, activation = "softmax")

# compile ----
model %>% compile(
                loss = "categorical_crossentropy",
                optimizer = optimizer_adam(),
                metrics = c("accuracy")
)

# train ----
history <- model %>% fit(
                train_x, train_y,
                epochs = 30,
                batch_size = 128,
                validation_split = 0.2
)

# evaluation ----
model %>% evaluate(test_x, test_y)

# predict classes ----
model %>% predict_classes(test_x)