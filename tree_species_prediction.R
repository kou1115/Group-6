# 树种预测任务 - 使用随机森林模型
# 目标：预测8种树种在不同地块中的出现概率

# 加载必要的库
library(randomForest)
library(dplyr)
library(readr)
library(ggplot2)        # 可视化
library(corrplot)       # 相关性分析可视化
library(VIM)            # 缺失值可视化
library(caret)          # 交叉验证和模型评估
library(pROC)           # ROC曲线分析
library(gridExtra)      # 多图布局
library(RColorBrewer)   # 颜色调色板
library(viridis)        # 颜色调色板
library(sf)             # 地理空间数据处理
library(maps)           # 地图数据
library(mapdata)        # 扩展地图数据
library(car)            # 方差分析
library(lmtest)         # 线性模型检验
library(broom)          # 整理模型输出
library(reshape2)       # 数据重构
library(knitr)          # 报告生成
library(pheatmap)       # 热图
library(tidyr)          # 数据整理
library(tibble)         # 数据框处理

# 设置图形输出参数
options(warn = -1)  # 暂时关闭警告
par(family = "sans")  # 设置字体

# 设置随机种子以确保结果的可重复性
set.seed(42)

# 读取数据
train_data <- read_csv("train.csv")
test_data <- read_csv("test.csv")

# 查看数据结构
cat("训练数据维度:", dim(train_data), "\n")
cat("测试数据维度:", dim(test_data), "\n")

# 查看树种类型
unique_species <- unique(train_data$Species)
cat("树种类型:", length(unique_species), "\n")
print(unique_species)

# 数据预处理
# 选择特征列（排除id、plot、Species、pres.abs）
feature_cols <- c("long", "lat", "disturb", "rainann", "soildepth", 
                  "soilfert", "tempann", "topo", "easting", "northing")

# 检查缺失值
cat("训练数据中的缺失值:\n")
print(sapply(train_data[feature_cols], function(x) sum(is.na(x))))

cat("测试数据中的缺失值:\n")
print(sapply(test_data[feature_cols], function(x) sum(is.na(x))))

# 处理缺失值（如果有）
if(any(sapply(train_data[feature_cols], function(x) sum(is.na(x))) > 0)) {
  cat("检测到缺失值，进行处理...\n")
  # 用中位数填充数值型变量的缺失值
  for(col in feature_cols) {
    if(sum(is.na(train_data[[col]])) > 0) {
      train_data[[col]][is.na(train_data[[col]])] <- median(train_data[[col]], na.rm = TRUE)
    }
    if(sum(is.na(test_data[[col]])) > 0) {
      test_data[[col]][is.na(test_data[[col]])] <- median(test_data[[col]], na.rm = TRUE)
    }
  }
}

# 数据完整性检查
cat("数据完整性检查:\n")
cat("训练数据行数:", nrow(train_data), "\n")
cat("测试数据行数:", nrow(test_data), "\n")
cat("特征数量:", length(feature_cols), "\n")
cat("树种数量:", length(unique_species), "\n")

# ======================================================================
# 修改1: 使用logloss作为评估指标
# ======================================================================

# 定义logloss函数
logloss <- function(actual, predicted) {
  # 确保预测值在[0,1]范围内
  predicted <- pmax(pmin(predicted, 1 - 1e-15), 1e-15)
  return(-mean(actual * log(predicted) + (1 - actual) * log(1 - predicted)))
}

# ======================================================================
# 修改2: 每个树种单独预测再整合
# ======================================================================

# 存储每个树种的预测结果
species_predictions <- list()
species_models <- list()

# 对每个树种单独训练模型并进行预测
for(species in unique_species) {
  cat("\n正在处理树种:", species, "\n")
  
  # 准备该树种的训练数据
  species_train <- train_data[train_data$Species == species, ]
  
  # 准备特征和目标变量
  X_train <- species_train[, feature_cols]
  y_train <- species_train$pres.abs
  
  # 训练随机森林模型
  rf_model <- randomForest(x = X_train, y = as.factor(y_train), 
                          ntree = 500,
                          mtry = sqrt(length(feature_cols)),
                          importance = TRUE)
  
  # 保存模型
  species_models[[species]] <- rf_model
  
  # 查看模型性能
  cat("模型OOB错误率:", rf_model$err.rate[rf_model$ntree, "OOB"], "\n")
  
  # 准备该树种的测试数据
  species_test <- test_data[test_data$Species == species, ]
  X_test <- species_test[, feature_cols]
  
  # 进行预测（获取概率）
  species_pred <- predict(rf_model, X_test, type = "prob")
  
  # 提取出现概率（类别1的概率）
  species_prob <- species_pred[, "1"]
  
  # 存储该树种的预测结果
  species_predictions[[species]] <- data.frame(
    id = species_test$id,
    Species = species,
    pred = species_prob
  )
  
  # 显示该树种的预测概率分布
  cat("预测概率分布:\n")
  print(summary(species_prob))
}

# 整合所有树种的预测结果
all_predictions <- do.call(rbind, species_predictions)

# 创建最终的提交格式
final_predictions <- all_predictions %>%
  select(id, pred) %>%
  arrange(id)

# 确保预测值在[0,1]范围内
final_predictions$pred <- pmax(0, pmin(1, final_predictions$pred))

# 显示预测结果统计
cat("\n最终预测结果统计:\n")
print(summary(final_predictions$pred))

# 生成提交文件
write_csv(final_predictions, "submission.csv")
cat("\n提交文件已生成: submission.csv\n")

# ======================================================================
# 修改3: 删除每种变量进行预测，找到对哪个树种影响最大
# ======================================================================

cat("\n=== 变量删除影响分析 ===\n")

# 存储变量删除的影响结果
variable_deletion_impact <- data.frame()

# 对每个树种，删除每个变量进行预测
for(species in unique_species) {
  cat("\n分析树种", species, "的变量删除影响...\n")
  
  # 获取该树种的训练数据
  species_train <- train_data[train_data$Species == species, ]
  species_test <- test_data[test_data$Species == species, ]
  
  # 基准模型（使用所有变量）
  X_train_full <- species_train[, feature_cols]
  y_train <- species_train$pres.abs
  
  rf_full <- randomForest(x = X_train_full, y = as.factor(y_train), 
                          ntree = 500, importance = TRUE)
  
  # 获取基准预测
  X_test_full <- species_test[, feature_cols]
  pred_full <- predict(rf_full, X_test_full, type = "prob")[, "1"]
  
  # 计算基准logloss（如果有真实值的话，这里用训练集的交叉验证）
  # 使用训练集进行交叉验证来评估基准性能
  cv_results_full <- numeric(5)
  for(i in 1:5) {
    set.seed(i)
    train_indices <- sample(1:nrow(species_train), 0.8 * nrow(species_train))
    cv_train <- species_train[train_indices, ]
    cv_test <- species_train[-train_indices, ]
    
    cv_model <- randomForest(x = cv_train[, feature_cols], 
                            y = as.factor(cv_train$pres.abs), 
                            ntree = 500)
    cv_pred <- predict(cv_model, cv_test[, feature_cols], type = "prob")[, "1"]
    cv_results_full[i] <- logloss(cv_test$pres.abs, cv_pred)
  }
  baseline_logloss <- mean(cv_results_full)
  
  # 对每个变量进行删除测试
  for(var_to_remove in feature_cols) {
    cat("  删除变量:", var_to_remove, "\n")
    
    # 创建删除该变量后的特征集
    remaining_features <- feature_cols[feature_cols != var_to_remove]
    
    # 训练删除变量后的模型
    X_train_reduced <- species_train[, remaining_features]
    rf_reduced <- randomForest(x = X_train_reduced, y = as.factor(y_train), 
                              ntree = 500, importance = TRUE)
    
    # 交叉验证评估删除变量后的性能
    cv_results_reduced <- numeric(5)
    for(i in 1:5) {
      set.seed(i)
      train_indices <- sample(1:nrow(species_train), 0.8 * nrow(species_train))
      cv_train <- species_train[train_indices, ]
      cv_test <- species_train[-train_indices, ]
      
      cv_model <- randomForest(x = cv_train[, remaining_features], 
                              y = as.factor(cv_train$pres.abs), 
                              ntree = 500)
      cv_pred <- predict(cv_model, cv_test[, remaining_features], type = "prob")[, "1"]
      cv_results_reduced[i] <- logloss(cv_test$pres.abs, cv_pred)
    }
    reduced_logloss <- mean(cv_results_reduced)
    
    # 计算性能变化
    logloss_change <- reduced_logloss - baseline_logloss
    
    # 存储结果
    temp_result <- data.frame(
      Species = species,
      Removed_Variable = var_to_remove,
      Baseline_LogLoss = baseline_logloss,
      Reduced_LogLoss = reduced_logloss,
      LogLoss_Change = logloss_change,
      Performance_Impact = ifelse(logloss_change > 0, "性能下降", "性能提升")
    )
    
    variable_deletion_impact <- rbind(variable_deletion_impact, temp_result)
  }
}

# 分析变量删除的影响
cat("\n变量删除影响分析结果:\n")

# 找出对每个树种影响最大的变量
max_impact_by_species <- variable_deletion_impact %>%
  group_by(Species) %>%
  filter(LogLoss_Change == max(LogLoss_Change)) %>%
  arrange(desc(LogLoss_Change))

cat("\n对每个树种影响最大的变量:\n")
print(max_impact_by_species)

# 找出对整体影响最大的变量
overall_impact <- variable_deletion_impact %>%
  group_by(Removed_Variable) %>%
  summarise(
    Avg_LogLoss_Change = mean(LogLoss_Change),
    Max_LogLoss_Change = max(LogLoss_Change),
    Min_LogLoss_Change = min(LogLoss_Change),
    SD_LogLoss_Change = sd(LogLoss_Change),
    Affected_Species_Count = sum(LogLoss_Change > 0)
  ) %>%
  arrange(desc(Avg_LogLoss_Change))

cat("\n对整体影响最大的变量:\n")
print(overall_impact)

# 可视化变量删除影响
# 创建热图显示每个树种对每个变量删除的敏感度
impact_matrix <- variable_deletion_impact %>%
  select(Species, Removed_Variable, LogLoss_Change) %>%
  pivot_wider(names_from = Removed_Variable, values_from = LogLoss_Change) %>%
  column_to_rownames("Species") %>%
  as.matrix()

png("variable_deletion_impact_heatmap.png", width = 12, height = 8, units = "in", res = 300)
pheatmap(impact_matrix, 
         main = "变量删除对树种预测性能的影响 (LogLoss变化)",
         cluster_rows = TRUE,
         cluster_cols = TRUE,
         color = colorRampPalette(c("blue", "white", "red"))(100),
         fontsize = 10)
dev.off()

# 创建箱线图显示每个变量的影响分布
p_impact_box <- ggplot(variable_deletion_impact, 
                       aes(x = reorder(Removed_Variable, LogLoss_Change), 
                           y = LogLoss_Change, fill = Removed_Variable)) +
  geom_boxplot(alpha = 0.7) +
  geom_jitter(width = 0.2, alpha = 0.5) +
  coord_flip() +
  theme_minimal() +
  labs(title = "变量删除对预测性能的影响分布",
       x = "删除的变量", y = "LogLoss变化") +
  theme(legend.position = "none") +
  scale_fill_viridis_d()

ggsave("variable_deletion_impact_boxplot.png", p_impact_box, width = 10, height = 6, dpi = 300)

# 创建每个树种对变量删除的敏感度排序
species_sensitivity <- variable_deletion_impact %>%
  group_by(Species) %>%
  summarise(
    Avg_LogLoss_Change = mean(LogLoss_Change),
    Max_LogLoss_Change = max(LogLoss_Change),
    Min_LogLoss_Change = min(LogLoss_Change),
    SD_LogLoss_Change = sd(LogLoss_Change)
  ) %>%
  arrange(desc(Avg_LogLoss_Change))

p_species_sensitivity <- ggplot(species_sensitivity, 
                               aes(x = reorder(Species, Avg_LogLoss_Change), 
                                   y = Avg_LogLoss_Change, fill = Species)) +
  geom_col(alpha = 0.7) +
  geom_errorbar(aes(ymin = Avg_LogLoss_Change - SD_LogLoss_Change, 
                    ymax = Avg_LogLoss_Change + SD_LogLoss_Change), 
                width = 0.2, alpha = 0.7) +
  coord_flip() +
  theme_minimal() +
  labs(title = "各树种对变量删除的敏感度排序",
       x = "树种", y = "平均LogLoss变化") +
  theme(legend.position = "none") +
  scale_fill_viridis_d()

ggsave("species_sensitivity_to_variable_deletion.png", p_species_sensitivity, 
       width = 10, height = 6, dpi = 300)

# 找出最敏感和最不敏感的树种
cat("\n最敏感的树种 (对变量删除影响最大):\n")
print(species_sensitivity %>% head(3))

cat("\n最不敏感的树种 (对变量删除影响最小):\n")
print(species_sensitivity %>% tail(3))

# 找出最重要的变量（删除后影响最大的）
cat("\n最重要的变量 (删除后影响最大):\n")
print(overall_impact %>% head(3))

cat("\n最不重要的变量 (删除后影响最小):\n")
print(overall_impact %>% tail(3))

# 保存详细结果
write_csv(variable_deletion_impact, "variable_deletion_impact_results.csv")
write_csv(overall_impact, "overall_variable_impact.csv")
write_csv(species_sensitivity, "species_sensitivity_analysis.csv")

cat("\n变量删除影响分析完成!\n")
cat("生成的文件包括:\n")
cat("- variable_deletion_impact_results.csv: 详细的变量删除影响结果\n")
cat("- overall_variable_impact.csv: 整体变量影响分析\n")
cat("- species_sensitivity_analysis.csv: 树种敏感度分析\n")
cat("- variable_deletion_impact_heatmap.png: 变量删除影响热图\n")
cat("- variable_deletion_impact_boxplot.png: 变量删除影响箱线图\n")
cat("- species_sensitivity_to_variable_deletion.png: 树种敏感度排序图\n")

# ======================================================================
# 扩展分析部分（保持原有功能，但使用logloss）
# ======================================================================

# 1. 变量重要性分析
cat("\n=== 变量重要性分析 ===\n")

# 存储所有树种的重要性分析结果
all_importance <- list()

# 为每个树种计算重要性
for(species in unique_species) {
  cat("计算", species, "的变量重要性...\n")
  
  # 使用已训练的模型
  rf_model <- species_models[[species]]
  
  # 获取重要性分数
  importance_scores <- importance(rf_model)
  all_importance[[species]] <- importance_scores
}

# 创建重要性汇总表
importance_summary <- data.frame()
for(species in unique_species) {
  temp_df <- data.frame(
    Species = species,
    Feature = rownames(all_importance[[species]]),
    MeanDecreaseAccuracy = all_importance[[species]][, "MeanDecreaseAccuracy"],
    MeanDecreaseGini = all_importance[[species]][, "MeanDecreaseGini"]
  )
  importance_summary <- rbind(importance_summary, temp_df)
}

# 计算特征的平均重要性
avg_importance <- importance_summary %>%
  group_by(Feature) %>%
  summarise(
    Avg_MeanDecreaseAccuracy = mean(MeanDecreaseAccuracy),
    Avg_MeanDecreaseGini = mean(MeanDecreaseGini),
    SD_MeanDecreaseAccuracy = sd(MeanDecreaseAccuracy),
    SD_MeanDecreaseGini = sd(MeanDecreaseGini)
  ) %>%
  arrange(desc(Avg_MeanDecreaseAccuracy))

cat("\n平均变量重要性排序:\n")
print(avg_importance)

# 可视化重要性分析
# 创建重要性热图
importance_matrix <- importance_summary %>%
  select(Species, Feature, MeanDecreaseAccuracy) %>%
  pivot_wider(names_from = Feature, values_from = MeanDecreaseAccuracy) %>%
  column_to_rownames("Species") %>%
  as.matrix()

# 绘制重要性热图
png("variable_importance_heatmap.png", width = 12, height = 8, units = "in", res = 300)
pheatmap(importance_matrix, 
         main = "变量重要性热图 (Mean Decrease Accuracy)",
         scale = "column",
         cluster_rows = TRUE,
         cluster_cols = TRUE,
         color = viridis(100),
         fontsize = 10)
dev.off()

# 绘制重要性箱线图
p1 <- ggplot(importance_summary, aes(x = reorder(Feature, MeanDecreaseAccuracy), 
                                    y = MeanDecreaseAccuracy)) +
  geom_boxplot(aes(fill = Feature), alpha = 0.7) +
  coord_flip() +
  theme_minimal() +
  labs(title = "变量重要性分布 (Mean Decrease Accuracy)",
       x = "特征变量", y = "重要性分数") +
  theme(legend.position = "none")

ggsave("variable_importance_boxplot.png", p1, width = 10, height = 6, dpi = 300)

cat("\n重要性分析图表已保存!\n")

# 2. 交叉验证（使用logloss）
cat("\n=== 交叉验证 (使用LogLoss) ===\n")

# 设置交叉验证参数
cv_folds <- 5
cv_repeats <- 3

# 存储交叉验证结果
cv_results <- data.frame()

for(species in unique_species) {
  cat("进行", species, "的交叉验证...\n")
  
  species_data <- train_data[train_data$Species == species, ]
  X <- species_data[, feature_cols]
  y <- species_data$pres.abs
  
  # 存储每次交叉验证的结果
  cv_logloss <- numeric(cv_folds * cv_repeats)
  cv_accuracy <- numeric(cv_folds * cv_repeats)
  
  fold_count <- 1
  
  for(repeat_idx in 1:cv_repeats) {
    set.seed(repeat_idx)
    fold_indices <- createFolds(y, k = cv_folds)
    
    for(fold_idx in 1:cv_folds) {
      # 分割数据
      train_indices <- unlist(fold_indices[-fold_idx])
      test_indices <- fold_indices[[fold_idx]]
      
      X_train_cv <- X[train_indices, ]
      y_train_cv <- y[train_indices]
      X_test_cv <- X[test_indices, ]
      y_test_cv <- y[test_indices]
      
      # 训练模型
      rf_cv <- randomForest(x = X_train_cv, y = as.factor(y_train_cv), 
                           ntree = 500)
      
      # 预测
      pred_prob_cv <- predict(rf_cv, X_test_cv, type = "prob")[, "1"]
      pred_class_cv <- predict(rf_cv, X_test_cv)
      
      # 计算性能指标
      cv_logloss[fold_count] <- logloss(y_test_cv, pred_prob_cv)
      cv_accuracy[fold_count] <- mean(pred_class_cv == y_test_cv)
      
      fold_count <- fold_count + 1
    }
  }
  
  # 存储结果
  temp_result <- data.frame(
    Species = species,
    Mean_LogLoss = mean(cv_logloss),
    SD_LogLoss = sd(cv_logloss),
    Mean_Accuracy = mean(cv_accuracy),
    SD_Accuracy = sd(cv_accuracy)
  )
  
  cv_results <- rbind(cv_results, temp_result)
}

# 显示交叉验证结果
cat("\n交叉验证结果汇总:\n")
print(cv_results)

# 计算平均性能
avg_performance <- cv_results %>%
  summarise(
    Avg_LogLoss = mean(Mean_LogLoss, na.rm = TRUE),
    Avg_Accuracy = mean(Mean_Accuracy, na.rm = TRUE)
  )

cat("\n平均性能指标:\n")
print(avg_performance)

# 可视化交叉验证结果
# LogLoss性能比较图
p3 <- ggplot(cv_results, aes(x = reorder(Species, Mean_LogLoss), 
                             y = Mean_LogLoss, fill = Species)) +
  geom_col(alpha = 0.7) +
  geom_errorbar(aes(ymin = Mean_LogLoss - SD_LogLoss, 
                    ymax = Mean_LogLoss + SD_LogLoss), 
                width = 0.2, alpha = 0.7) +
  coord_flip() +
  theme_minimal() +
  labs(title = "各树种LogLoss性能排序",
       x = "树种", y = "LogLoss值") +
  theme(legend.position = "none") +
  scale_fill_viridis_d()

ggsave("cv_logloss_performance_ranking.png", p3, width = 10, height = 6, dpi = 300)

cat("\n交叉验证图表已保存!\n")

# 4. 模型多物种泛化性能评估
cat("\n=== 模型多物种泛化性能评估 ===\n")

# 进行跨物种预测评估
generalization_results <- data.frame()

for(i in 1:length(unique_species)) {
  for(j in 1:length(unique_species)) {
    if(i != j) {
      train_species <- unique_species[i]
      test_species <- unique_species[j]
      
      cat("训练物种:", train_species, "-> 测试物种:", test_species, "\n")
      
      # 准备训练数据
      train_species_data <- train_data[train_data$Species == train_species, ]
      X_train <- train_species_data[, feature_cols]
      y_train <- as.factor(train_species_data$pres.abs)
      
      # 准备测试数据
      test_species_data <- train_data[train_data$Species == test_species, ]
      X_test <- test_species_data[, feature_cols]
      y_test <- as.factor(test_species_data$pres.abs)
      
      # 训练模型
      rf_model <- randomForest(x = X_train, y = y_train, ntree = 500)
      
      # 预测
      pred_prob <- predict(rf_model, X_test, type = "prob")
      pred_class <- predict(rf_model, X_test)
      
      # 计算性能指标
      cm <- confusionMatrix(pred_class, y_test)
      
      # 计算AUC
      if(ncol(pred_prob) == 2) {
        roc_obj <- roc(y_test, pred_prob[, "1"])
        auc_value <- auc(roc_obj)
      } else {
        auc_value <- NA
      }
      
      # 存储结果
      temp_result <- data.frame(
        Train_Species = train_species,
        Test_Species = test_species,
        Accuracy = cm$overall["Accuracy"],
        Sensitivity = cm$byClass["Sensitivity"],
        Specificity = cm$byClass["Specificity"],
        AUC = as.numeric(auc_value)
      )
      
      generalization_results <- rbind(generalization_results, temp_result)
    }
  }
}

# 计算泛化性能统计
generalization_summary <- generalization_results %>%
  group_by(Train_Species) %>%
  summarise(
    Avg_Accuracy = mean(Accuracy, na.rm = TRUE),
    Avg_Sensitivity = mean(Sensitivity, na.rm = TRUE),
    Avg_Specificity = mean(Specificity, na.rm = TRUE),
    Avg_AUC = mean(AUC, na.rm = TRUE),
    SD_Accuracy = sd(Accuracy, na.rm = TRUE),
    SD_AUC = sd(AUC, na.rm = TRUE)
  ) %>%
  arrange(desc(Avg_AUC))

cat("\n泛化性能汇总（按训练物种）:\n")
print(generalization_summary)

# 比较物种内vs物种间性能
within_species_auc <- cv_results$Mean_LogLoss # 使用LogLoss作为性能指标
between_species_auc <- generalization_results$AUC[!is.na(generalization_results$AUC)]

performance_comparison <- data.frame(
  Type = c(rep("物种内", length(within_species_auc)), 
           rep("物种间", length(between_species_auc))),
  AUC = c(within_species_auc, between_species_auc)
)

cat("\n物种内 vs 物种间性能比较:\n")
cat("物种内平均LogLoss:", mean(within_species_auc, na.rm = TRUE), "\n")
cat("物种间平均AUC:", mean(between_species_auc, na.rm = TRUE), "\n")
cat("性能差异:", mean(within_species_auc, na.rm = TRUE) - mean(between_species_auc, na.rm = TRUE), "\n")

# 可视化泛化性能
# 泛化性能热图
generalization_matrix <- generalization_results %>%
  select(Train_Species, Test_Species, AUC) %>%
  pivot_wider(names_from = Test_Species, values_from = AUC) %>%
  column_to_rownames("Train_Species") %>%
  as.matrix()

png("generalization_heatmap.png", width = 10, height = 8, units = "in", res = 300)
pheatmap(generalization_matrix, 
         main = "跨物种泛化性能热图 (AUC)",
         cluster_rows = TRUE,
         cluster_cols = TRUE,
         color = viridis(100),
         na_col = "white",
         fontsize = 10)
dev.off()

# 物种内vs物种间性能比较箱线图
p5 <- ggplot(performance_comparison, aes(x = Type, y = AUC, fill = Type)) +
  geom_boxplot(alpha = 0.7) +
  geom_jitter(width = 0.2, alpha = 0.5) +
  theme_minimal() +
  labs(title = "物种内 vs 物种间模型性能比较",
       x = "预测类型", y = "AUC值") +
  scale_fill_viridis_d() +
  theme(legend.position = "none")

ggsave("within_vs_between_species_performance.png", p5, width = 8, height = 6, dpi = 300)

# 各物种泛化能力排序
p6 <- ggplot(generalization_summary, aes(x = reorder(Train_Species, Avg_AUC), 
                                         y = Avg_AUC, fill = Train_Species)) +
  geom_col(alpha = 0.7) +
  geom_errorbar(aes(ymin = Avg_AUC - SD_AUC, ymax = Avg_AUC + SD_AUC), 
                width = 0.2, alpha = 0.7) +
  coord_flip() +
  theme_minimal() +
  labs(title = "各物种模型的泛化能力排序",
       x = "训练物种", y = "平均AUC值") +
  theme(legend.position = "none") +
  scale_fill_viridis_d()

ggsave("species_generalization_ranking.png", p6, width = 10, height = 6, dpi = 300)

cat("\n泛化性能评估图表已保存!\n")

# 5. 分布地图可视化
cat("\n=== 分布地图可视化 ===\n")

# 准备地图数据
# 合并训练和测试数据的地理信息
train_geo <- train_data %>%
  select(long, lat, Species, pres.abs) %>%
  mutate(data_type = "训练数据")

test_geo <- test_data %>%
  select(long, lat, Species) %>%
  mutate(pres.abs = NA, data_type = "测试数据")

all_geo_data <- rbind(train_geo, test_geo)

# 获取地图边界
lon_range <- range(all_geo_data$long, na.rm = TRUE)
lat_range <- range(all_geo_data$lat, na.rm = TRUE)

# 扩展边界
lon_buffer <- diff(lon_range) * 0.1
lat_buffer <- diff(lat_range) * 0.1

# 创建各树种的分布地图
species_maps <- list()

for(species in unique_species) {
  species_data <- all_geo_data[all_geo_data$Species == species, ]
  
  # 创建基础地图
  p_map <- ggplot(species_data, aes(x = long, y = lat)) +
    # 添加地理边界（如果需要）
    geom_point(aes(color = factor(pres.abs), shape = data_type), 
               size = 2, alpha = 0.7) +
    scale_color_manual(values = c("0" = "red", "1" = "green", "NA" = "blue"),
                       labels = c("0" = "不存在", "1" = "存在", "NA" = "待预测"),
                       name = "物种状态") +
    scale_shape_manual(values = c("训练数据" = 16, "测试数据" = 17),
                       name = "数据类型") +
    theme_minimal() +
    labs(title = paste("树种", species, "的地理分布"),
         x = "经度", y = "纬度") +
    coord_fixed(ratio = 1) +
    theme(legend.position = "bottom")
  
  species_maps[[species]] <- p_map
  
  # 保存单个物种分布图
  ggsave(paste0("species_distribution_", species, ".png"), p_map, 
         width = 10, height = 8, dpi = 300)
}

# 创建所有物种的综合分布图
all_species_plot <- ggplot(train_geo, aes(x = long, y = lat)) +
  geom_point(aes(color = Species, shape = factor(pres.abs)), 
             size = 1.5, alpha = 0.6) +
  scale_shape_manual(values = c("0" = 1, "1" = 16),
                     labels = c("0" = "不存在", "1" = "存在"),
                     name = "物种状态") +
  scale_color_viridis_d(name = "树种") +
  theme_minimal() +
  labs(title = "所有树种的综合地理分布",
       x = "经度", y = "纬度") +
  coord_fixed(ratio = 1) +
  theme(legend.position = "bottom")

ggsave("all_species_distribution.png", all_species_plot, 
       width = 12, height = 10, dpi = 300)

# 创建密度分布图
density_plot <- ggplot(train_geo, aes(x = long, y = lat)) +
  geom_density_2d_filled(alpha = 0.7) +
  geom_point(data = train_geo[train_geo$pres.abs == 1, ], 
             aes(color = Species), size = 1, alpha = 0.8) +
  scale_color_viridis_d(name = "树种") +
  theme_minimal() +
  labs(title = "树种分布密度图",
       x = "经度", y = "纬度") +
  coord_fixed(ratio = 1) +
  theme(legend.position = "bottom")

ggsave("species_density_distribution.png", density_plot, 
       width = 12, height = 10, dpi = 300)

# 创建预测概率分布图（使用之前生成的预测结果）
# 合并预测结果和地理信息
pred_geo <- merge(final_predictions, test_data[, c("id", "long", "lat", "Species")], 
                  by = "id", all.x = TRUE)

# 创建预测概率热图
prediction_heatmap <- ggplot(pred_geo, aes(x = long, y = lat, fill = pred)) +
  geom_point(shape = 21, size = 2, alpha = 0.8) +
  scale_fill_viridis_c(name = "预测概率", 
                       option = "plasma") +
  facet_wrap(~Species, ncol = 3) +
  theme_minimal() +
  labs(title = "各树种预测概率分布图",
       x = "经度", y = "纬度") +
  coord_fixed(ratio = 1) +
  theme(legend.position = "bottom",
        strip.text = element_text(size = 10))

ggsave("prediction_probability_heatmap.png", prediction_heatmap, 
       width = 15, height = 12, dpi = 300)

# 环境梯度可视化
# 选择几个重要的环境变量进行可视化
env_vars <- c("rainann", "tempann", "soildepth", "topo")
env_plots <- list()

for(var in env_vars) {
  p_env <- ggplot(train_geo, aes_string(x = "long", y = "lat", color = var)) +
    geom_point(size = 1.5, alpha = 0.7) +
    scale_color_viridis_c(name = var) +
    theme_minimal() +
    labs(title = paste("环境变量", var, "的空间分布"),
         x = "经度", y = "纬度") +
    coord_fixed(ratio = 1)
  
  env_plots[[var]] <- p_env
  ggsave(paste0("environmental_", var, "_distribution.png"), p_env, 
         width = 10, height = 8, dpi = 300)
}

# 创建环境变量组合图
png("environmental_variables_combined.png", width = 15, height = 12, units = "in", res = 300)
grid.arrange(grobs = env_plots, ncol = 2, nrow = 2)
dev.off()

cat("\n分布地图可视化已完成!\n")
cat("生成的地图文件包括:\n")
cat("- 各树种单独分布图\n")
cat("- 所有树种综合分布图\n")
cat("- 密度分布图\n")
cat("- 预测概率热图\n")
cat("- 环境变量分布图\n")

cat("\n脚本执行完成！\n")
cat("主要修改包括:\n")
cat("1. 使用LogLoss作为评估指标\n")
cat("2. 每个树种单独预测再整合\n")
cat("3. 删除每种变量进行预测，分析对树种的影响\n")
cat("4. 生成详细的变量删除影响分析报告\n") 