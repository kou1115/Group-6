import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Input
from tensorflow.keras.optimizers import Adam

# 1. 读取数据
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# 2. 特征列
features = ["disturb", "rainann", "soildepth", "soilfert",
            "tempann", "topo", "easting", "northing"]

all_predictions = []
logscore_list = []

# 3. 获取物种列表
species_list = train["Species"].unique()

for species in species_list:
    print(f"训练物种: {species}")

    # 提取该物种数据
    train_s = train[train["Species"] == species].copy()
    test_s = test[test["Species"] == species].copy()

    X = train_s[features].values
    y = train_s["pres.abs"].values

    if len(np.unique(y)) < 2:
        print("跳过：标签不平衡")
        continue

    # === 样本按northing排序，保证空间邻近性 ===
    sort_idx = np.argsort(train_s["northing"].values)
    train_s = train_s.iloc[sort_idx]
    y = y[sort_idx]

    # === 特征按相关性排序 ===
    corrs = []
    for f in features:
        corr = abs(np.corrcoef(train_s[f].values, y)[0, 1])
        if np.isnan(corr):
            corr = 0
        corrs.append((f, corr))
    sorted_features = [f for f, _ in sorted(corrs, key=lambda x: x[1], reverse=True)]

    X = train_s[sorted_features].values
    X_test = test_s[sorted_features].values

    # 数据归一化
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    X_test = scaler.transform(X_test)

    # 转换为CNN输入形状
    X_scaled = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    # 划分训练集与验证集
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    # CNN模型
    model = Sequential([
        Input(shape=(X_train.shape[1], 1)),
        Conv1D(128, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=Adam(), loss='binary_crossentropy')
    model.fit(X_train, y_train, epochs=100, batch_size=64, verbose=0)

    # 验证集预测并计算log loss
    val_pred = model.predict(X_val).flatten()
    val_pred = np.clip(val_pred, 1e-15, 1 - 1e-15)
    logscore = log_loss(y_val, val_pred)
    logscore_list.append((species, logscore))
    print(f"log loss: {logscore:.5f}")

    # 测试集预测
    test_pred = model.predict(X_test).flatten()
    result_df = pd.DataFrame({
        "id": test_s["id"],
        "pred": test_pred
    })
    all_predictions.append(result_df)

# 保存提交文件
submission = pd.concat(all_predictions).sort_values("id")
submission.to_csv("submission_CNN_sorted.csv", index=False)
print("submission_CNN_sorted.csv")
