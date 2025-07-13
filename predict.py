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
features = ["long", "lat", "rainann", "soildepth", "soilfert",
            "tempann", "topo", "easting", "northing"]

# 3. 准备保存结果
all_predictions = []
logscore_list = []

# 4. 获取所有物种
species_list = train["Species"].unique()

# 5. 对每个物种建模
for i, species in enumerate(species_list, start=1):
    print(f"\n📦 训练物种 {i}: {species}")

    # 提取该物种数据
    train_s = train[train["Species"] == species].copy()
    test_s = test[test["Species"] == species].copy()
    X = train_s[features].values
    y = train_s["pres.abs"].values

    # pres.abs 全为 0 或 1 的物种跳过
    if len(np.unique(y)) < 2:
        print(f"⚠️ pres.abs 全为 {np.unique(y)[0]}，跳过")
        continue

    # 特征缩放
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    X_test = scaler.transform(test_s[features].values)

    # 维度调整
    X_scaled = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    # 划分训练/验证集，确保包含 0 和 1
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    # 构建 CNN 模型
    model = Sequential([
        Input(shape=(X_train.shape[1], 1)),
        Conv1D(32, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

    # 训练模型
    model.fit(X_train, y_train, epochs=30, batch_size=128, verbose=0)

    # 验证预测
    val_pred = model.predict(X_val).flatten()
    val_pred = np.clip(val_pred, 1e-15, 1 - 1e-15)
    logscore = log_loss(y_val, val_pred)
    print(f"✅ log loss: {logscore:.5f}")
    logscore_list.append((species, logscore))

    # 测试预测
    test_pred = model.predict(X_test).flatten()
    result_df = pd.DataFrame({
        "id": test_s["id"],
        "pred": test_pred
    })
    all_predictions.append(result_df)

# 6. 合并预测结果
submission = pd.concat(all_predictions).sort_values("id")
submission.to_csv("submission.csv", index=False)
print("submission.csv")


