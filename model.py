import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import KFold
from lightgbm import LGBMRegressor as LGBMR, early_stopping
from xgboost import XGBRegressor as XGBR
from catboost import CatBoostRegressor as CBR
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, ParameterSampler
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import Ridge, Lasso, ElasticNet
import pickle
import numpy as np

class cfg:
    trainfilepath = "train_processed.csv"
    testfilepath = "test_processed.csv"
    outfilepath = "myoutput"
    state = 8
    n_iter = 10  # 每个模型的参数搜索次数
    early_stop_rounds = 50

class future_engineer:
    traindata = pd.read_csv(cfg.trainfilepath)
    testdata = pd.read_csv(cfg.testfilepath)
    headerstrian = set(traindata.columns)
    headerstest = set(testdata.columns)
    target = headerstrian.symmetric_difference(headerstest).pop()
    print("获取的目标列是：", target)

class dataset_split:
    # 划分训练集和验证集（保持原始测试集不变）
    X_full = future_engineer.traindata.drop([future_engineer.target], axis=1)
    Y_full = future_engineer.traindata[future_engineer.target]
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    X_test = future_engineer.testdata
    print("数据分隔完成（含验证集）")

class HyperparameterSearch:
    @staticmethod
    def lgbm_search():
        param_dist = {
            'learning_rate': [0.012, 0.016,0.014,0.018,0.01, 0.02],
            'max_depth': [4,5,6,7,8],
            'colsample_bytree': [0.8, 0.9, 0.95],
            'reg_alpha': [0.001, 0.01, 0.1],
            'reg_lambda': [0.001, 0.01, 0.1]
        }
        best_score = np.inf
        best_model = None
        
        for params in ParameterSampler(param_dist, n_iter=cfg.n_iter, random_state=cfg.state):
            score_list = []
            model = LGBMR(
                objective="regression",
                n_estimators=1000,
                **params,
                random_state=cfg.state,
                verbosity=-1
            )
            for train_index, val_index in dataset_split.kf.split(dataset_split.X_full):
                # 修改前（错误）
                
                X_train = dataset_split.X_full.iloc[train_index]
                X_val = dataset_split.X_full.iloc[val_index]
                y_train = dataset_split.Y_full.iloc[train_index]
                y_val = dataset_split.Y_full.iloc[val_index]
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    callbacks=[early_stopping(cfg.early_stop_rounds)],)
                pred = model.predict(X_val)
                score_list.append(mean_absolute_error(y_val, pred))
            score = np.mean(score_list)
            score_list = []  # 清空列表以便下次使用
            if score < best_score:
                best_score = score
                best_model = model
        print(f"LGBM 最佳验证MAE: {best_score:.4f}")
        print(best_model)
        return best_model

    @staticmethod
    def xgb_search():
        param_dist = {
            'learning_rate': [0.012, 0.016,0.014,0.018,0.01, 0.02],
            'max_depth': [4, 5,6, 7,8],
            'colsample_bytree': [0.8, 0.9, 0.95],
            'reg_alpha': [0.001, 0.01, 0.1],
            'gamma': [0, 0.1, 0.2]
        }
        best_score = np.inf
        best_model = None

        for params in ParameterSampler(param_dist, n_iter=cfg.n_iter, random_state=cfg.state):
            score_list = []
            model = XGBR(
                objective="reg:squarederror",
                n_estimators=1000,
                
                **params,
                random_state=cfg.state,
                n_jobs=4,
                tree_method="gpu_hist",
                predictor="gpu_predictor",
                gpu_id=0,
                eval_metric="mae",
            )
            for train_index, val_index in dataset_split.kf.split(dataset_split.X_full):
                # 修改前（错误）
                
                X_train = dataset_split.X_full.iloc[train_index]
                X_val = dataset_split.X_full.iloc[val_index]
                y_train = dataset_split.Y_full.iloc[train_index]
                y_val = dataset_split.Y_full.iloc[val_index]
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False,)
                pred = model.predict(X_val)
                score_list.append(mean_absolute_error(y_val, pred))
            score = np.mean(score_list)
            score_list = []  # 清空列表以便下次使用
            if score < best_score:
                best_score = score
                best_model = model
        print(f"XGB 最佳验证MAE: {best_score:.4f}")
        print(best_model)
        return best_model

    @staticmethod
    def cat_search():
        param_dist = {
                'learning_rate': [0.012, 0.016,0.014,0.018,0.01, 0.02],
                'depth': [4,5, 6,7, 8],  # 替换max_depth → depth[3,5,7](@ref)
                
                'l2_leaf_reg': [0.1, 0.5, 1.0, 10]  # 扩展范围[3](@ref)
}
        best_score = np.inf
        best_model = None

        for params in ParameterSampler(param_dist, n_iter=cfg.n_iter, random_state=cfg.state):
            score_list = []
            model = CBR(
                loss_function="RMSE",
                iterations=1000,
                **params,
                random_state=cfg.state,
                verbose=0,
                early_stopping_rounds=cfg.early_stop_rounds,
                task_type="GPU",
                devices='0'
            )
            for train_index, val_index in dataset_split.kf.split(dataset_split.X_full):
                # 修改前（错误）
                
                X_train = dataset_split.X_full.iloc[train_index]
                X_val = dataset_split.X_full.iloc[val_index]
                y_train = dataset_split.Y_full.iloc[train_index]
                y_val = dataset_split.Y_full.iloc[val_index]
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False,)
                pred = model.predict(X_val)
                score_list.append(mean_absolute_error(y_val, pred))
            score = np.mean(score_list)
            score_list = []  # 清空列表以便下次使用
            if score < best_score:
                best_score = score
                best_model = model
        print(f"CatBoost 最佳验证MAE: {best_score:.4f}")
        print(best_model)
        return best_model


class model:
    models = {
        "LGBM": HyperparameterSearch.lgbm_search(),
        "XGB": HyperparameterSearch.xgb_search(),
        "CatBoost": HyperparameterSearch.cat_search()
    }
    print("\n所有基础模型训练完成")

class StackingModel:
    @staticmethod
    def generate_first_level_features():
        """生成第一层模型的特征（基于交叉验证）"""
        print("\n开始生成第一层Stacking特征...")
        X = dataset_split.X_full
        y = dataset_split.Y_full
        
        # 为训练集创建空的特征矩阵
        first_level_train = np.zeros((X.shape[0], len(model.models)))
        
        # 使用交叉验证生成训练集特征
        for i, (name, m) in enumerate(model.models.items()):
            print(f"生成 {name} 的交叉验证预测...")
            for train_idx, val_idx in dataset_split.kf.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train = y.iloc[train_idx]
                
                # 克隆模型并在训练集上训练
                if name == "LGBM":
                    clone_model = LGBMR(**m.get_params())
                elif name == "XGB":
                    clone_model = XGBR(**m.get_params())
                elif name == "CatBoost":
                    clone_model = CBR(**m.get_params())
                
                clone_model.fit(X_train, y_train)
                first_level_train[val_idx, i] = clone_model.predict(X_val)
        
        # 为测试集创建特征矩阵
        first_level_test = np.zeros((dataset_split.X_test.shape[0], len(model.models)))
        for i, (name, m) in enumerate(model.models.items()):
            print(f"使用 {name} 生成测试集预测...")
            first_level_test[:, i] = m.predict(dataset_split.X_test)
        
        print("第一层Stacking特征生成完成")
        return first_level_train, first_level_test
    
    @staticmethod
    def train_second_level_models(first_level_train, y):
        """训练第二层模型"""
        print("\n开始训练第二层模型...")
        second_level_models = {
            "Ridge": Ridge(alpha=1.0, random_state=cfg.state),
            "Lasso": Lasso(alpha=0.01, random_state=cfg.state),
            "ElasticNet": ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=cfg.state),
            "LGBM_meta": LGBMR(n_estimators=200, learning_rate=0.01, random_state=cfg.state)
        }
        
        # 为训练集创建空的第二层特征矩阵
        second_level_train = np.zeros((first_level_train.shape[0], len(second_level_models)))
        
        # 使用交叉验证生成第二层训练集特征
        for i, (name, m) in enumerate(second_level_models.items()):
            print(f"训练第二层模型 {name}...")
            for train_idx, val_idx in dataset_split.kf.split(first_level_train):
                X_train, X_val = first_level_train[train_idx], first_level_train[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                # 克隆模型并训练
                if name == "LGBM_meta":
                    clone_model = LGBMR(**m.get_params())
                elif name == "Ridge":
                    clone_model = Ridge(**m.get_params())
                elif name == "Lasso":
                    clone_model = Lasso(**m.get_params())
                elif name == "ElasticNet":
                    clone_model = ElasticNet(**m.get_params())
                
                clone_model.fit(X_train, y_train)
                second_level_train[val_idx, i] = clone_model.predict(X_val)
            
            # 在全部数据上重新训练模型
            m.fit(first_level_train, y)
        
        print("第二层模型训练完成")
        return second_level_models, second_level_train
    
    @staticmethod
    def generate_second_level_test_features(first_level_test, second_level_models):
        """为测试集生成第二层特征"""
        print("\n为测试集生成第二层特征...")
        second_level_test = np.zeros((first_level_test.shape[0], len(second_level_models)))
        
        for i, (name, m) in enumerate(second_level_models.items()):
            print(f"使用第二层模型 {name} 生成测试集预测...")
            second_level_test[:, i] = m.predict(first_level_test)
        
        print("第二层测试集特征生成完成")
        return second_level_test
    
    @staticmethod
    def train_final_model(second_level_train, y):
        """训练最终模型"""
        print("\n训练最终模型...")
        final_model = Ridge(alpha=0.5, random_state=cfg.state)
        
        # 评估最终模型性能
        final_scores = []
        for train_idx, val_idx in dataset_split.kf.split(second_level_train):
            X_train, X_val = second_level_train[train_idx], second_level_train[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            final_model.fit(X_train, y_train)
            pred = final_model.predict(X_val)
            score = mean_absolute_error(y_val, pred)
            final_scores.append(score)
        
        print(f"最终模型交叉验证MAE: {np.mean(final_scores):.4f} ± {np.std(final_scores):.4f}")
        
        # 在全部数据上重新训练最终模型
        final_model.fit(second_level_train, y)
        print("最终模型训练完成")
        return final_model
    
    @staticmethod
    def run_stacking():
        """执行完整的两层Stacking流程"""
        # 生成第一层特征
        first_level_train, first_level_test = StackingModel.generate_first_level_features()
        
        # 训练第二层模型
        second_level_models, second_level_train = StackingModel.train_second_level_models(
            first_level_train, dataset_split.Y_full)
        
        # 为测试集生成第二层特征
        second_level_test = StackingModel.generate_second_level_test_features(
            first_level_test, second_level_models)
        
        # 训练最终模型
        final_model = StackingModel.train_final_model(second_level_train, dataset_split.Y_full)
        
        # 生成最终预测
        final_prediction = final_model.predict(second_level_test)
        
        # 保存模型和预测结果
        stacking_models = {
            'base_models': model.models,
            'second_level_models': second_level_models,
            'final_model': final_model
        }
        
        # 创建输出目录（如果不存在）
        if not os.path.exists(cfg.outfilepath):
            os.makedirs(cfg.outfilepath)
        
        # 保存模型
        with open(os.path.join(cfg.outfilepath, 'stacking_models.pkl'), 'wb') as f:
            pickle.dump(stacking_models, f)
        
        # 保存预测结果
        test_ids = future_engineer.testdata["id"]
        pd.DataFrame({
            "id": test_ids,
            future_engineer.target: final_prediction
        }).to_csv(os.path.join(cfg.outfilepath, "stacking_submission.csv"), index=False)
        
        print("\n两层Stacking模型训练完成，预测结果已保存")
        return final_prediction

class predict:
    # 运行单个模型预测
    predictions = {}
    for name, model in model.models.items():
        predictions[name] = model.predict(dataset_split.X_test)
        print(f"{name} 预测完成")
    
    # 运行Stacking模型
    stacking_prediction = StackingModel.run_stacking()
    predictions["Stacking"] = stacking_prediction
    print("Stacking预测完成")

class writefile:
    test_ids = future_engineer.testdata["id"]
    for name, pred in predict.predictions.items():
        filename = os.path.join(cfg.outfilepath, f"{name}_submission.csv")
        pd.DataFrame({
            "id": test_ids,
            future_engineer.target: pred
        }).to_csv(filename, index=False)
        print(f"结果已保存至：{filename}")

# 比较各模型性能（可选）
def compare_models():
    print("\n各模型预测结果比较:")
    for name, pred in predict.predictions.items():
        if name != "Stacking":  # 基础模型
            base_mae = mean_absolute_error(predict.predictions["Stacking"], pred)
            print(f"{name} vs Stacking MAE差异: {base_mae:.4f}")
    
    # 计算各模型预测的相关性
    pred_df = pd.DataFrame(predict.predictions)
    correlation = pred_df.corr()
    print("\n模型预测相关性矩阵:")
    print(correlation)

# 执行模型比较
compare_models()