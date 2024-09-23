1.把想测试的项目文件夹从experient_data文件夹里复制到experient_data文件夹外
2.把尾注有（1）（2）（3）脚本里面的全局变量Project_name换成当前要测的项目名字
3.执行（1）脚本，会自动分割正负样本集
4.执行（2）脚本，会自动分割训练集和测试集（9：1且保证正负样本比例平衡）
5.执行（3）脚本，从训练集里提取xtrain ytrain 训练随机森林分类器。并从测试集里提取ytest, 和分类器用xtest预测出的xPredict比较，测f1_score,precision，recall三个指标

SITAR的所有用到的数据集皆已上传至experient_data文件夹中
