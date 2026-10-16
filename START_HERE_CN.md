# 上传与运行说明

建议仓库名称：`hlft-traffic-imputation`。

Description 可填：

```text
Traffic data imputation with hyperbolic tensor factorization and Riemannian optimization across four real-world datasets.
```

## 上传 GitHub

1. 解压 `HLFT_GitHub_Package.zip`。
2. 打开解压后的 `hlft-traffic-imputation` 文件夹。
3. 新建仓库，填写上面的名称与 description。用于展示时选 Public。
4. 不需要让 GitHub 自动创建 README 或 .gitignore；文件包中已经有。
5. 仓库创建后，点击 uploading an existing file。
6. 把项目文件夹里面的文件和子文件夹一起拖进去，保留结构，不要把整个 ZIP 上传，也不要在仓库外面再套一层项目目录。
7. 提交说明填 `Add HLFT traffic imputation project`，点击 Commit changes。
8. 检查首页封面、结果图片、论文链接与 notebook 是否正常。

数据文件已压成 .txt.gz，代码可直接读取；最大单文件约 15.5 MB。不要先解压这些 .gz 再上传，Seattle 的原始训练文件更大。

## 本地运行

使用 Python 3.11 或 3.12。在项目文件夹打开终端，执行：

```powershell
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python hlft.py --dataset Guangzhou --max-epochs 100
```

要运行全部四个城市：

```powershell
python hlft.py --dataset all --max-epochs 100
```

也可以打开根目录的 `HLFT_Traffic_Imputation.ipynb`，按顺序运行。数据目录默认是旁边的 data 文件夹。

结果保存在 outputs/ 下，包括最终指标、验证集曲线、模型参数、预测样本和运行记录。

## README 的结果口径

README 将论文中的历史结果与本次整理版本的实测结果分开说明。新运行沿用原脚本的 log10(speed+1) 处理，并在原数据速度单位上评估；论文示例是归一化误差，不能直接用两个数相比较。

原脚本通过测试集选轮次，这里已经改为通过验证集选最佳轮次。原始脚本保留在 reference/ 中，供你对照。页尾使用 Maintainer 和你的姓名。
