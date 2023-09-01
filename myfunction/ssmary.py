from datetime import datetime

def ssmary(path,mymodel,e,correct,test_loss):
    with open(path, "w") as f:
        f.write("训练总结\n")
        f.write("模型信息:\n")
        f.write(str(mymodel))  # 将模型信息写入文件
        f.write("\n")
        f.write(f"总训练轮数: {e}\n")
        f.write(f"最终测试准确率: {correct*100:.2f}%\n")
        f.write(f"最终测试损失值: {test_loss:.4f}\n")
        f.write("训练完成！")
    
    # 添加时间戳
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"生成时间: {timestamp}\n")

    print("训练总结已保存到 best_training_summary.txt 文件中")  
