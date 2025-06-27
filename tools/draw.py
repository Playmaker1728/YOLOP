import os
import sys
import argparse
import re
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
os.chdir(BASE_DIR)

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--YOLOP', nargs='+', type=str,
                        default=[
                            './runs/BddDataset/_2025-03-04-03-43/_2025-03-04-03-43_train.log',
                            './runs/BddDataset/_2025-03-04-20-10/_2025-03-04-20-10_train.log',
                            './runs/BddDataset/_2025-03-04-22-14/_2025-03-04-22-14_train.log',
                            './runs/BddDataset/_2025-03-04-23-23/_2025-03-04-23-23_train.log',
                            './runs/BddDataset/_2025-03-05-07-46/_2025-03-05-07-46_train.log',
                            './runs/BddDataset/_2025-03-06-05-51/_2025-03-06-05-51_train.log',
                            './runs/BddDataset/_2025-03-10-19-35/_2025-03-10-19-35_train.log',
                            './runs/BddDataset/_2025-04-02-15-32/_2025-04-02-15-32_train.log',
                            './runs/BddDataset/_2025-04-09-02-37/_2025-04-09-02-37_train.log',
                            './runs/BddDataset/_2025-04-10-03-14/_2025-04-10-03-14_train.log'
                        ],
                        help='YOLOP_history_log')
    parser.add_argument('--save-path', type=str, default='./analysis', help='保存路径')
    args = parser.parse_args()
    return args

def get_data(mode: str, args):
    """从文件中读取数据,生成csv文件
    Args:
        mode: 读取的模型模式
        args: 参数
    Return:
        data: np.ndarray
        csv_file: str
    """
    
    if mode == 'YOLOP':
        log = [args.YOLOP] if isinstance(args.YOLOP, str) else args.YOLOP
    else:
        raise NameError('请选择正确的参数')
    
    da, ll, det = [], [], []
    for file in log:
        with open(file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            pattern = re.compile(r'\((\d+\.?\d*)\)')
            for line in lines:
                if line.startswith("Driving area Segment:"):
                    matchs = [np.float16(x) for x in re.findall(pattern, line)]
                    da.append(matchs)
                elif line.startswith("Lane line Segment:"):
                    matchs = [np.float16(x) for x in re.findall(pattern, line)]
                    ll.append(matchs)
                elif line.startswith("Detect:"):
                    matchs = [np.float16(x) for x in re.findall(pattern, line)]
                    det.append(matchs)
    assert len(da) == len(ll) == len(det), '数据维度不匹配！'
    
    csv_file = Path(args.save_path) / f"{mode}_log.csv"
    if not csv_file.exists():
        csv_file.parent.mkdir(parents=True, exist_ok=True)   
    # if not os.path.exists(args.save_path):
    #     os.makedirs(args.save_path)
    # csv_file = os.path.join(args.save_path, mode+'_log.csv')
    
    fieldnames = ['da_acc', 'da_iou', 'da_miou', 
                  'll_acc', 'll_iou', 'll_miou',
                  'det_p', 'det_r', 'det_map50', 'det_map95']
    data = np.c_[np.array(da), np.array(ll), np.array(det)]
    
    with csv_file.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(fieldnames)
        writer.writerows(data)
    print(f'数据写入至:{csv_file.resolve()}')
        
    return data, csv_file

def analysis_data(data: np.ndarray, file_path: Path):
    """分析处理数据,生成数据图"""
    
    epoch = np.arange(0, data.shape[0])
    
    #设置中文字体 使用python3.12.0版本,matplotlib版本兼容问题
    plt.rcParams["font.family"] = ["Times New Roman", "SimSun"]  # 英文字体为新罗马，中文字体为宋体
    plt.rcParams["font.serif"] = ["Times New Roman", "SimSun"]  # 衬线字体
    plt.rcParams["font.sans-serif"] = ["Times New Roman", "SimSun", "Arial", "SimHei"]  # 无衬线字体，与Latex相关
    plt.rcParams["mathtext.fontset"] = "custom" # 设置LaTeX字体为用户自定义，这里演示，就不用computer modern了
    #设置负号显示
    plt.rcParams['axes.unicode_minus'] = False
    
    plt.figure()
    plt.title('可行驶区域分割综合性能指标')
    plt.plot(epoch, data[:, 0], label='da Acc', color='blue', linewidth=2)
    plt.plot(epoch, data[:, 1], label='da IoU', color='green', linewidth=2)
    plt.plot(epoch, data[:, 2], label='da mIoU', color='orange', linewidth=2)
    plt.gca().yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    plt.xlabel('epochs')
    plt.ylabel('综合指标')
    plt.grid(True)
    plt.legend(loc='lower right')
    pic = file_path.parent / '01可行驶区域综合指标.png'
    plt.savefig(pic)
    print(f'save pic path is {pic}')
    plt.show()
    
    plt.figure()
    plt.title('车道线识别综合性能指标')
    plt.plot(epoch, data[:, 3], label='ll Acc', color='blue', linewidth=2)
    plt.plot(epoch, data[:, 4], label='ll IoU', color='green', linewidth=2)
    plt.plot(epoch, data[:, 5], label='ll mIoU', color='orange', linewidth=2)
    plt.gca().yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    plt.xlabel('epochs')
    plt.ylabel('综合指标')
    plt.grid(True)
    plt.legend(loc='lower right')
    pic = file_path.parent / '02车道线识别综合指标.png'
    plt.savefig(pic)
    print(f'save pic path is {pic}')
    plt.show()
    
    plt.figure()
    plt.title('目标检测综合性能指标')
    plt.plot(epoch, data[:, 6], label='det Precision', color='red', linewidth=2)
    plt.plot(epoch, data[:, 7], label='det Recall', color='brown', linewidth=2)
    plt.plot(epoch, data[:, 8], label='det mAP(50%)', color='purple', linewidth=2)
    plt.plot(epoch, data[:, 9], label='det mAP(50-95%)', color='lime', linewidth=2)
    plt.gca().yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    plt.xlabel('epochs')
    plt.ylabel('综合指标')
    plt.grid(True)
    plt.legend(loc='lower right')
    pic = file_path.parent / '03目标检测综合指标.png'
    plt.savefig(pic)
    print(f'save pic path is {pic}')
    plt.show()

def main():
    args = get_parser()

    YOLOP_data, YOLOP_csv = get_data('YOLOP', args)
    analysis_data(YOLOP_data, YOLOP_csv)
    
    print(f'len: {YOLOP_data.shape[0]}')
    print(f'da_acc_max is {YOLOP_data[50:, 0].max()*100:.3f}%')
    print(f'da_iou_max is {YOLOP_data[50:, 1].max()*100:.3f}%')
    print(f'da_miou_max is {YOLOP_data[50:, 2].max()*100:.3f}%')
    print(f'll_acc_max is {YOLOP_data[50:, 3].max()*100:.3f}%')
    print(f'll_iou_max is {YOLOP_data[50:, 4].max()*100:.3f}%')
    print(f'll_miou_max is {YOLOP_data[50:, 5].max()*100:.3f}%')
    print(f'det_p_max is {YOLOP_data[50:, 6].max()*100:.3f}%')
    print(f'det_r_max is {YOLOP_data[50:, 7].max()*100:.3f}%')
    print(f'det_map@0.5_max is {YOLOP_data[50:, 8].max()*100:.3f}%')
    print(f'det_map@0.5:0.95_max is {YOLOP_data[50:, 9].max()*100:.3f}%')
if __name__ == '__main__':
      main()
    