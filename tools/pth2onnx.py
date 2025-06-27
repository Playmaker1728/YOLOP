import os,sys
import torch
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(BASE)
sys.path.append(BASE)

import onnx

from lib.models import get_net

checkpoint_path = r'E:\Dissertation_Project\YOLOP\weights\End-to-end.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = get_net(None)
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['state_dict'])
model.eval().to(device)

input = torch.randn(1, 3, 192, 608).to(device)
_, _, h, w = input.shape
# onnx_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'yolop-{h}-{w}.onnx')
onnx_path = os.path.join('.\\weights', f'yolop-{h}-{w}.onnx')
with torch.no_grad():
    torch.onnx.export(
        model,
        input,
        onnx_path,
        input_names=['input'],
        output_names=['det_out','da_seg_out', 'll_seg_out'],
        opset_version=12,
        dynamic_axes={
            'input': {0: 'batch', 2: 'height', 3: 'width'}
        },
    )
    print('onnx 导出完成！')
    

onnx_model = onnx.load(onnx_path)
onnx.checker.check_model(onnx_model)
print('无报错onnx模型载入成功!')