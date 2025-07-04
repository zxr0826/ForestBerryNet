import torch
import torch.nn as nn

from .MB import MB_TaylorFormer
from .CPAEhance import CPA_arch

__all__ = ['TaylorCPA']

class TaylorCPA(nn.Module):
    """
    "Taylor-CPA" 融合模型
      - MB_TaylorFormer: 去雾 (或其他图像复原)
      - CPA_arch       : 自适应增强 (Chain-of-thought Prompt)
    既可保持输入输出的空间尺寸一致(640×640)。
    """
    def __init__(self,
                 in_ch=3, out_ch=3,
                 # MB_TaylorFormer相关参数
                 mb_embed_dim=32,
                 # CPA_arch相关参数
                 cpa_dim=4,
                 cpa_prompt_inch=128,
                 cpa_prompt_size=32
                 ):
        super(TaylorCPA, self).__init__()

        # 1) 去雾/复原：MB_TaylorFormer
        self.dehaze_net = MB_TaylorFormer(
            in_ch=in_ch,
            out_ch=out_ch,
            embed_dim=mb_embed_dim
            # 如果你的MB_TaylorFormer构造函数有更多参数，
            # 记得在此一并传递
        )

        # 2) 自适应增强：CPA_arch
        self.cpa_enh = CPA_arch(
            c_in=out_ch,  
            c_out=out_ch,
            dim=cpa_dim,
            prompt_inch=cpa_prompt_inch,
            prompt_size=cpa_prompt_size
            # 若CPA_arch还有更多自定义参数，也在此一并传递
        )

    def forward(self, x):
        """
        x: (B, in_ch, 640, 640)
        return: (B, out_ch, 640, 640)
        """
        # 第一步：去雾/复原
        x_dehaze = self.dehaze_net(x)

        # 第二步：自适应增强
        x_enh = self.cpa_enh(x_dehaze)

        return x_enh


##############################################################
# 测试脚本
##############################################################
if __name__ == "__main__":
    # 实例化 "TaylorCPA" 模型
    model = TaylorCPA(
        in_ch=3,
        out_ch=3,
        mb_embed_dim=32,
        cpa_dim=4,
        cpa_prompt_inch=128,
        cpa_prompt_size=32
    )

    # 构造一个 (B=1, C=3, H=W=640) 的随机输入
    x_test = torch.randn(1, 3, 640, 640)

    with torch.no_grad():
        y = model(x_test)

    print("输入尺寸:", x_test.shape)
    print("输出尺寸:", y.shape)

    # 如果中间网络保持分辨率不变，则输出应该是 (1,3,640,640)